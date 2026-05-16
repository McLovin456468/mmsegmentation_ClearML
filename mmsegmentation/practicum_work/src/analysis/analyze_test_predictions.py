import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmseg.apis import init_model
from mmseg.utils import register_all_modules
from pathlib import Path
from tqdm import tqdm
from mmseg.registry import DATASETS


def analyze_test_predictions(config_path, checkpoint_path, output_dir):
    """Анализ предсказаний модели на тестовом датасете"""

    print("=" * 50)
    print("АНАЛИЗ КАЧЕСТВА МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 50)

    # Регистрируем все модули
    register_all_modules()

    # Загружаем конфиг
    cfg = Config.fromfile(config_path)
    cfg.launcher = 'none'

    # Инициализируем модель
    print(f"\n Загрузка модели из {checkpoint_path}")
    model = init_model(cfg, checkpoint_path, device='cuda:0')
    model.eval()

    # Создаем тестовый датасет
    test_dataset_cfg = cfg.test_dataloader.dataset
    test_dataset = DATASETS.build(test_dataset_cfg)
    print(f" Тестовый датасет содержит {len(test_dataset)} изображений")

    # Создаем директории для сохранения результатов
    vis_dir = Path(output_dir) / 'visualizations'
    vis_dir.mkdir(exist_ok=True, parents=True)

    # Словари для хранения метрик
    sample_metrics = []
    class_dice_scores = {0: [], 1: [], 2: []}  # 0: bg, 1: cat, 2: dog

    # Цветовая палитра для визуализации
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    class_names = ['background', 'cat', 'dog']

    print("\n Анализируем предсказания...")

    # Анализируем каждый семпл
    for i in tqdm(range(len(test_dataset)), desc="Processing"):
        data = test_dataset[i]

        # Подготавливаем данные для модели в правильном формате
        inputs = data['inputs'].unsqueeze(0).cuda()
        data_samples = [data['data_samples']]

        # Создаем пакет данных в формате, который ожидает модель
        data_batch = {
            'inputs': inputs,
            'data_samples': data_samples
        }

        # Инференс
        with torch.no_grad():
            results = model.test_step(data_batch)

        # Получаем предсказание
        pred_seg = results[0].pred_sem_seg.data.cpu().numpy()[0]
        gt_seg = data['data_samples'].gt_sem_seg.data.cpu().numpy()

        # Считаем Dice для каждого класса
        dice_scores = []
        for class_id in range(3):
            gt_class = (gt_seg == class_id).astype(np.float32)
            pred_class = (pred_seg == class_id).astype(np.float32)

            intersection = (gt_class * pred_class).sum()
            union = gt_class.sum() + pred_class.sum()

            if union > 0:
                dice = (2 * intersection) / union
            else:
                dice = 1.0 if gt_class.sum() == 0 else 0.0

            dice_scores.append(dice)
            class_dice_scores[class_id].append(dice)

        mean_dice = np.mean(dice_scores)
        sample_metrics.append({
            'idx': i,
            'mean_dice': mean_dice,
            'dice_background': dice_scores[0],
            'dice_cat': dice_scores[1],
            'dice_dog': dice_scores[2],
            'inputs': inputs.cpu(),
            'gt_seg': gt_seg,
            'pred_seg': pred_seg
        })

    # Сортируем по качеству
    sample_metrics.sort(key=lambda x: x['mean_dice'])

    # Выводим общую статистику
    print("\n" + "=" * 50)
    print(" ОБЩАЯ СТАТИСТИКА")
    print("=" * 50)

    mean_dice = np.mean([s['mean_dice'] for s in sample_metrics])
    print(f"Средний mDice по тестовой выборке: {mean_dice * 100:.2f}%")

    print("\n Метрики по классам:")
    for class_id, scores in class_dice_scores.items():
        mean_class_dice = np.mean(scores)
        print(f"  {class_names[class_id]}: {mean_class_dice * 100:.2f}%")

    # Сохраняем лучшие и худшие предсказания
    save_predictions(sample_metrics, output_dir, vis_dir, palette, class_names)

    return sample_metrics


def save_predictions(sample_metrics, output_dir, vis_dir, palette, class_names):
    """Сохраняет примеры лучших и худших предсказаний"""

    # Берем по 5 лучших и худших
    num_samples = min(5, len(sample_metrics) // 2)

    # Худшие и лучшие
    worst_samples = sample_metrics[:num_samples]
    best_samples = sample_metrics[-num_samples:]

    print("\n" + "=" * 50)
    print("  СОХРАНЕНИЕ ПРИМЕРОВ")
    print("=" * 50)

    # Средние значения для денормализации
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])

    # Сохраняем худшие примеры
    print("\n ХУДШИЕ ПРЕДСКАЗАНИЯ (lowest Dice):")
    for i, sample_info in enumerate(worst_samples):
        print(f"\n{i + 1}. Индекс {sample_info['idx']}:")
        print(f"   mDice = {sample_info['mean_dice'] * 100:.2f}%")
        print(f"   background: {sample_info['dice_background'] * 100:.2f}%, "
              f"cat: {sample_info['dice_cat'] * 100:.2f}%, "
              f"dog: {sample_info['dice_dog'] * 100:.2f}%")

        # Извлекаем данные
        img = sample_info['inputs'].numpy()[0].transpose(1, 2, 0)
        gt_seg = sample_info['gt_seg'].squeeze()
        pred_seg = sample_info['pred_seg']

        # Денормализация
        img_display = img * std + mean
        img_display = np.clip(img_display, 0, 255).astype(np.uint8)

        # Создаем цветные маски
        gt_color = palette[gt_seg]
        pred_color = palette[pred_seg]

        # Визуализация
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_display)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        axes[1].imshow(gt_color)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(pred_color)
        axes[2].set_title(f'Prediction (Dice: {sample_info["mean_dice"] * 100:.1f}%)')
        axes[2].axis('off')

        plt.suptitle(f'Worst Sample #{i + 1} (Index: {sample_info["idx"]})')
        plt.tight_layout()
        plt.savefig(vis_dir / f'worst_{i + 1}_idx{sample_info["idx"]}.png',
                    bbox_inches='tight', dpi=150)
        plt.close()

    # Сохраняем лучшие примеры
    print("\n ЛУЧШИЕ ПРЕДСКАЗАНИЯ (highest Dice):")
    for i, sample_info in enumerate(reversed(best_samples)):
        print(f"\n{i + 1}. Индекс {sample_info['idx']}:")
        print(f"   mDice = {sample_info['mean_dice'] * 100:.2f}%")
        print(f"   background: {sample_info['dice_background'] * 100:.2f}%, "
              f"cat: {sample_info['dice_cat'] * 100:.2f}%, "
              f"dog: {sample_info['dice_dog'] * 100:.2f}%")

        # Извлекаем данные
        img = sample_info['inputs'].numpy()[0].transpose(1, 2, 0)
        gt_seg = sample_info['gt_seg'].squeeze()
        pred_seg = sample_info['pred_seg']

        # Денормализация
        img_display = img * std + mean
        img_display = np.clip(img_display, 0, 255).astype(np.uint8)

        # Создаем цветные маски
        gt_color = palette[gt_seg]
        pred_color = palette[pred_seg]

        # Визуализация
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_display)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        axes[1].imshow(gt_color)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(pred_color)
        axes[2].set_title(f'Prediction (Dice: {sample_info["mean_dice"] * 100:.1f}%)')
        axes[2].axis('off')

        plt.suptitle(f'Best Sample #{i + 1} (Index: {sample_info["idx"]})')
        plt.tight_layout()
        plt.savefig(vis_dir / f'best_{i + 1}_idx{sample_info["idx"]}.png',
                    bbox_inches='tight', dpi=150)
        plt.close()

    # Создаем сводную таблицу метрик
    create_summary_table(sample_metrics, class_names, output_dir)

    print(f"\n Визуализации сохранены в {vis_dir}")


def create_summary_table(sample_metrics, class_names, output_dir):
    """Создает сводную таблицу метрик"""

    import pandas as pd

    # Создаем DataFrame без тензоров
    df_data = []
    for s in sample_metrics:
        df_data.append({
            'Index': s['idx'],
            'mDice': s['mean_dice'] * 100,
            class_names[0]: s['dice_background'] * 100,
            class_names[1]: s['dice_cat'] * 100,
            class_names[2]: s['dice_dog'] * 100
        })

    df = pd.DataFrame(df_data)
    df = df.round(2)

    # Сохраняем в CSV
    csv_path = Path(output_dir) / 'test_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f" Метрики сохранены в {csv_path}")

    # Создаем текстовый отчет
    txt_path = Path(output_dir) / 'test_report.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ОТЧЕТ ПО ТЕСТИРОВАНИЮ МОДЕЛИ\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Всего проанализировано семплов: {len(df)}\n\n")

        f.write("Общие метрики:\n")
        f.write(f"  Средний mDice: {df['mDice'].mean():.2f}%\n")
        f.write(f"  Медианный mDice: {df['mDice'].median():.2f}%\n")
        f.write(f"  Std mDice: {df['mDice'].std():.2f}%\n")
        f.write(f"  Min mDice: {df['mDice'].min():.2f}%\n")
        f.write(f"  Max mDice: {df['mDice'].max():.2f}%\n\n")

        f.write("Метрики по классам:\n")
        for class_name in class_names:
            f.write(f"  {class_name}: {df[class_name].mean():.2f}% ± {df[class_name].std():.2f}%\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"ТОП-5 ХУДШИХ ПРЕДСКАЗАНИЙ\n")
        f.write("=" * 60 + "\n")
        worst_df = df.nsmallest(5, 'mDice')[['Index', 'mDice', class_names[0], class_names[1], class_names[2]]]
        f.write(worst_df.to_string(index=False))

        f.write("\n\n" + "=" * 60 + "\n")
        f.write(f"ТОП-5 ЛУЧШИХ ПРЕДСКАЗАНИЙ\n")
        f.write("=" * 60 + "\n")
        best_df = df.nlargest(5, 'mDice')[['Index', 'mDice', class_names[0], class_names[1], class_names[2]]]
        f.write(best_df.to_string(index=False))

    print(f" Отчет сохранен в {txt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (best_mDice_epoch_85.pth)')  # обновлено
    parser.add_argument('--output-dir', type=str, default='test_analysis_results_v3_85epoch',  # обновлено
                        help='Output directory')

    args = parser.parse_args()

    metrics = analyze_test_predictions(
        args.config,
        args.checkpoint,
        args.output_dir
    )