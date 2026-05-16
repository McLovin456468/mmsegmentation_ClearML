_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/practice_dataset.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/practice_schedule.py'
]

# Размер входных изображений
crop_size = (256, 256)

# Предобработка данных
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

# Модель DeepLabV3+ с оптимизированными параметрами
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        depth=50,
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1)
    ),
    decode_head=dict(
        num_classes=3,
        dilations=(1, 6, 12, 18),
        c1_in_channels=256,
        c1_channels=48,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=1.0,
                # Временно убираем class_weight до решения проблемы с CUDA
                # class_weight=[0.3, 5.0, 5.0]
            ),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=1.5
            )
        ]
    ),
    auxiliary_head=dict(
        num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=0.4
                # class_weight=[0.3, 5.0, 5.0]
            ),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=0.6
            )
        ]
    )
)

# ✅ Усиленные аугментации (БЕЗ RandomGaussianBlur)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.2, direction='vertical'),
    dict(type='RandomRotate', prob=0.5, degree=(-30, 30)),
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='RandomCutOut',
         prob=0.3,
         n_holes=(1, 3),
         cutout_ratio=(0.1, 0.2)),
    dict(type='PackSegInputs')  # RandomGaussianBlur удален
]

# Обновляем даталоадер
train_dataloader = dict(
    batch_size=6,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='PracticeDataset',
        data_root='data/practice_dataset',
        data_prefix=dict(
            img_path='img/train',
            seg_map_path='labels/train'),
        pipeline=train_pipeline,
        img_suffix='.jpg',
        seg_map_suffix='.png')
)

# Оптимизируем параметры обучения для DeepLabV3+
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0005, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0))

# ClearML визуализатор
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='ClearMLVisBackend',
        init_kwargs=dict(
            project_name='YaPracticum',
            task_name='deeplabv3plus_r50_256x256_exp2',
        )
    )
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Seed
randomness = dict(seed=42)