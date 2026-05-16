_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/practice_dataset.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/practice_schedule.py'
]

# Размер входных изображений
crop_size = (256, 256)

# Предобработка данных - добавляем size
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)  # ВАЖНО: добавляем size сюда

# Модель
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        depth=50
    ),
    decode_head=dict(
        num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=1.0
            ),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=1.0
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
            ),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=0.4
            )
        ]
    )
)

# ClearML визуализатор
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(  # закомментируем ClearML пока не решим проблему с SSL
        type='ClearMLVisBackend',
        init_kwargs=dict(
            project_name='YaPracticum',
            task_name='pspnet_r50_256x256_baseline_final',
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