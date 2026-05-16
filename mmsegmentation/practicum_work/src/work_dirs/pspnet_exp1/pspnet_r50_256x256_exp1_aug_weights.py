_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/practice_dataset.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/practice_schedule.py'
]

crop_size = (256, 256)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

# ✅ Усиленные аугментации (оставляем)
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
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
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

# Модель (❌ убираем проблемные class_weight)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(depth=50),
    decode_head=dict(
        num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=1.0
                # class_weight удален
            ),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=1.5  # ✅ увеличенный вес оставляем
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
                # class_weight удален
            ),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=0.6  # ✅ увеличенный вес оставляем
            )
        ]
    )
)

# ClearML
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='ClearMLVisBackend',
        init_kwargs=dict(
            project_name='YaPracticum',
            task_name='pspnet_r50_256x256_exp1_aug_weights',
        )
    )
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

randomness = dict(seed=42)