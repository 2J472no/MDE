dataset_type = 'CustomDepthDataset'
data_root = 'data/md2e/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='DepthLoadAnnotations'),
    dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ColorAug', prob=0.5,
         gamma_range=[0.9, 1.1],
         brightness_range=[0.75, 1.25],
         color_range=[0.9, 1.1]),
    dict(type='MD2EDepth2Edge', quantile=0.9),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'depth_gt', 'edge_map'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor',
                    'flip', 'flip_direction', 'img_norm_cfg')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect',
         keys=['img'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor',
                    'flip', 'flip_direction', 'img_norm_cfg')),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=False,
        depth_scale=1,
        pipeline=train_pipeline,
        min_depth=1e-3,
        max_depth=200),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        depth_scale=1,
        pipeline=test_pipeline,
        min_depth=1e-3,
        max_depth=200),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        depth_scale=1,
        pipeline=test_pipeline,
        min_depth=1e-3,
        max_depth=200))
