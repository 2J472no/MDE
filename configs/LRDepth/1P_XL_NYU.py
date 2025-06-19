_base_ = [
    '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='EVA',
        patch_size=(32, 32),
    ),
    decode_head=dict(
        type='OneP',
        final_norm=False,
        in_channels=[64, 128, 2048, 1024],
        group_size=(4, 8, 32, 16),
        channels=64,
        kernel=6,
        scale_up=True,
        min_depth=1e-3,
        max_depth=10,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    )

SyncBN = True

# batch size
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

max_lr = 2.5e-5

optimizer = dict(
    type='AdamW',
    lr=max_lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'blocks': dict(lr_mult=0.1),
        }))

lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=3200,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


optimizer_config = dict()

runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=800)
evaluation = dict(by_epoch=False,
                  start=0,
                  interval=800,
                  pre_eval=True,
                  rule='less',
                  save_best='abs_rel',
                  greater_keys=("a1", "a2", "a3"),
                  less_keys=("abs_rel", "rmse"))

# iter runtime
log_config = dict(
    _delete_=True,
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])