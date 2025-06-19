_base_ = [
    '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py',
]
# bash tools/dist_train7.sh 2
model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='Backbone1P',
    ),
    decode_head=dict(
        type='IOS',
        final_norm=False,
        in_channels=[192, 384, 512, 768],
        group_size=(4, 8, 32, 16),
        scale_up=True,
        min_depth=1e-3,
        # max_depth=255,
        max_depth=10.,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    )

# find_unused_parameters=True
SyncBN = True

# batch size
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)


max_lr = 1e-5

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
        })
)

lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=800,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


optimizer_config = dict()

runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=800)
evaluation = dict(by_epoch=False,
                  start=0,
                  interval=8000,
                  pre_eval=True,
                  rule='less',
                  save_best='abs_rel',
                  greater_keys=("a1", "a2", "a3"),
                  less_keys=("abs_rel", "rmse"))

# iter runtime
log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])