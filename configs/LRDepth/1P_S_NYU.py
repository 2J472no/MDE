_base_ = [
    '../_base_/models/ehs.py', '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_24x.py'
]
# bash tools/dist_train7.sh 2
model = dict(
    backbone=dict(
        type='ResNet1P',
        init_cfg=dict(
            type='Pretrained',
            # checkpoint='resnet50-0676ba61.pth'),
            checkpoint='torchvision://resnet50'
        ),
    ),
    decode_head=dict(
        type='OneP',
        final_norm=False,
        in_channels=[8, 16, 32, 1024, 2048],
        group_size=(1, 2, 4, 32, 16),
        channels=8,
        scale_up=True,
        min_depth=1e-3,
        max_depth=10,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    )
#
#
# find_unused_parameters=True
SyncBN = True

# batch size
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)


#
# # schedules
# # optimizer
max_lr = 1e-4
# max_lr = 2.5e-5

optimizer = dict(
    type='AdamW',
    lr=max_lr,
    betas=(0.9, 0.999),
    weight_decay=0.,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            # 'backbone': dict(lr_mult=0.1),
            # 'decode_head': dict(lr_mult=10.),
        })
)

lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=3200,
                 warmup_ratio=1e-6,
                 # power=1.0,
                 min_lr=1e-6,
                 by_epoch=False)


optimizer_config = dict()
# optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=4)

# runner = dict(type='IterBasedRunnerAmp', max_iters=320000)
runner = dict(type='IterBasedRunner', max_iters=320000)
# runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=3200)
evaluation = dict(by_epoch=False,
                  start=0,
                  interval=3200,
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