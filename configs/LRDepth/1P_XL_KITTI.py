_base_ = [
    # '../_base_/models/ehs.py',
    # '../_base_/datasets/nyu.py',
    '../_base_/datasets/kitti.py',
    '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_24x.py'
]
# bash tools/dist_train7.sh 2
backbone_norm_cfg = dict(type='LN', requires_grad=True)

model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='EVA',
        patch_size=(28, 28),
        # img_size=(352, 1216),
    ),
    decode_head=dict(
        type='OneP',
        final_norm=False,
        scale_up=True,
        # in_channels=[128, 256, 512, 1024],
        # group_size=(4, 8, 16, 32),
        # channels=128,
        # in_channels=[512, 768, 2048, 1024],
        # group_size=(4, 8, 32, 16),
        # channels=512,
        # kernel=8,
        in_channels=[32, 64, 1536, 1024],
        group_size=(2, 4, 32, 16),
        channels=32,
        kernel=6,
        # scale_up=True,
        # channels=16,
        min_depth=1e-3,
        max_depth=80,
        loss_decode=dict(
            # type='SigLossScale', valid_mask=True, loss_weight=1.0, max_depth=10,)),
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    )
#
#
# find_unused_parameters=True
SyncBN = True

# batch size
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
)


# #
# # # schedules
# # # optimizer
max_lr = 2.5e-5
# max_lr = 1e-4

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
# optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=16)

# runner = dict(type='IterBasedRunnerAmp', max_iters=320000)
runner = dict(type='IterBasedRunner', max_iters=240000)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=1600)
evaluation = dict(by_epoch=False,
                  start=0,
                  interval=1600,
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