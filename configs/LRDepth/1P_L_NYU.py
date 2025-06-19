_base_ = [
    '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py',
]
# bash tools/dist_train7.sh 2
# backbone_norm_cfg = dict(type='LN', requires_grad=True)

model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='SwinTransformer',
        pretrained=r'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
        embed_dims=192,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.5,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        # norm_cfg=backbone_norm_cfg,
        pretrain_style='official',
    ),
    decode_head=dict(
        type='OneP',
        final_norm=False,
        in_channels=[32, 64, 2048, 1536],
        group_size=(2, 4, 32, 16),
        channels=32,
        scale_up=True,
        min_depth=1e-3,
        max_depth=10,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=10)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    )

SyncBN = True

# batch size
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

max_lr = 1e-5

optimizer = dict(
    type='Adam',
    lr=max_lr,
    betas=(0.9, 0.999),
    weight_decay=0.,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
        }))

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
                  interval=800,
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