_base_ = [
    '../_base_/datasets/kitti.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(type='EmptyE'),
    decode_head=dict(
        type='DINOxConv',
        in_channels=0,
        channels=96,
        scale_up=True,
        min_depth=1e-3,
        max_depth=80.0,
        dino_variant='large',
        dino_checkpoint='checkpoints/dinov2_vitl14_reg4_pretrain.pth',
        decoder_name='convnextv2_base.fcmae_ft_in22k_in1k_384',
        decoder_pretrained=True,
        loss_decode=dict(type='SigLoss', valid_mask=True, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

SyncBN = True
find_unused_parameters = True

data_root = 'data/kitti'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(data_root=data_root),
    val=dict(data_root=data_root),
    test=dict(data_root=data_root))

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={'m1': dict(lr_mult=0.01), 'm2': dict(lr_mult=0.1)}))

lr_config = dict(policy='poly', warmup='linear', warmup_iters=800,
                 warmup_ratio=1e-6, power=1.0, min_lr=0.0, by_epoch=False)
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=800)
evaluation = dict(by_epoch=False, start=0, interval=800, pre_eval=True,
                  rule='less', save_best='abs_rel',
                  greater_keys=('a1', 'a2', 'a3'), less_keys=('abs_rel', 'rmse'))
log_config = dict(_delete_=True, interval=50,
                  hooks=[dict(type='TextLoggerHook', by_epoch=False)])
