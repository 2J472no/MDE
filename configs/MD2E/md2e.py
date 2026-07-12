_base_ = [
    '../_base_/datasets/md2e.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(type='EmptyE'),
    decode_head=dict(
        type='MD2EHead',
        encoder_variant='large',
        encoder_checkpoint=None,
        channels=32,
        in_channels=[96, 192, 384, 768, 1536],
        min_depth=1e-1,
        max_depth=200,
        scale_up=True,
        loss_decode=dict(
            type='MD2ELoss',
            valid_mask=True,
            max_depth=200,
            edge_weight=0.1,
            sqe_weight=1.0,
            depth_weight=100.0,
            edge_consistency_weight=1.0,
            loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

find_unused_parameters = True
SyncBN = True

max_lr = 1e-4
multi1 = 0.85
lr_dict = {
    'encoder.cls_token': dict(lr_mult=multi1 ** 30),
    'encoder.patch_embed': dict(lr_mult=multi1 ** 30),
    'encoder.register_tokens': dict(lr_mult=multi1 ** 30),
    'token2feature': dict(lr_mult=multi1 ** 5),
    'decoder_mono.upconv_3': dict(lr_mult=multi1 ** 4),
    'decoder_mono.upconv_2': dict(lr_mult=multi1 ** 3),
    'decoder_mono.upconv_1': dict(lr_mult=multi1 ** 2),
}
for i in range(24):
    lr_dict[f'encoder.blocks.{i}'] = dict(lr_mult=multi1 ** (29 - i))

optimizer = dict(
    type='AdamW',
    lr=max_lr,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys=lr_dict))

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=3200,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=2000000)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=3200)
evaluation = dict(
    by_epoch=False,
    start=0,
    interval=3200,
    pre_eval=True,
    rule='less',
    save_best='abs_rel',
    greater_keys=('a1', 'a2', 'a3'),
    less_keys=('abs_rel', 'rmse'))

log_config = dict(
    _delete_=True,
    interval=100,
    hooks=[dict(type='CustomizedTextLoggerHook', by_epoch=False)])
