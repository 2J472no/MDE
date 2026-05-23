_base_ = [
    '../_base_/datasets/nyu.py',
    '../_base_/default_runtime.py',
]

model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(type='EmptyE'),
    decode_head=dict(
        type='SwinXConv',
        in_channels=0,
        channels=96,
        scale_up=True,
        min_depth=1e-3,
        max_depth=10.0,
        img_size=(480, 640),
        encoder_name='swin_large_patch4_window7_224.ms_in22k',
        decoder_name='convnext_base.fb_in22k_ft_in1k_384',
        encoder_pretrained=True,
        decoder_pretrained=True,
        loss_decode=dict(type='SigLoss', valid_mask=True, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

SyncBN = True
find_unused_parameters = True

data = dict(samples_per_gpu=2, workers_per_gpu=4)

multi1 = 0.95
multi2 = 0.8
blocks = (2, 2, 18, 2)
block_sum = sum(blocks)
count1 = 0
lr_dict = {'m1.patch_embed': dict(lr_mult=multi1 ** block_sum)}
for i in range(4):
    for j in range(blocks[i]):
        lr_dict[f'm1.layers.{i}.blocks.{j}'] = dict(lr_mult=multi1 ** (block_sum - count1))
        count1 += 1
blocks = (3, 3, 27, 3)
block_sum = sum(blocks)
count2 = 0
for i in range(4):
    for j in range(blocks[i]):
        lr_dict[f'm2.stages.{i}.blocks.{j}'] = dict(lr_mult=multi2 ** (block_sum - count2))
        count2 += 1

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys=lr_dict))

lr_config = dict(policy='poly', warmup='linear', warmup_iters=1600,
                 warmup_ratio=1e-6, power=1.0, min_lr=0.0, by_epoch=False)
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=800)
evaluation = dict(by_epoch=False, start=0, interval=800, pre_eval=True,
                  rule='less', save_best='abs_rel',
                  greater_keys=('a1', 'a2', 'a3'), less_keys=('abs_rel', 'rmse'))
log_config = dict(_delete_=True, interval=50,
                  hooks=[dict(type='TextLoggerHook', by_epoch=False)])
