# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(
        type='ResNetH',
        depth=50,
        num_stages=4,
        # out_indices=(0, 1, 2, 3, 4),
        # out_indices=(1, 2, 3, 4),
        out_indices=(0, 1, 2, 3),
        style='pytorch',
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='EHeads',
        scale_up=True,
        # in_channels=[64, 256, 512, 1024, 2048],
        in_channels=[256, 512, 1024, 2048],
        # channels=32, # last one
        channels=1, # last one
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
