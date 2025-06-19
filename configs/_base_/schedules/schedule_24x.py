# optimizer
# max_lr=2.50e-5
max_lr=1e-4
# optimizer = dict(type='Adam',
#                  lr=max_lr, betas=(0.9, 0.999),
#                  # weight_decay=0.01,
#                     paramwise_cfg=dict(
#                             custom_keys={
#                                 'decode_head': dict(lr_mult=10.),
#                             })
#                  )

optimizer = dict(
    type='Adam',
    lr=max_lr,
    betas=(0.9, 0.999),
    # weight_decay=0.1,
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'absolute_pos_embed': dict(decay_mult=0.),
    #         'relative_position_bias_table': dict(decay_mult=0.),
    #         'norm': dict(decay_mult=0.),
    #         # 'backbone': dict(lr_mult=0.1),
    #         # 'decode_head': dict(lr_mult=4.),
    #     })
)

lr_config = dict(policy='poly',
                 # warmup='linear',
                 # warmup_iters=3200,
                 # warmup_ratio=1e-6,
                 # power=1.,
                 min_lr=1e-6,
                 by_epoch=True)
# learning policy
# lr_config = dict(
#     policy='OneCycle',
#     max_lr=max_lr,
#     div_factor=25,
#     final_div_factor=100,
#     by_epoch=False,
# )
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(by_epoch=True, max_keep_ckpts=2, interval=1)
# evaluation = dict(by_epoch=True, interval=6, pre_eval=True)
evaluation = dict(by_epoch=True,
                  # start=0,
                  interval=1,
                  pre_eval=True,
                  rule='less',
                  save_best='abs_rel',
                  greater_keys=("a1", "a2", "a3"),
                  less_keys=("abs_rel", "rmse"))