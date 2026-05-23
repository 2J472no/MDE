_base_ = ['./dino_x_conv_nyu.py']

model = dict(
    decode_head=dict(
        type='DINOxConvI'))

data = dict(samples_per_gpu=1, workers_per_gpu=4)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=400)
evaluation = dict(by_epoch=False, start=0, interval=400, pre_eval=True,
                  rule='less', save_best='abs_rel',
                  greater_keys=('a1', 'a2', 'a3'), less_keys=('abs_rel', 'rmse'))
