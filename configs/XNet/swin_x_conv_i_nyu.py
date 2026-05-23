_base_ = ['./swin_x_conv_nyu.py']

model = dict(
    decode_head=dict(
        type='SwinXConvI'))
