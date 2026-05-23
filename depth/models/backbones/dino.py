import torch

from torch import nn

from ..builder import BACKBONES

from ..decode_heads.DINOv2 import vit_large_reg, vit_small_reg

@BACKBONES.register_module()
class DINO1P(nn.Module):
    def __init__(self,
                 # basemodel_name='tf_efficientnet_b5_ap',
                 # out_index=[4, 5, 6, 8, 11]
                 ):
        super(DINO1P, self).__init__()
        m1 = vit_large_reg()
        # state_dict1 = torch.load('./DINO.pth', map_location='cpu')
        # m1.load_state_dict(state_dict1)

        m1.norm = nn.Identity()

        self.m1 = m1

    def forward(self, x):
        x, size, pad = self.m1.patch(x)
        B, C, H, W = size
        pad_h, pad_w = pad
        H = H + pad_h
        W = W + pad_w
        features = self.m1.forward_features(x, size, pad)
        x = features[:, 5:, :]
        x = x.transpose(1, 2).reshape(B, 1024, H // 14, W // 14)

        return x