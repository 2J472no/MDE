import torch

from torch import nn
from mmcv.runner import BaseModule

from ..builder import BACKBONES
import timm

@BACKBONES.register_module()
class Swin1P(BaseModule):
    def __init__(self,
                 img_size=(480, 640)
                 ):
        super(Swin1P, self).__init__()
        basemodel = timm.create_model(
            # 'swin_large_patch4_window7_224.ms_in22k',
            #                           'swin_large_patch4_window12_384.ms_in22k',
                                      'swin_large_patch4_window12_384.ms_in22k_ft_in1k',
                                      img_size=img_size,
                                      pretrained=True,
                                      # drop_path_rate=0.5,
                                      )
        basemodel.norm = nn.Identity()
        basemodel.head = nn.Identity()

        self.original_model = basemodel

    def forward(self, x):
        return self.original_model(x).permute(0, 3, 1, 2)
