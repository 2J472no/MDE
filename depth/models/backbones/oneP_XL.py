import torch

from torch import nn

from ..builder import BACKBONES
import timm
import numpy as np
import torch.nn.functional as F

@BACKBONES.register_module()
class XLBackbone(nn.Module):
    """EfficientNet backbone.
    Following Adabins, this is a hack version of EfficientNet, where potential bugs exist.

    I recommend to utilize mmcls efficientnet presented at https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b0_8xb32_in1k.py

    Args:
        basemodel_name (str): Name of pre-trained EfficientNet. Default: tf_efficientnet_b5_ap.
        out_index List(int): Output from which stages. Default: [4, 5, 6, 8, 11].

    """
    def __init__(self,
                 # basemodel_name='tf_efficientnet_b5_ap',
                 # out_index=[4, 5, 6, 8, 11]
                 # img_size=
                 patch_size=16
                 ):
        super(XLBackbone, self).__init__()
        # basemodel_name = 'eva02_large_patch14_224.mim_in22k'
        basemodel_name = 'eva02_large_patch14_224.mim_m38m'

        self.patch_size = 22
        basemodel = timm.create_model(basemodel_name, pretrained=True,
                                      drop_path_rate=0.5,
                                      dynamic_img_size=True,
                                      dynamic_img_pad=True,
                                      patch_size=self.patch_size,
                                      # class_token=False
                                      )
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        basemodel.fc_norm = nn.Identity()
        basemodel.head = nn.Identity()
        # basemodel.cls_token = None
        # basemodel.pos_embed = nn.Parameter(basemodel.pos_embed[:, :-1, :])


        self.original_model = basemodel
        # self.out_index = out_index

    def forward(self, x):
        B, _, H, W = x.shape
        # size = (int(H/32*14), int(W/32*14))
        # x = F.interpolate(x, size=size, mode='bicubic')

        x = self.original_model.forward_features(x)
        # print(x.shape)
        x = x[:, :-1, :].view(B, int(np.ceil(H/self.patch_size)), int(np.ceil(W/self.patch_size)), -1).permute(0, 3, 1, 2)
        # x = x[:, :-1, :].view(B, size[0]//self.patch_size, size[1]//self.patch_size, -1).permute(0, 3, 1, 2)
        # x = torch.pixel_unshuffle(x, 2)


        return x