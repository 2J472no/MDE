import torch
import torch.nn as nn

import torch.nn.functional as F

from depth.models.builder import HEADS
from .decode_head import DepthBaseDecodeHead
from timm.layers.drop import DropPath
from timm.layers.norm import LayerNorm2d


class block(nn.Module):
    def __init__(self, in_channels,
                 hid_channels=None, num=7, group_size=1, out_channels=None):
        super(block, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if hid_channels is None:
            hid_channels = in_channels//2


        self.conv_mlp = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(in_channels, hid_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hid_channels, out_channels, kernel_size=1)
        )

        self.in_channels = in_channels
        self.groups = in_channels // group_size
        if self.groups <= 0:
            self.groups = 1
        self.num = num

        drop_rate = 0.

        if self.num > 1:
            for i in range(self.num):
                self.add_module(f'conv{2 * i + 1}',
                                nn.Sequential(
                                    # DropPath(drop_rate),
                                    # nn.Dropout2d(drop_rate),
                                    nn.Dropout2d(drop_rate),
                                    nn.GroupNorm(self.groups, self.in_channels),
                                    # DropPath(drop_rate),
                                    nn.Conv2d(self.in_channels, self.in_channels, kernel_size=2 * i + 1, padding=i,
                                              groups=self.groups
                                              )
                                )
                                )

        self.act = nn.GELU()

    def get_equivalent_kernel_bias(self):
        kernel = 0.
        bias = 0.
        if self.num > 1:
            for i in range(self.num):
                conv_weight, conv_bias = (getattr(self, f'conv{2 * i + 1}')[2].weight.data,
                                          getattr(self, f'conv{2 * i + 1}')[2].bias.data)
                norm_weight, norm_bias = (getattr(self, f'conv{2 * i + 1}')[1].weight.data,
                                          getattr(self, f'conv{2 * i + 1}')[1].bias.data)

                # ======================================================================================================
                norm_weight = norm_weight.view(self.gs, -1).transpose(0, 1).repeat(self.gs, 1).unsqueeze(-1).unsqueeze(-1)
                norm_bias = norm_bias.view(self.gs, -1).transpose(0, 1).repeat(
                    self.gs, 1).unsqueeze(-1).unsqueeze(-1)
                kernel += F.pad(conv_weight, (self.num - i - 1,) * 4) * norm_weight
                bias += conv_bias + torch.sum(
                    (conv_weight * norm_bias).view(conv_bias.size(0), -1), -1)
                # ======================================================================================================
                # Even though this re-parameterization method is inconsistent with the model in train mode,
                # the results are still very close.
                # kernel += F.pad(conv_weight, (self.num - i - 1,) * 4) * norm_weight.view(norm_weight.size(0), 1, 1, 1)
                # bias += conv_bias + torch.sum(
                #     (conv_weight * norm_bias.view(norm_weight.size(0), 1, 1, 1)).view(conv_bias.size(0), -1), -1)
                # ======================================================================================================

        return kernel, bias

    def train(self, mode: bool = True):
        if mode == True:
            if hasattr(self, 'rp_conv'):
                delattr(self, 'rp_conv')
        else:
            if self.num > 1:
                self.rp_conv = nn.Sequential(
                    nn.GroupNorm(self.groups, self.in_channels),
                    nn.Conv2d(self.in_channels, self.in_channels,
                              groups=self.groups,
                              kernel_size=self.num * 2 - 1,
                              padding=self.num - 1)).to(self.conv1[1].weight.device)
                self.rp_conv[1].weight.data, self.rp_conv[1].bias.data = self.get_equivalent_kernel_bias()
            else:
                self.rp_conv = nn.Identity()
                
        super().train(mode)

    def forward(self, x):

        if self.training:
            out = 0.
            if self.num > 1:
                for i in range(self.num):
                    out = out + getattr(self, f'conv{2 * i + 1}')(x)
            else:
                out = x
        else:
            out = self.rp_conv(x)

        return self.conv_mlp(out)


class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, up=True):
        super().__init__()

        self.conv = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))

    def forward(self, x):
        return self.conv(x)

@HEADS.register_module()
class OneP(DepthBaseDecodeHead):
    """From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation
    This head is implemented of `BTS: <https://arxiv.org/abs/1907.10326>`_.
    Args:
        final_norm (int): Whether apply norm on the final depth prediction.
            Used to handle different focal lens in datasets.
            Default: False. (False in NYU, True in KITTI)
        num_features (List): Out channels of decoder layers.
            While it can be re-written to 'up_sample_channels' as DenseDepth
            head, we follow the original design.
            Default: 512.

    """

    # channels
    def __init__(self, final_norm,
                 group_size=(2, 2, 2, 8, 16),
                 kernel=None,
                 **kwargs):
        super(OneP, self).__init__(**kwargs)

        feat_out_channels = self.in_channels
        self.final_norm = final_norm

        self.stage = len(group_size) + 1
        kernel = kernel if kernel is not None else self.stage + 1
        for i in range(1, self.stage):
            self.add_module(f'block{i}', block(
                self.in_channels[-i],
                group_size=group_size[-i],
                num=kernel-i,
            ))

            if i != self.stage-1:
                self.add_module(f'up{i}', upsample(
                    self.in_channels[-i],
                    self.in_channels[-i-1],
                ))

    def forward(self, features, img_metas):
        x = features
        for i in range(1, self.stage):
            x = getattr(self, f'block{i}')(x)

            if i != self.stage-1:
                x = getattr(self, f'up{i}')(x)

        output = self.depth_pred(x)
        return output


