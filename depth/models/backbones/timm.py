import torch

from torch import nn

from ..builder import BACKBONES
import timm

@BACKBONES.register_module()
class Backbone1P(nn.Module):
    def __init__(self,
                 basemodel_name='tf_efficientnet_b4.ns_jft_in1k',):
        super(Backbone1P, self).__init__()
        self.backbone = timm.create_model(basemodel_name,
                                          drop_rate=0.4,
                                          drop_path_rate=0.2,
                                          pretrained=True)

        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        x = self.backbone.forward_features(x)

        return x