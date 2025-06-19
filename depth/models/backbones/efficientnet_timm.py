import torch

from torch import nn

from ..builder import BACKBONES
import timm

@BACKBONES.register_module()
class EfficientNet1PTimm(nn.Module):
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
                 ):
        super(EfficientNet1PTimm, self).__init__()
        basemodel_name = 'efficientnet_b5'
        # basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch',
        #                            basemodel_name, pretrained=True,
        #                            drop_connect_rate=0.2,
        #                            drop_rate=0.4
        #                            )
        basemodel = timm.create_model(basemodel_name, pretrained=True,
                                      drop_rate=0.4,
                                      drop_path_rate=0.2
                                      )
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        self.original_model = basemodel
        # self.out_index = out_index

    def forward(self, x):
        # features = [x]
        # for k, v in self.original_model._modules.items():
        #     if (k == 'blocks'):
        #         for ki, vi in v._modules.items():
        #             x = vi(x)
        #     else:
        #         x = v(x)
        x = self.original_model(x)
        return x