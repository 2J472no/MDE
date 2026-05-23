import torch.nn as nn
from ..builder import BACKBONES


@BACKBONES.register_module()
class EmptyE(nn.Module):
    def __init__(self, out_index=(4, 5, 6, 8, 11)):
        super().__init__()
        self.out_index = out_index

    def forward(self, x):
        return x
