
import mmcv
import copy
import torch

import numpy as np
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from depth.ops import resize
from depth.models.builder import build_loss


class GTDepthDecodeHead(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 in_channels,
                 channels=96,
                 conv_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 loss_decode=dict(
                     type='SigLoss',
                     valid_mask=True,
                     loss_weight=10),
                 sampler=None,
                 align_corners=False,
                 min_depth=1e-3,
                 max_depth=None,
                 norm_cfg=None,
                 classify=False,
                 n_bins=256,
                 bins_strategy='UD',
                 norm_strategy='linear',
                 scale_up=False,
                 ):
        super(GTDepthDecodeHead, self).__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.loss_decode = build_loss(loss_decode)
        self.align_corners = align_corners
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.norm_cfg = norm_cfg
        self.classify = classify
        self.n_bins = n_bins
        self.scale_up = scale_up


        self.fp16_enabled = False
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def extra_repr(self):
        """Extra repr."""
        s = f'align_corners={self.align_corners}'
        return s

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass
    
    @auto_fp16()
    @abstractmethod
    def forward(self, inputs, img_metas):
        """Placeholder of forward function."""
        pass

    def forward_train(self, img, inputs, img_metas, depth_gt, train_cfg, **kwargs):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            depth_gt (Tensor): GT depth
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        edge_map = kwargs.get("edge_map", None)
        depth_pred = self.forward(inputs, img_metas, depth_gt)
        losses = self.losses(depth_pred, depth_gt, edge_map)
        if torch.isnan(losses['loss_depth']):
            losses['loss_depth'] = torch.nan_to_num(losses['loss_depth'])
            # losses['loss_depth'] = torch.zeros_like(losses['loss_depth'], device=losses['loss_depth'].device,
            #                                         dtype=losses['loss_depth'].dtype)

        # log_imgs = self.log_images(img[0], depth_pred[0], depth_gt[0], img_metas[0])
        # losses.update(**log_imgs)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `depth/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output depth map.
        """
        return self.forward(inputs, img_metas)

    @force_fp32(apply_to=('depth_pred', ))
    def losses(self, depth_pred, depth_gt, edge_map):
        """Compute depth loss."""
        loss = dict()
        # depth_pred = resize(
        #     input=depth_pred,
        #     size=depth_gt.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners,
        #     warning=False)
        loss['loss_depth'] = self.loss_decode(
            depth_pred,
            depth_gt, edge_map
        )
        return loss

    def log_images(self, img_path, depth_pred, depth_gt, img_meta):
        show_img = copy.deepcopy(img_path.detach().cpu().permute(1, 2, 0))
        show_img = show_img.numpy().astype(np.float32)
        show_img = mmcv.imdenormalize(show_img, 
                                      img_meta['img_norm_cfg']['mean'],
                                      img_meta['img_norm_cfg']['std'], 
                                      img_meta['img_norm_cfg']['to_rgb'])
        show_img = np.clip(show_img, 0, 255)
        show_img = show_img.astype(np.uint8)
        show_img = show_img[:, :, ::-1]
        show_img = show_img.transpose(0, 2, 1)
        show_img = show_img.transpose(1, 0, 2)

        depth_pred = depth_pred / torch.max(depth_pred)
        depth_gt = depth_gt / torch.max(depth_gt)

        depth_pred_color = copy.deepcopy(depth_pred.detach().cpu())
        depth_gt_color = copy.deepcopy(depth_gt.detach().cpu())

        return {"img_rgb": show_img, "img_depth_pred": depth_pred_color, "img_depth_gt": depth_gt_color}