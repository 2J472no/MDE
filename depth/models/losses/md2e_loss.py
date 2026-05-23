import torch
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import LOSSES


@LOSSES.register_module()
class MD2ELoss(nn.Module):
    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None,
                 edge_weight=0.1,
                 sqe_weight=1.0,
                 depth_weight=100.0,
                 edge_consistency_weight=1.0,
                 depth_loss_lambda=0.85,
                 eps=1e-6):
        super().__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.edge_weight = edge_weight
        self.sqe_weight = sqe_weight
        self.depth_weight = depth_weight
        self.edge_consistency_weight = edge_consistency_weight
        self.depth_loss_lambda = depth_loss_lambda
        self.eps = eps

    def depth2edge_score(self, depth):
        inv_depth = 1.0 / depth.clamp_min(self.eps)
        depth_w2 = F.pad(inv_depth, (1, 0, 0, 0))
        depth_w1 = F.pad(inv_depth, (0, 1, 0, 0))
        depth_h2 = F.pad(inv_depth, (0, 0, 1, 0))
        depth_h1 = F.pad(inv_depth, (0, 0, 0, 1))
        depth_rd1 = F.pad(inv_depth, (0, 1, 0, 1))
        depth_ru2 = F.pad(inv_depth, (1, 0, 1, 0))
        depth_ru1 = F.pad(inv_depth, (0, 1, 1, 0))
        depth_lu2 = F.pad(inv_depth, (1, 0, 0, 1))

        dh = depth_h1[:, :, 1:, :] - depth_h2[:, :, :-1, :]
        dw = depth_w1[:, :, :, 1:] - depth_w2[:, :, :, :-1]
        dru = depth_ru1[:, :, :-1, 1:] - depth_lu2[:, :, 1:, :-1]
        drd = depth_rd1[:, :, 1:, 1:] - depth_ru2[:, :, :-1, :-1]

        edge = torch.abs(dh) + torch.abs(dw) + torch.abs(dru) + torch.abs(drd)
        edge = F.unfold(edge, kernel_size=3, padding=1)
        edge = torch.softmax(edge, dim=1)
        edge = edge[:, 4, :] - edge.amin(dim=1)
        return edge.view(depth.shape[0], 1, depth.shape[2], depth.shape[3])

    def depth2edge_binary(self, depth, quantile=0.9):
        edge = self.depth2edge_score(depth)
        flat = edge.flatten(1)
        threshold = torch.quantile(flat, quantile, dim=-1, keepdim=True)
        return (flat >= threshold).float().view_as(edge)

    def class_balanced_edge_loss(self, pred, target, valid):
        pred = pred.clamp(self.eps, 1.0 - self.eps)
        target = target.float().clamp(0.0, 1.0)
        if valid.sum() == 0:
            return pred.sum() * 0.0

        target_valid = target[valid]
        pos = target_valid.sum()
        neg = target_valid.numel() - pos
        theta = neg / (pos + neg + self.eps)

        loss = -(theta * target * torch.log(pred) +
                 (1.0 - theta) * (1.0 - target) * torch.log(1.0 - pred))
        return loss[valid].mean()

    def metric_depth_loss(self, pred, target, valid):
        if valid.sum() == 0:
            return pred.sum() * 0.0
        xi = torch.log(pred[valid].clamp_min(self.eps)) - torch.log(target[valid].clamp_min(self.eps))
        return xi.pow(2).mean() - self.depth_loss_lambda * xi.mean().pow(2)

    def forward(self, depth_pred, depth_gt, depth_edge_gt=None):
        if isinstance(depth_pred, (tuple, list)):
            if len(depth_pred) == 4:
                _, edge_pred, final_depth, sqe = depth_pred
            elif len(depth_pred) == 5:
                _, edge_pred, final_depth, sqe, _ = depth_pred
            else:
                raise ValueError('MD2ELoss expects 4 or 5 outputs from MD2EHead during training.')
        else:
            return self.loss_weight * self.metric_depth_loss(
                depth_pred, depth_gt, self._valid_depth_mask(depth_gt))

        if final_depth.shape[-2:] != depth_gt.shape[-2:]:
            final_depth = F.interpolate(final_depth, depth_gt.shape[-2:],
                                        mode='bilinear', align_corners=False)
        if edge_pred.shape[-2:] != depth_gt.shape[-2:]:
            edge_pred = F.interpolate(edge_pred, depth_gt.shape[-2:],
                                      mode='bilinear', align_corners=False)

        valid_depth = self._valid_depth_mask(depth_gt)
        if depth_edge_gt is None:
            with torch.no_grad():
                depth_edge_gt = self.depth2edge_binary(depth_gt)
        else:
            if depth_edge_gt.shape[-2:] != depth_gt.shape[-2:]:
                depth_edge_gt = F.interpolate(depth_edge_gt, depth_gt.shape[-2:],
                                              mode='nearest')
        valid_edge = valid_depth & (depth_edge_gt >= 0)
        edge_target = depth_edge_gt.clamp(0.0, 1.0)

        loss_edge = self.class_balanced_edge_loss(edge_pred, edge_target, valid_edge)

        valid_float = valid_depth.float()
        t_pred = sqe(edge_pred * valid_float)
        with torch.no_grad():
            target_for_sqe = edge_target * valid_float
        t_gt = sqe(target_for_sqe)
        loss_sqe = F.l1_loss(t_pred, t_gt)

        loss_depth = self.metric_depth_loss(final_depth, depth_gt, valid_depth)

        depth_edge_pred = self.depth2edge_score(final_depth)
        loss_mse = F.mse_loss(depth_edge_pred[valid_depth], edge_pred[valid_depth]) if valid_depth.sum() > 0 else edge_pred.sum() * 0.0

        loss = (self.edge_weight * loss_edge +
                self.sqe_weight * loss_sqe +
                self.depth_weight * loss_depth +
                self.edge_consistency_weight * loss_mse)
        return self.loss_weight * loss

    def _valid_depth_mask(self, depth_gt):
        if not self.valid_mask:
            return torch.ones_like(depth_gt, dtype=torch.bool)
        valid = depth_gt > 0
        if self.max_depth is not None:
            valid = valid & (depth_gt <= self.max_depth)
        return valid
