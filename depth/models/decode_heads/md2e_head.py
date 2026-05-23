import torch
import torch.nn as nn
import torch.nn.functional as F

from depth.models.builder import HEADS
from .gt_depth import GTDepthDecodeHead
from .DINOv2_ import vit_large_reg, vit_small_reg


class CvnxtBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, layer_scale=1.0, expansion=4):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        return shortcut + x.permute(0, 3, 1, 2)


class ConvUpsample(nn.Module):
    def __init__(self, hidden_dim, div=2, num_layers=2, expansion=4,
                 layer_scale=1.0, kernel_size=7):
        super().__init__()
        self.convs = nn.ModuleList([
            CvnxtBlock(hidden_dim, kernel_size=kernel_size,
                       expansion=expansion, layer_scale=layer_scale)
            for _ in range(num_layers)
        ])
        out_dim = hidden_dim // div
        self.up = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        )
        self.apply(self.init_weights_)

    @staticmethod
    def init_weights_(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return self.up(x)


class Readout(nn.Module):
    def __init__(self, in_features, use_cls_token=True, num_register_tokens=4):
        super().__init__()
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.project_patch = nn.Linear(in_features, in_features)
            self.project_learn = nn.Linear(
                (1 + num_register_tokens) * in_features, in_features,
                bias=False)
            self.act = nn.GELU()
        else:
            self.project = nn.Identity()

    def forward(self, x):
        if self.use_cls_token:
            x_patch = self.project_patch(x[0])
            x_learn = self.project_learn(x[1]).expand_as(x_patch).contiguous()
            return self.act(x_patch + x_learn)
        return self.project(x)


class Token2Feature(nn.Module):
    def __init__(self, vit_channel, feature_channel, scale_factor,
                 use_cls_token=True, num_register_tokens=4):
        super().__init__()
        self.readoper = Readout(vit_channel, use_cls_token,
                                num_register_tokens)
        if scale_factor == 2:
            self.sample = ConvUpsample(vit_channel, vit_channel // feature_channel)
        elif scale_factor == 4:
            self.sample = nn.Sequential(
                ConvUpsample(vit_channel, vit_channel // feature_channel),
                ConvUpsample(feature_channel, 1),
            )
        else:
            self.sample = nn.Identity()

    def forward(self, x):
        x = self.readoper(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.sample(x)


class EncoderFeature(nn.Module):
    def __init__(self, vit_channel, feature_channels, use_cls_token=True,
                 num_register_tokens=4):
        super().__init__()
        self.read_3 = Token2Feature(vit_channel, feature_channels[3], 1,
                                    use_cls_token, num_register_tokens)
        self.read_2 = Token2Feature(vit_channel, feature_channels[2], 1,
                                    use_cls_token, num_register_tokens)
        self.read_1 = Token2Feature(vit_channel, feature_channels[1], 2,
                                    use_cls_token, num_register_tokens)
        self.read_0 = Token2Feature(vit_channel, feature_channels[0], 4,
                                    use_cls_token, num_register_tokens)

    def forward(self, ref_feature):
        x = self.read_3(ref_feature[3])
        x2 = self.read_2(ref_feature[2])
        x1 = self.read_1(ref_feature[1])
        x0 = self.read_0(ref_feature[0])
        return x, x2, x1, x0


class FuseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fuse=True,
                 upsample=True, scale_factor=2):
        super().__init__()
        self.fuse = fuse
        self.upsample = upsample
        self.scale_factor = scale_factor
        self.way_trunk = CvnxtBlock(in_channels)
        if fuse:
            self.way_branch = CvnxtBlock(in_channels)
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2=None):
        if x2 is not None:
            x1 = x1 + self.way_branch(x2)
        out = self.way_trunk(x1)
        if self.upsample:
            out = F.interpolate(out, scale_factor=self.scale_factor,
                                mode='bilinear', align_corners=False)
        return self.out_conv(out)


class DecoderFeature(nn.Module):
    def __init__(self, decoder_channels):
        super().__init__()
        self.upconv_3 = FuseBlock(decoder_channels[4], decoder_channels[3],
                                  fuse=False, upsample=False)
        self.upconv_2 = FuseBlock(decoder_channels[3], decoder_channels[2])
        self.upconv_1 = FuseBlock(decoder_channels[2], decoder_channels[1])
        self.upconv_0 = FuseBlock(decoder_channels[1], decoder_channels[0])

    def forward(self, ref_feature):
        x, x2, x1, x0 = ref_feature
        x = self.upconv_3(x)
        x = self.upconv_2(x, x2)
        x = self.upconv_1(x, x1)
        x = self.upconv_0(x, x0)
        return x


class DPT(nn.Module):
    def __init__(self, encoder_variant='large', encoder_checkpoint=None):
        super().__init__()
        self.num_register_tokens = 4
        if encoder_variant == 'small':
            self.vit_channel = 384
            self.feature_channels = [96, 192, 384, 384]
            self.decoder_channels = [96, 192, 384, 384, 384]
            self.encoder = vit_small_reg(checkpoint=encoder_checkpoint)
        elif encoder_variant == 'large':
            self.vit_channel = 1024
            self.feature_channels = [256, 512, 1024, 1024]
            self.decoder_channels = [128, 256, 512, 1024, 1024]
            self.encoder = vit_large_reg(checkpoint=encoder_checkpoint)
        else:
            raise ValueError("encoder_variant must be 'small' or 'large'")

        self.token2feature = EncoderFeature(
            self.vit_channel, self.feature_channels, True,
            self.num_register_tokens)
        self.decoder_mono = DecoderFeature(self.decoder_channels)

    def forward(self, x):
        _, _, fh, fw = x.size()
        tokens, size, pad = self.encoder.patch(x)
        B, _, H, W = size
        pad_h, pad_w = pad
        H = H + pad_h
        W = W + pad_w
        features = self.encoder.forward_features(tokens, size, pad)
        H_patch, W_patch = H // 14, W // 14
        vit_features = [
            [ft[:, 1 + self.num_register_tokens:, :].view(
                B, H_patch, W_patch, self.vit_channel),
             ft[:, 0:1 + self.num_register_tokens, :].view(
                B, 1, 1,
                self.vit_channel * (1 + self.num_register_tokens))]
            for ft in features
        ]
        encoder_features = self.token2feature(vit_features)
        ref_feat = self.decoder_mono(encoder_features)
        return F.interpolate(ref_feat, size=(fh // 2, fw // 2),
                             mode='bilinear', align_corners=False)


def _constrain(u, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(u)


def _softfloor(x, floor, beta=0.01):
    return floor + F.softplus((x - floor) / beta) * beta


class SpectralQuantileEstimator(nn.Module):
    def __init__(self, K=256, p_range=(0.35, 0.65),
                 lambda_range=(5.0, 80.0), r0_range=(0.02, 0.10),
                 alpha_floor=0.02, softfloor_beta=0.01,
                 bin_sigma_ratio=1.5, eps=1e-12):
        super().__init__()
        self.K = K
        self.p_lo, self.p_hi = p_range
        self.lam_lo, self.lam_hi = lambda_range
        self.r0_lo, self.r0_hi = r0_range
        self.alpha_floor = alpha_floor
        self.softfloor_beta = softfloor_beta
        self.bin_sigma_ratio = bin_sigma_ratio
        self.eps = eps
        self.p_u = nn.Parameter(torch.tensor(0.0))
        self.lam_u = nn.Parameter(torch.tensor(0.0))
        self.r0_u = nn.Parameter(torch.tensor(0.0))
        self.win_mix_logit = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _freq_grid(H, W, device):
        fy = torch.fft.fftfreq(H, d=1.0).to(device)
        fx = torch.fft.fftfreq(W, d=1.0).to(device)
        fy = torch.fft.fftshift(fy)
        fx = torch.fft.fftshift(fx)
        FY = fy.view(H, 1).expand(H, W)
        FX = fx.view(1, W).expand(H, W)
        return torch.sqrt(FX * FX + FY * FY)

    def _power(self, x):
        _, _, H, W = x.shape
        m = torch.sigmoid(self.win_mix_logit)
        if H > 1 and W > 1:
            wy = torch.hann_window(H, periodic=True, device=x.device)
            wx = torch.hann_window(W, periodic=True, device=x.device)
            hann = wy[:, None] * wx[None, :]
            window = (1 - m) + m * hann
        else:
            window = torch.ones(H, W, device=x.device)
        x = x * window
        x = x - x.mean(dim=(-1, -2), keepdim=True)
        fft = torch.fft.fft2(x, dim=(-2, -1))
        power = fft.real ** 2 + fft.imag ** 2
        return torch.fft.fftshift(power, dim=(-2, -1))

    def forward(self, edge_map):
        x = edge_map.float()
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.size(1) != 1:
            x = x.mean(1, keepdim=True)

        N, _, H, W = x.shape
        device = x.device
        radius = self._freq_grid(H, W, device)
        r_max = radius.max().detach()
        R = radius.view(1, 1, H, W)

        p = _constrain(self.p_u, self.p_lo, self.p_hi)
        lam = _constrain(self.lam_u, self.lam_lo, self.lam_hi)
        r0 = _constrain(self.r0_u, self.r0_lo, self.r0_hi)

        high_pass = 1.0 - torch.exp(- (radius / r0) ** 2)
        power = self._power(x)
        energy = (power * high_pass).sum(1) + self.eps

        bins = torch.linspace(0.0, r_max.item(), self.K, device=device)
        if self.K > 1:
            delta = (bins[1] - bins[0]).detach()
        else:
            delta = torch.tensor(1.0, device=device)
        sigma = (self.bin_sigma_ratio * delta).clamp(min=1e-6)

        weights = torch.exp(-0.5 * ((R - bins.view(1, self.K, 1, 1)) / sigma) ** 2)
        weights = weights / (weights.sum(dim=1, keepdim=True) + self.eps)
        radial_energy = (energy.unsqueeze(1) * weights).sum(dim=(-1, -2))
        pdf = radial_energy / (radial_energy.sum(dim=1, keepdim=True) + self.eps)
        cdf = torch.cumsum(pdf, dim=1)
        quantile_weights = torch.exp(-lam * torch.abs(cdf - p))
        quantile_weights = quantile_weights / (quantile_weights.sum(dim=1, keepdim=True) + self.eps)
        f_p = (quantile_weights * bins.view(1, -1)).sum(dim=1, keepdim=True)
        f_safe = _softfloor(f_p, self.alpha_floor * r_max, self.softfloor_beta)
        return (r_max / (f_safe + self.eps)).to(torch.float32).view(N, 1, 1, 1)


@HEADS.register_module()
class MD2EHead(GTDepthDecodeHead):
    def __init__(self, encoder_variant='large', encoder_checkpoint=None,
                 finetune=False, **kwargs):
        super().__init__(**kwargs)
        self.finetune = finetune
        self.depth_model = DPT(encoder_variant, encoder_checkpoint)
        self.depth_head = nn.Sequential(
            ConvUpsample(128 if encoder_variant == 'large' else 96, 4),
            nn.Conv2d(32 if encoder_variant == 'large' else 24, 1,
                      kernel_size=3, padding=1),
        )
        self.edge_head = nn.Sequential(
            ConvUpsample(128 if encoder_variant == 'large' else 96, 4),
            nn.Conv2d(32 if encoder_variant == 'large' else 24, 1,
                      kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.tim = SpectralQuantileEstimator()

    def forward(self, feature, img_metas, gt=None):
        B, _, H, W = feature.shape
        decoded = self.depth_model(feature)
        edge_pred = self.edge_head(decoded)
        t_pred = self.tim(edge_pred)
        depth_logits = self.depth_head(decoded)
        max_depth = self.max_depth if self.max_depth is not None else 200.0
        depth_pred = max_depth / (1.0 + torch.exp(-depth_logits * t_pred))
        depth_pred = F.interpolate(depth_pred, size=(H, W), mode='bilinear',
                                   align_corners=False)

        if self.finetune:
            return depth_pred

        edge_pred = F.interpolate(edge_pred, size=(H, W), mode='bilinear',
                                  align_corners=False)
        if self.training:
            return feature, edge_pred, depth_pred, self.tim
        return depth_pred
