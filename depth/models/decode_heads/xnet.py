import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from depth.models.builder import HEADS
from .decode_head import DepthBaseDecodeHead
from .DINOv2 import vit_large_reg, vit_small_reg

try:
    from timm.layers.norm import RmsNorm
except Exception:
    RmsNorm = nn.LayerNorm


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class ReOrderBLC(nn.Module):
    """RS module for token features in BLC format.

    It follows the paper's Re-Shape module: normalization, two linear layers,
    and PixelShuffle between the linear projections to reshape low-resolution
    encoder tokens to a resolution usable by the reversed pre-trained decoder.
    """

    def __init__(self, in_channels, out_channels, hidden_channels=None, up_factor=4, num_extra_tokens=5):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = up_factor ** 2 * 4 if up_factor is not None else (in_channels + out_channels) // 2
        self.up_factor = up_factor
        self.num_extra_tokens = num_extra_tokens
        self.norm = RmsNorm(in_channels)
        self.proj1 = nn.Linear(in_channels, hidden_channels)
        self.resize = nn.PixelShuffle(up_factor) if up_factor is not None and up_factor > 1 else nn.Identity()
        if up_factor is not None and up_factor > 1:
            hidden_channels = hidden_channels // (up_factor ** 2)
        self.proj2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, height, width):
        x = x[:, self.num_extra_tokens:, :]
        x = self.norm(x)
        x = F.silu(self.proj1(x))
        if self.up_factor is not None and self.up_factor > 1:
            batch, _, channels = x.shape
            x = x.transpose(1, 2).reshape(batch, channels, height, width)
            x = self.resize(x)
            x = x.reshape(
                batch,
                channels // (self.up_factor ** 2),
                height * width * (self.up_factor ** 2),
            ).transpose(1, 2)
        return self.proj2(x)


class ReOrderBHWC(nn.Module):
    """RS module for convolutional features.

    Input can be BCHW or BHWC. Output is BHWC, matching the PFP concatenation
    path used by xNet.
    """

    def __init__(self, in_channels, out_channels, hidden_channels=None, up_factor=1):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = up_factor ** 2 * 4 if up_factor is not None else (in_channels + out_channels) // 2
        self.up_factor = up_factor
        self.norm = RmsNorm(in_channels)
        self.proj1 = nn.Linear(in_channels, hidden_channels)
        self.resize = nn.PixelShuffle(up_factor) if up_factor is not None and up_factor > 1 else nn.Identity()
        if up_factor is not None and up_factor > 1:
            hidden_channels = hidden_channels // (up_factor ** 2)
        self.proj2 = nn.Linear(hidden_channels, out_channels)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, bchw=True):
        if bchw:
            x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = F.silu(self.proj1(x))
        x = self.resize(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.proj2(x)


def _load_dino_checkpoint(model, checkpoint):
    if checkpoint is None:
        return
    state_dict = torch.load(checkpoint, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        if key == 'mask_token':
            continue
        cleaned[key] = value

    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError:
        remapped = {}
        for key, value in cleaned.items():
            if key.startswith('blocks.') and not key.startswith('blocks.0.'):
                key = 'blocks.0' + key[len('blocks'):]
            remapped[key] = value
        model.load_state_dict(remapped, strict=False)


def _make_dino(variant, checkpoint, multi_output=True):
    if variant == 'large':
        model = vit_large_reg(checkpoint=None, multi_output=multi_output)
        channels = 1024
    elif variant == 'small':
        model = vit_small_reg(checkpoint=None, multi_output=multi_output)
        channels = 384
    else:
        raise ValueError(f'Unsupported dino_variant: {variant}')
    _load_dino_checkpoint(model, checkpoint)
    model.norm = nn.Identity()
    return model, channels


def _disable_classifier(model):
    for attr in ('head', 'fc'):
        if hasattr(model, attr):
            setattr(model, attr, nn.Identity())
    for attr in ('norm', 'norm_pre'):
        if hasattr(model, attr):
            setattr(model, attr, nn.Identity())


class _XNetPFPHead(DepthBaseDecodeHead):
    def _run_convnext_pfp(self, x):
        pfp = []
        x1 = self.m2.stages[0](x)
        x1 = GradMultiply.apply(x1, self.decoder_grad_scale)
        pfp.append(self.rbx1(x1, bchw=True))

        x2 = self.m2.stages[1](x1)
        x2 = GradMultiply.apply(x2, self.decoder_grad_scale)
        pfp.append(self.rbx2(x2, bchw=True))

        x3 = self.m2.stages[2](x2)
        x3 = GradMultiply.apply(x3, self.decoder_grad_scale)
        pfp.append(self.rbx3(x3, bchw=True))

        x4 = self.m2.stages[3](x3)
        x4 = GradMultiply.apply(x4, self.decoder_grad_scale)
        pfp.append(self.rbx4(x4, bchw=True))

        x = torch.cat(pfp, dim=3)
        x = self.rbf(x, bchw=False).permute(0, 3, 1, 2).contiguous()
        return self.depth_pred(x)


@HEADS.register_module()
class DINOxConv(_XNetPFPHead):
    """xNet for MDE: DINOv2 encoder x ConvNeXt/ConvNeXtV2 pre-trained decoder.

    Encoded DINO tokens are reshaped by RS and multiplied with the decoder stem
    features. The four ConvNeXt decoder stages are then reshaped by PFP to
    produce pixel-level depth.
    """

    def __init__(self,
                 dino_variant='large',
                 dino_checkpoint='checkpoints/dinov2_vitl14_reg4_pretrain.pth',
                 decoder_name='convnextv2_base.fcmae_ft_in22k_in1k_384',
                 decoder_pretrained=True,
                 decoder_drop_path_rate=0.3,
                 decoder_channels=(128, 256, 512, 1024),
                 pfp_channels=128,
                 token_hidden_channels=1024,
                 token_up_factor=4,
                 decoder_grad_scale=0.25,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_size = 14
        self.token_up_factor = token_up_factor
        self.decoder_grad_scale = decoder_grad_scale
        self.decoder_channels = tuple(decoder_channels)
        self.pfp_channels = pfp_channels

        self.m1, dino_channels = _make_dino(dino_variant, dino_checkpoint, multi_output=True)
        num_register_tokens = getattr(self.m1, 'num_register_tokens', 4)
        num_extra_tokens = 1 + num_register_tokens

        self.m2 = timm.create_model(
            decoder_name,
            pretrained=decoder_pretrained,
            drop_path_rate=decoder_drop_path_rate,
        )
        _disable_classifier(self.m2)

        self.rb4 = ReOrderBLC(
            dino_channels,
            self.decoder_channels[0],
            token_hidden_channels,
            up_factor=token_up_factor,
            num_extra_tokens=num_extra_tokens,
        )
        self.rbx1 = ReOrderBHWC(self.decoder_channels[0], pfp_channels, 384, up_factor=1)
        self.rbx2 = ReOrderBHWC(self.decoder_channels[1], pfp_channels, 768, up_factor=2)
        self.rbx3 = ReOrderBHWC(self.decoder_channels[2], pfp_channels, 1536, up_factor=4)
        self.rbx4 = ReOrderBHWC(self.decoder_channels[3], pfp_channels, 3072, up_factor=8)
        self.rbf = ReOrderBHWC(pfp_channels * 4, pfp_channels, 512, up_factor=2)
        self.conv_depth = nn.Conv2d(pfp_channels, 1, kernel_size=3, padding=1, stride=1)

    def _encode_tokens(self, image):
        tokens, size, pad = self.m1.patch(image)
        batch, _, height, width = size
        pad_h, pad_w = pad
        height = height + pad_h
        width = width + pad_w
        features = self.m1.forward_features(tokens, size, pad)
        feature = features[-1] if isinstance(features, (list, tuple)) else features
        h = height // self.patch_size
        w = width // self.patch_size
        return feature, batch, h, w

    def _encode_for_stem(self, image):
        feature, batch, h, w = self._encode_tokens(image)
        feature = self.rb4(feature, h, w)
        feature = feature.reshape(
            batch,
            h * self.token_up_factor,
            w * self.token_up_factor,
            -1,
        ).permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, inputs, img_metas):
        image = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        encoded = self._encode_for_stem(image)
        stem = self.m2.stem(image)
        encoded = F.interpolate(encoded, size=stem.shape[-2:], mode='bilinear', align_corners=False)
        x = stem * encoded
        return self._run_convnext_pfp(x)


@HEADS.register_module()
class DINOxConvI(DINOxConv):
    """xNet-I variant: DINO encoded features are mixed directly with RGB image."""

    def __init__(self,
                 image_mix_hidden_channels=1024,
                 image_mix_up_factor=4,
                 **kwargs):
        super().__init__(token_hidden_channels=image_mix_hidden_channels,
                         token_up_factor=image_mix_up_factor,
                         **kwargs)
        dino_channels = 1024 if kwargs.get('dino_variant', 'large') == 'large' else 384
        num_register_tokens = getattr(self.m1, 'num_register_tokens', 4)
        self.rb4 = ReOrderBLC(
            dino_channels,
            3,
            image_mix_hidden_channels,
            up_factor=image_mix_up_factor,
            num_extra_tokens=1 + num_register_tokens,
        )

    def forward(self, inputs, img_metas):
        image = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        encoded = self._encode_for_stem(image)
        encoded = F.interpolate(encoded, size=image.shape[-2:], mode='bilinear', align_corners=False)
        x = self.m2.stem(image * encoded)
        return self._run_convnext_pfp(x)


@HEADS.register_module()
class SwinXConv(_XNetPFPHead):
    """xNet for MDE: Swin encoder x ConvNeXt pre-trained decoder."""

    def __init__(self,
                 encoder_name='swin_large_patch4_window7_224.ms_in22k',
                 decoder_name='convnext_base.fb_in22k_ft_in1k_384',
                 encoder_pretrained=True,
                 decoder_pretrained=True,
                 encoder_drop_path_rate=0.3,
                 img_size=None,
                 encoder_channels=(192, 384, 768, 1536),
                 decoder_channels=(128, 256, 512, 1024),
                 pfp_channels=96,
                 decoder_grad_scale=0.25,
                 **kwargs):
        super().__init__(**kwargs)
        self.decoder_grad_scale = decoder_grad_scale
        self.encoder_channels = tuple(encoder_channels)
        self.decoder_channels = tuple(decoder_channels)
        create_kwargs = dict(pretrained=encoder_pretrained, drop_path_rate=encoder_drop_path_rate)
        if img_size is not None:
            create_kwargs['img_size'] = img_size
        self.m1 = timm.create_model(encoder_name, **create_kwargs)
        self.m2 = timm.create_model(decoder_name, pretrained=decoder_pretrained)
        _disable_classifier(self.m1)
        _disable_classifier(self.m2)

        self.rb4 = ReOrderBHWC(self.encoder_channels[3], self.decoder_channels[0], 4096, up_factor=8)
        self.rbx1 = ReOrderBHWC(self.decoder_channels[0], pfp_channels, 256, up_factor=1)
        self.rbx2 = ReOrderBHWC(self.decoder_channels[1], pfp_channels, 512, up_factor=2)
        self.rbx3 = ReOrderBHWC(self.decoder_channels[2], pfp_channels, 1024, up_factor=4)
        self.rbx4 = ReOrderBHWC(self.decoder_channels[3], pfp_channels, 2048, up_factor=8)
        self.rbf = ReOrderBHWC(pfp_channels * 4, pfp_channels, 512, up_factor=2)
        self.conv_depth = nn.Conv2d(pfp_channels, 1, kernel_size=3, padding=1, stride=1)

    def _encode_for_stem(self, image):
        encoded = self.m1(image)
        encoded = self.rb4(encoded, bchw=False).permute(0, 3, 1, 2).contiguous()
        return encoded

    def forward(self, inputs, img_metas):
        image = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        encoded = self._encode_for_stem(image)
        stem = self.m2.stem(image)
        encoded = F.interpolate(encoded, size=stem.shape[-2:], mode='bilinear', align_corners=False)
        x = stem * encoded
        return self._run_convnext_pfp(x)


@HEADS.register_module()
class SwinXConvI(SwinXConv):
    """xNet-I variant: Swin encoded features are mixed directly with RGB image."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rb4 = ReOrderBHWC(self.encoder_channels[3], 3, 4096, up_factor=8)

    def forward(self, inputs, img_metas):
        image = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        encoded = self._encode_for_stem(image)
        encoded = F.interpolate(encoded, size=image.shape[-2:], mode='bilinear', align_corners=False)
        x = self.m2.stem(image * encoded)
        return self._run_convnext_pfp(x)


@HEADS.register_module()
class XNetDINOConv(DINOxConv):
    pass


@HEADS.register_module()
class XNetDINOConvI(DINOxConvI):
    pass


@HEADS.register_module()
class XNetSwinConv(SwinXConv):
    pass


@HEADS.register_module()
class XNetSwinConvI(SwinXConvI):
    pass
