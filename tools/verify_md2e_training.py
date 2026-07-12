import argparse
import os.path as osp
import sys

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from mmcv.utils import Config, DictAction

from depth.models import build_depther
from tools.train_md2e import MD2E_MULTI_RES_SCALES, parse_scale


FORBIDDEN_CHECKPOINT_KEYS = ('optimizer', 'meta', 'epoch', 'iter', 'state_dict')


def parse_args():
    parser = argparse.ArgumentParser(description='Verify MD2E training smoke test')
    parser.add_argument(
        '--config',
        default='configs/MD2E/md2e.py',
        help='MD2E config file path')
    parser.add_argument(
        '--checkpoint',
        default='checkpoints/md2e_multires_best_mde.pth',
        help='clean public MD2E checkpoint')
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='device used by the smoke test')
    parser.add_argument('--height', type=int, default=112)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-backward', action='store_true')
    parser.add_argument(
        '--dry-run-config',
        action='store_true',
        help='only print the multi-resolution schedule')
    parser.add_argument(
        '--multi-res-scales',
        nargs='+',
        help='override multi-resolution crops, for example 352x704 768x1024')
    parser.add_argument('--options', nargs='+', action=DictAction)
    return parser.parse_args()


def get_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    return torch.device(name)


def get_scales(args):
    if args.multi_res_scales:
        return tuple(parse_scale(scale) for scale in args.multi_res_scales)
    return MD2E_MULTI_RES_SCALES


def dry_run_schedule(args):
    scales = get_scales(args)
    base_pixels = 768 * 1024
    print('MD2E multi-resolution schedule:')
    for rank, crop_size in enumerate(scales):
        samples = max(1, base_pixels // (crop_size[0] * crop_size[1]))
        print(
            f'  rank {rank:02d}: crop={crop_size[0]}x{crop_size[1]}, '
            f'samples_per_gpu={samples}')


def build_model(config_path, options=None):
    cfg = Config.fromfile(config_path)
    if options is not None:
        cfg.merge_from_dict(options)
    cfg.model.decode_head.encoder_checkpoint = None
    model = build_depther(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    return model


def load_clean_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    forbidden = [key for key in FORBIDDEN_CHECKPOINT_KEYS if key in checkpoint]
    if forbidden:
        raise RuntimeError(
            f'Checkpoint is not clean; found top-level keys: {forbidden}')

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            'Checkpoint does not strictly match MD2E. '
            f'missing={len(missing)}, unexpected={len(unexpected)}')
    return len(checkpoint)


def make_synthetic_batch(batch_size, height, width, device):
    image = torch.rand(batch_size, 3, height, width, device=device)

    y = torch.linspace(0.0, 1.0, height, device=device).view(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, 1, width)
    depth = 0.5 + 20.0 * (0.65 * x + 0.35 * y)
    depth = depth.expand(batch_size, 1, height, width).contiguous()

    img_metas = [
        dict(
            filename='synthetic',
            ori_filename='synthetic',
            ori_shape=(height, width, 3),
            img_shape=(height, width, 3),
            pad_shape=(height, width, 3),
            scale_factor=1.0,
            flip=False,
            flip_direction=None,
            img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True))
        for _ in range(batch_size)
    ]
    return image, depth, img_metas


def scalar_loss(losses):
    total = None
    for value in losses.values():
        if torch.is_tensor(value):
            item = value.mean()
        elif isinstance(value, (list, tuple)):
            tensors = [v.mean() for v in value if torch.is_tensor(v)]
            if not tensors:
                continue
            item = sum(tensors)
        else:
            continue
        total = item if total is None else total + item
    if total is None:
        raise RuntimeError('No tensor loss was produced.')
    return total


def main():
    args = parse_args()
    if args.dry_run_config:
        dry_run_schedule(args)
        return

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    model = build_model(args.config, args.options)
    parameter_count = load_clean_checkpoint(model, args.checkpoint)
    model.to(device)
    model.train()

    image, depth, img_metas = make_synthetic_batch(
        args.batch_size, args.height, args.width, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    optimizer.zero_grad(set_to_none=True)

    losses = model.forward_train(image, img_metas, depth)
    total = scalar_loss(losses)
    if not torch.isfinite(total):
        raise RuntimeError(f'Non-finite loss: {total.item()}')

    if not args.no_backward:
        total.backward()
        optimizer.step()

    loss_items = ', '.join(
        f'{key}={value.detach().float().mean().item():.6f}'
        for key, value in losses.items()
        if torch.is_tensor(value))
    print(f'Loaded clean checkpoint keys: {parameter_count}')
    print(
        f'Smoke input: batch={args.batch_size}, '
        f'size={args.height}x{args.width}, device={device}')
    print(f'Losses: {loss_items}')
    print('MD2E training smoke test passed.')


if __name__ == '__main__':
    main()
