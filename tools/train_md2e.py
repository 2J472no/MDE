# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import sys
import time

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from depth import __version__
from depth.apis import set_random_seed, train_depther
from depth.datasets import build_dataset
from depth.models import build_depther
from depth.utils import collect_env, get_root_logger


MD2E_MULTI_RES_SCALES = (
    (352, 704),
    (480, 640),
    (512, 512),
    (512, 768),
    (616, 1064),
    (704, 704),
    (768, 1024),
    (352, 1216),
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MD2E')
    parser.add_argument(
        '--config',
        default='configs/MD2E/md2e.py',
        help='MD2E config file path')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/md2e',
        help='directory to save logs and checkpoints')
    parser.add_argument('--load-from', help='checkpoint file to load weights')
    parser.add_argument('--resume-from', help='checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='skip validation during training')
    parser.add_argument(
        '--fixed-resolution',
        action='store_true',
        help='use the crop size defined in the config')
    parser.add_argument(
        '--multi-res-scales',
        nargs='+',
        help='override multi-resolution crops, for example 352x704 768x1024')
    parser.add_argument(
        '--multi-res-stride',
        type=int,
        default=1,
        help='number of ranks sharing the same crop before moving to the next')
    parser.add_argument(
        '--multi-res-offset',
        type=int,
        default=0,
        help='offset applied to the rank before selecting the crop')
    parser.add_argument(
        '--samples-per-gpu',
        type=int,
        help='override samples_per_gpu after crop selection')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        default=1,
        type=int,
        help='number of gpus for non-distributed training')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='gpu ids for non-distributed training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='set deterministic options for CUDNN')
    parser.add_argument('--options', nargs='+', action=DictAction)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def parse_scale(text):
    text = text.lower().replace(',', 'x')
    parts = [p for p in text.split('x') if p]
    if len(parts) != 2:
        raise ValueError(f'Invalid crop size: {text}')
    return int(parts[0]), int(parts[1])


def get_rank():
    for key in ('RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        value = os.environ.get(key)
        if value is not None:
            return int(value)
    return 0


def get_scales(args):
    if args.multi_res_scales:
        return tuple(parse_scale(scale) for scale in args.multi_res_scales)
    return MD2E_MULTI_RES_SCALES


def patch_random_crop(pipeline, crop_size):
    for transform in pipeline:
        if transform.get('type') == 'RandomCrop':
            transform['crop_size'] = tuple(crop_size)
            return

    insert_at = 0
    for idx, transform in enumerate(pipeline):
        if transform.get('type') == 'DepthLoadAnnotations':
            insert_at = idx + 1
            break
    pipeline.insert(
        insert_at,
        dict(type='RandomCrop', crop_size=tuple(crop_size), resize_prob=0.0))


def apply_multi_resolution(cfg, args):
    if args.fixed_resolution:
        if args.samples_per_gpu is not None:
            cfg.data.samples_per_gpu = args.samples_per_gpu
        return None

    scales = get_scales(args)
    if not scales:
        raise ValueError('At least one crop size is required.')
    if args.multi_res_stride < 1:
        raise ValueError('--multi-res-stride must be >= 1.')

    rank = get_rank()
    scale_index = ((rank + args.multi_res_offset) // args.multi_res_stride) % len(scales)
    crop_size = scales[scale_index]

    train_cfgs = cfg.data.train if isinstance(cfg.data.train, (list, tuple)) else [cfg.data.train]
    for train_cfg in train_cfgs:
        patch_random_crop(train_cfg.pipeline, crop_size)

    base_pixels = 768 * 1024
    samples_per_gpu = max(1, base_pixels // (crop_size[0] * crop_size[1]))
    if args.samples_per_gpu is not None:
        samples_per_gpu = args.samples_per_gpu
    cfg.data.samples_per_gpu = samples_per_gpu
    cfg.md2e_multi_resolution = dict(
        rank=rank,
        scale_index=scale_index,
        crop_size=crop_size,
        samples_per_gpu=samples_per_gpu)
    return cfg.md2e_multi_resolution


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('MD2E training requires CUDA.')

    if args.launcher != 'none':
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
    elif args.gpu_ids is not None:
        torch.cuda.set_device(args.gpu_ids[0])
    else:
        torch.cuda.set_device(0)

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    multi_res_meta = apply_multi_resolution(cfg, args)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    logger.info(f'Distributed training: {distributed}')
    if multi_res_meta is not None:
        logger.info(f'MD2E multi-resolution: {multi_res_meta}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_depther(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            depth_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text)
        meta.update(cfg.checkpoint_config.meta)

    train_depther(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
