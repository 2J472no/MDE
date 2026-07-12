#!/usr/bin/env bash

CONFIG=${1:-configs/MD2E/md2e.py}
GPUS=${2:-1}
PORT=${PORT:-38423}

PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="$PORT" \
    "$(dirname "$0")/train_md2e.py" --config "$CONFIG" --launcher pytorch "${@:3}"
