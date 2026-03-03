#!/bin/bash
# ACM 训练：默认使用 GPU 1,2,3,4（避免占用 GPU 0）
# 若外部已设置 CUDA_VISIBLE_DEVICES，则尊重外部设置（便于换卡/共享机器时手动选卡）
set -e
cd "$(dirname "$0")"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=1,2,3,4
fi
echo "Using GPUs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /data2/lyh/miniconda3/etc/profile.d/conda.sh ]; then
    source /data2/lyh/miniconda3/etc/profile.d/conda.sh
fi
conda activate sparse 2>/dev/null || source activate sparse 2>/dev/null || true

python3 sparse_diffusion/main.py +experiment=acm_train "$@"
