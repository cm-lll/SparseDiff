#!/bin/bash
# ACM 四卡训练：强制使用 GPU 1,2,3,4（避免占用 GPU 0）
set -e
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=1,2,3,4
echo "Using GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /data2/lyh/miniconda3/etc/profile.d/conda.sh ]; then
    source /data2/lyh/miniconda3/etc/profile.d/conda.sh
fi
conda activate sparse 2>/dev/null || source activate sparse 2>/dev/null || true

python3 sparse_diffusion/main.py +experiment=acm_train "$@"
