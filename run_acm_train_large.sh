#!/bin/bash
# ACM 大型训练：四卡，更多 epoch/步数/采样，开启验证与测试阶段采样
# 参考 2026-02-03 的 acm_train 参数，放大为 acm_train_large
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

python3 sparse_diffusion/main.py +experiment=acm_train_large "$@"
