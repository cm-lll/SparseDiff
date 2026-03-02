#!/bin/bash
# 使用训练好的 ACM 模型做子图生成（采样）
# 用法:
#   ./run_acm_sample.sh <checkpoint路径>
#   CKPT=path/to/last.ckpt ./run_acm_sample.sh
# 示例:
#   ./run_acm_sample.sh checkpoints/acm_train/last.ckpt
#   ./run_acm_sample.sh outputs/2025-02-25/12-00-00-acm_train/checkpoints/acm_train/last.ckpt

set -e
cd "$(dirname "$0")"

#  checkpoint 路径：优先用第一个参数，否则用环境变量 CKPT
CKPT="${1:-${CKPT}}"
if [ -z "$CKPT" ]; then
    echo "用法: $0 <checkpoint路径>"
    echo "  或: CKPT=path/to/last.ckpt $0"
    echo "示例: $0 checkpoints/acm_train/last.ckpt"
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "错误: 找不到 checkpoint 文件: $CKPT"
    exit 1
fi

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /data2/lyh/miniconda3/etc/profile.d/conda.sh ]; then
    source /data2/lyh/miniconda3/etc/profile.d/conda.sh
fi
conda activate sparse 2>/dev/null || source activate sparse 2>/dev/null || true

# 单卡采样，避免 DDP 拆分；如需指定 GPU 可设置 CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# +experiment=acm_train 与训练时一致；test_only 指向刚训练好的 ckpt；开启测试阶段采样
# 路径含 / 时用环境变量传入，Hydra 用 ${oc.env:ACM_CKPT} 读取
# 固定 563 节点/图，生成 5 张图；边数量由模型去噪生成（无法写死为 294），生成指标会写入 test_epoch*.json 与 wandb
export ACM_CKPT="$(realpath "$CKPT")"
python3 sparse_diffusion/main.py \
    +experiment=acm_train \
    'general.test_only=${oc.env:ACM_CKPT}' \
    general.enable_test_sampling=true \
    general.gpus=1 \
    general.sample_num_nodes=563 \
    general.final_model_samples_to_generate=5 \
    general.final_model_samples_to_save=5 \
    general.final_model_chains_to_save=1 \
    "${@:2}"
