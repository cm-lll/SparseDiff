#!/bin/bash
# 清理CUDA缓存并使用GPU测试（修复版本）

cd /data2/lyh/gnn_project/SparseDiff

# 初始化conda（用于非交互式shell）
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /data2/lyh/miniconda3/etc/profile.d/conda.sh ]; then
    source /data2/lyh/miniconda3/etc/profile.d/conda.sh
else
    echo "Warning: conda.sh not found, trying direct activation"
fi

echo "=== 清理CUDA缓存 ==="
# 先激活环境再清理缓存
conda activate sparse 2>/dev/null || source activate sparse 2>/dev/null || {
    echo "Error: Failed to activate conda environment 'sparse'"
    exit 1
}

python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('CUDA cache cleared!')"

echo ""
echo "=== 检查GPU状态 ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -3

echo ""
echo "=== 开始GPU测试 ==="
# 使用GPU进行测试（使用较小的batch_size避免OOM）
python3 sparse_diffusion/main.py \
  +experiment=debug.yaml \
  dataset.name=acm_subgraphs \
  dataset.datadir=data/ACM_subgraphs \
  train.n_epochs=2 \
  train.batch_size=4 \
  train.lr=0.0002 \
  general.gpus=1 \
  general.wandb=online \
  general.name=test_loss_fix_gpu \
  model.diffusion_steps=1000 \
  train.num_workers=4 \
  general.sample_every_val=1 \
  general.samples_to_generate=2 \
  train.save_model=False
