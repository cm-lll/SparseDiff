#!/bin/bash
# 清理CUDA缓存并使用GPU测试

cd /data2/lyh/gnn_project/SparseDiff

# 初始化conda（用于非交互式shell）
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /data2/lyh/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

echo "=== 清理CUDA缓存 ==="
python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('CUDA cache cleared!')" 2>/dev/null || echo "PyTorch not available in base environment"

echo ""
echo "=== 检查GPU状态 ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -3

echo ""
echo "=== 激活conda环境 ==="
conda activate sparse || source activate sparse || {
    echo "Error: Failed to activate conda environment 'sparse'"
    echo "Please activate it manually: conda activate sparse"
    exit 1
}

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

