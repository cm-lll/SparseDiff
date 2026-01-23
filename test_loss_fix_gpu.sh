#!/bin/bash
# 小规模训练命令 - 验证损失计算修复（使用GPU）

cd /data2/lyh/gnn_project/SparseDiff
conda activate sparse

# 清理CUDA缓存
python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

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

