#!/bin/bash
# 小规模训练命令 - 验证损失计算修复（修复OOM问题）

cd /data2/lyh/gnn_project/SparseDiff
conda activate sparse

# 使用更小的 batch_size 避免 OOM
python3 sparse_diffusion/main.py \
  +experiment=debug.yaml \
  dataset.name=acm_subgraphs \
  dataset.datadir=data/ACM_subgraphs \
  train.n_epochs=2 \
  train.batch_size=8 \
  train.lr=0.0002 \
  general.gpus=1 \
  general.wandb=online \
  general.name=test_loss_fix \
  model.diffusion_steps=1000 \
  train.num_workers=4 \
  general.sample_every_val=1 \
  general.samples_to_generate=4 \
  train.save_model=False

