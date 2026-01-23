#!/bin/bash
# 小规模训练命令 - 验证损失计算修复（使用CPU避免GPU竞争）

cd /data2/lyh/gnn_project/SparseDiff
conda activate sparse

# 使用CPU模式避免GPU竞争（因为已经有其他训练任务在运行）
python3 sparse_diffusion/main.py \
  +experiment=debug.yaml \
  dataset.name=acm_subgraphs \
  dataset.datadir=data/ACM_subgraphs \
  train.n_epochs=2 \
  train.batch_size=2 \
  train.lr=0.0002 \
  general.gpus=0 \
  general.wandb=online \
  general.name=test_loss_fix_cpu \
  model.diffusion_steps=1000 \
  train.num_workers=2 \
  general.sample_every_val=1 \
  general.samples_to_generate=2 \
  train.save_model=False

