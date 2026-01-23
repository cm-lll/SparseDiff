# 完整训练命令

## 基础训练命令（单GPU，用于测试）

```bash
cd /data2/lyh/gnn_project/SparseDiff
conda activate sparse

python3 sparse_diffusion/main.py \
  +experiment=debug.yaml \
  dataset.name=acm_subgraphs \
  dataset.datadir=data/ACM_subgraphs \
  train.n_epochs=100 \
  train.batch_size=32 \
  train.lr=0.0002 \
  general.gpus=1 \
  general.wandb=online \
  model.diffusion_steps=1000 \
  train.num_workers=8
```

## 完整训练命令（多GPU，8x RTX 2080）

```bash
cd /data2/lyh/gnn_project/SparseDiff
conda activate sparse

python3 sparse_diffusion/main.py \
  +experiment=debug.yaml \
  dataset.name=acm_subgraphs \
  dataset.datadir=data/ACM_subgraphs \
  train.n_epochs=1000 \
  train.batch_size=512 \
  train.lr=0.0002 \
  general.gpus=8 \
  general.wandb=online \
  model.diffusion_steps=1000 \
  train.num_workers=16 \
  general.sample_every_val=10 \
  general.samples_to_generate=100
```

## 推荐训练命令（中等规模，平衡速度和效果）

```bash
cd /data2/lyh/gnn_project/SparseDiff
conda activate sparse

python3 sparse_diffusion/main.py \
  +experiment=debug.yaml \
  dataset.name=acm_subgraphs \
  dataset.datadir=data/ACM_subgraphs \
  train.n_epochs=500 \
  train.batch_size=128 \
  train.lr=0.0002 \
  general.gpus=8 \
  general.wandb=online \
  model.diffusion_steps=1000 \
  train.num_workers=16 \
  general.sample_every_val=5 \
  general.samples_to_generate=50 \
  train.save_model=True
```

## 参数说明

- `train.n_epochs`: 训练轮数
- `train.batch_size`: 批次大小
- `train.lr`: 学习率
- `general.gpus`: GPU数量（0=CPU, 1=单GPU, 8=8个GPU）
- `general.wandb`: wandb模式（online/offline/disabled）
- `model.diffusion_steps`: 扩散步数
- `train.num_workers`: 数据加载器工作进程数
- `general.sample_every_val`: 每N个epoch采样一次
- `general.samples_to_generate`: 每次采样生成的图数量
- `train.save_model`: 是否保存模型

## 后台运行（使用 nohup）

```bash
nohup python3 sparse_diffusion/main.py \
  +experiment=debug.yaml \
  dataset.name=acm_subgraphs \
  dataset.datadir=data/ACM_subgraphs \
  train.n_epochs=500 \
  train.batch_size=128 \
  train.lr=0.0002 \
  general.gpus=8 \
  general.wandb=online \
  model.diffusion_steps=1000 \
  train.num_workers=16 \
  general.sample_every_val=5 \
  general.samples_to_generate=50 \
  train.save_model=True \
  > train.log 2>&1 &
```

## 使用 tmux（推荐）

```bash
# 创建新会话
tmux new -s sparse_train

# 在tmux中运行
cd /data2/lyh/gnn_project/SparseDiff
conda activate sparse

python3 sparse_diffusion/main.py \
  +experiment=debug.yaml \
  dataset.name=acm_subgraphs \
  dataset.datadir=data/ACM_subgraphs \
  train.n_epochs=500 \
  train.batch_size=128 \
  train.lr=0.0002 \
  general.gpus=8 \
  general.wandb=online \
  model.diffusion_steps=1000 \
  train.num_workers=16 \
  general.sample_every_val=5 \
  general.samples_to_generate=50 \
  train.save_model=True

# 分离会话: Ctrl+B, 然后按 D
# 重新连接: tmux attach -t sparse_train
```

