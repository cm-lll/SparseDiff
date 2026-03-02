# 完整训练命令

## 训练前释放 GPU（避免 OOM）

若上次训练异常退出或后台进程未结束，GPU 显存可能未释放，新训练易 OOM。建议每次开训前：

```bash
# 1. 看谁在占 GPU
nvidia-smi

# 2. 若有残留的 python 进程（如 main.py），记下 PID 后结束，例如：
# kill -9 <PID>

# 3. 再次确认显存已空
nvidia-smi
```

若在 Python 里跑（如 Jupyter），可在 import 后加一行清空当前进程的 CUDA 缓存：`torch.cuda.empty_cache()`。命令行重新起进程时一般不需要。

## 重试前先清理资源

每次说「再试一下」前，必须先结束上次训练占用的进程，否则会抢显存或卡住：

```bash
# 结束本项目的训练进程
pkill -f "sparse_diffusion/main.py"
pkill -f "run_acm_train.sh"
sleep 2
# 若 nvidia-smi 里仍有你的 python3（非 1266051），记下 PID 后：kill -9 <PID>
nvidia-smi
```

确认显存无你的进程后，再执行下面的 ACM 训练。

## ACM 四卡训练（+experiment=acm_train）

**必须使用 GPU 1,2,3,4**（不要用 0），否则可能失败。两种方式任选其一：

**方式一：用脚本（推荐）**

```bash
cd /data2/lyh/gnn_project/SparseDiff
chmod +x run_acm_train.sh
./run_acm_train.sh
```

脚本里已设置 `CUDA_VISIBLE_DEVICES=1,2,3,4`，并自动 `conda activate sparse`。

**方式二：命令行手动指定**

```bash
cd /data2/lyh/gnn_project/SparseDiff
conda activate sparse

export CUDA_VISIBLE_DEVICES=1,2,3,4
python3 sparse_diffusion/main.py +experiment=acm_train
```

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

## 大型训练与测试（acm_train_large，参考 2026-02-03 acm_train）

在 2026-02-03 的 `+experiment=acm_train` 基础上放大：更多 epoch、更大 batch、更多 diffusion 步数、更大模型、开启验证/测试采样并增加采样数量。

**一键启动（四卡，GPU 1,2,3,4）：**

```bash
cd /data2/lyh/gnn_project/SparseDiff
chmod +x run_acm_train_large.sh
./run_acm_train_large.sh
```

**或手动指定：**

```bash
cd /data2/lyh/gnn_project/SparseDiff
conda activate sparse
export CUDA_VISIBLE_DEVICES=1,2,3,4
python3 sparse_diffusion/main.py +experiment=acm_train_large
```

**与 2 月 3 日 acm_train 对比：**

| 项 | 2026-02-03 acm_train | acm_train_large |
|----|----------------------|------------------|
| gpus | 4 | 4 |
| n_epochs | 100 | 300 |
| batch_size | 4 | 16 |
| diffusion_steps | 100 | 200 |
| n_layers | 4 | 5 |
| hidden_dims (dx/dim_ffX) | 128/256 | 256/512 |
| enable_val_sampling | false | true |
| enable_test_sampling | false | true |
| samples_to_generate | 10 | 50 |
| final_model_samples_to_generate | 10 | 200 |

训练结束后会自动在测试集上跑采样（`enable_test_sampling=true`），生成图与指标会写入当前 run 目录并同步到 wandb。

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

