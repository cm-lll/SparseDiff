#!/bin/bash
# 仅做采样（test_only），不训练。需要已有 checkpoint。
#
# 用法:
#   ./sample_heterogeneous_quick.sh /path/to/last.ckpt [10|100] [diffusion_steps]
# 示例:
#   ./sample_heterogeneous_quick.sh /path/to/last.ckpt 10
#   ./sample_heterogeneous_quick.sh /path/to/last.ckpt 100 50
#   ./sample_heterogeneous_quick.sh /path/to/short.ckpt 10 20   # ckpt 是 short 脚本（20 步）训的
#
# 参数:
#   $1: checkpoint 绝对路径（必填）
#   $2: 生成图数量，10 或 100（默认 10）
#   $3: diffusion_steps，须与训练该 ckpt 时一致（默认 50，对应 train_heterogeneous_multi_epoch；short 脚本训的用 20）
#   $4: 可选，dataset.datadir；若 ckpt 用 100 子图 ACM 训练，需与训练一致，如 /data2/lyh/gnn_project/data/ACM_subgraphs
#   $5: 可选，每个图的节点数（用于大图训练、小图生成）；null=从训练分布采样，50=所有图50节点
#
# 重要：采样用的 diffusion_steps、dataset.datadir 必须和训练时一致，否则维度或噪声计划对不上。
# 若报 GLIBCXX_3.4.29：先运行 source scripts/fix_libstdc_conda.sh 或 --install 永久修复。

cd /data2/lyh/gnn_project/SparseDiff

if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/checkpoint.ckpt [10|100] [diffusion_steps] [datadir] [num_nodes]"
    echo "  $0 /path/to/last.ckpt 10        # 10 个图，50 步（multi_epoch 的 ckpt）"
    echo "  $0 /path/to/last.ckpt 100 50    # 100 个图，50 步"
    echo "  $0 /path/to/short.ckpt 10 20    # 10 个图，20 步（short 脚本的 ckpt）"
    echo "  $0 /path/to/last.ckpt 10 50 <datadir> 50  # 生成小图（每个图50节点）"
    exit 1
fi

CKPT="$1"
N=${2:-10}
T=${3:-50}
DATADIR_OVERRIDE="$4"
NUM_NODES_ARG="$5"

if [ ! -f "$CKPT" ]; then
    echo "Error: checkpoint not found: $CKPT"
    exit 1
fi

if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /data2/lyh/miniconda3/etc/profile.d/conda.sh ]; then
    source /data2/lyh/miniconda3/etc/profile.d/conda.sh
fi

conda activate sparse 2>/dev/null || source activate sparse 2>/dev/null || { echo "Error: conda env 'sparse'"; exit 1; }

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
[ -n "$CONDA_PREFIX" ] && export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "=== 仅采样（test_only） ==="
echo "  checkpoint: $CKPT"
echo "  生成数量: $N   diffusion_steps: $T（须与训练该 ckpt 时一致）"
[ -n "$DATADIR_OVERRIDE" ] && echo "  dataset.datadir: $DATADIR_OVERRIDE"
[ -n "$NUM_NODES_ARG" ] && echo "  每个图节点数: $NUM_NODES_ARG（用于大图训练、小图生成）"

DATADIR_ARG="dataset.datadir=${DATADIR_OVERRIDE:-data/ACM_subgraphs}"
NUM_NODES_CONFIG=""
[ -n "$NUM_NODES_ARG" ] && NUM_NODES_CONFIG="general.sample_num_nodes=$NUM_NODES_ARG"

# 使用引号包裹整个 test_only 值，避免 Hydra 解析错误
python3 sparse_diffusion/main.py \
  +experiment=debug.yaml \
  dataset.name=acm_subgraphs \
  $DATADIR_ARG \
  general.gpus=1 \
  general.wandb=disabled \
  general.name=sample_hetero_quick \
  "general.test_only=\"$CKPT\"" \
  general.final_model_samples_to_generate=$N \
  general.final_model_samples_to_save=30 \
  model.diffusion_steps=$T \
  model.edge_fraction=1.0 \
  train.num_workers=2 \
  model.extra_features=null \
  $NUM_NODES_CONFIG

echo ""
echo "=== 完成 ==="
echo "  输出在: outputs/.../sample_hetero_quick/ 下的 graphs/, chains/, epoch*_res_mean*.txt 等"
