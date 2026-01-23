#!/bin/bash
# 测试异质图修复后的实现 - 强制使用GPU 2

cd /data2/lyh/gnn_project/SparseDiff

# 初始化conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /data2/lyh/miniconda3/etc/profile.d/conda.sh ]; then
    source /data2/lyh/miniconda3/etc/profile.d/conda.sh
fi

echo "=== 清理CUDA缓存 ==="
conda activate sparse 2>/dev/null || source activate sparse 2>/dev/null || {
    echo "Error: Failed to activate conda environment 'sparse'"
    exit 1
}

python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('CUDA cache cleared!')"

echo ""
echo "=== 强制使用GPU 2（避免GPU 0内存不足） ==="
export CUDA_VISIBLE_DEVICES=2
echo "✓ 设置 CUDA_VISIBLE_DEVICES=2"

echo ""
echo "=== 检查数据文件 ==="
DATA_DIR="data/ACM_subgraphs"
PROCESSED_DIR="${DATA_DIR}/processed"
TRAIN_COUNTS_FILE="${PROCESSED_DIR}/train_edge_family_avg_counts.pickle"

if [ ! -f "$TRAIN_COUNTS_FILE" ]; then
    echo "⚠️  未找到 $TRAIN_COUNTS_FILE"
    echo "⚠️  需要重新处理数据以生成关系族平均边数文件"
    echo "⚠️  这将自动在首次运行时生成，但可能需要一些时间"
else
    echo "✓ 找到 $TRAIN_COUNTS_FILE"
fi

echo ""
echo "=== 开始测试（小规模配置，使用GPU 2） ==="
# 使用最小的batch_size和配置避免OOM
python3 sparse_diffusion/main.py \
  +experiment=debug.yaml \
  dataset.name=acm_subgraphs \
  dataset.datadir=data/ACM_subgraphs \
  train.n_epochs=2 \
  train.batch_size=1 \
  train.lr=0.0002 \
  general.gpus=1 \
  general.wandb=online \
  general.name=test_heterogeneous_fix_gpu2 \
  model.diffusion_steps=1000 \
  train.num_workers=2 \
  general.sample_every_val=1 \
  general.samples_to_generate=1 \
  train.save_model=False \
  model.extra_features=null

echo ""
echo "=== 测试完成 ==="
