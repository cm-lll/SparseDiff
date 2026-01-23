#!/bin/bash
# 测试异质图修复后的实现

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
echo "=== 检查GPU状态 ==="
# 获取所有GPU的内存使用情况
GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
echo "$GPU_INFO"

echo ""
echo "=== 自动选择GPU ==="
# 排除GPU 0（通常内存不足），优先选择其他GPU
# 解析GPU信息，选择内存使用最少的GPU
BEST_GPU=""
BEST_FREE_MEM=0
EXCLUDE_GPUS="0"  # 排除GPU 0

while IFS=',' read -r idx mem_used mem_total util; do
    # 清理空格
    idx=$(echo $idx | xargs)
    mem_used=$(echo $mem_used | xargs)
    mem_total=$(echo $mem_total | xargs)
    util=$(echo $util | xargs)
    
    # 跳过被排除的GPU
    if [[ " $EXCLUDE_GPUS " =~ " $idx " ]]; then
        echo "GPU $idx: 已排除（通常内存不足）"
        continue
    fi
    
    # 计算可用内存
    mem_free=$((mem_total - mem_used))
    
    # 如果这个GPU的可用内存更多，选择它
    if [ $mem_free -gt $BEST_FREE_MEM ]; then
        BEST_GPU=$idx
        BEST_FREE_MEM=$mem_free
    fi
    
    echo "GPU $idx: 已用 ${mem_used}MB / ${mem_total}MB, 可用 ${mem_free}MB, 利用率 ${util}%"
done <<< "$GPU_INFO"

# 如果所有GPU都被排除或没有找到合适的GPU，使用GPU 2作为备选
if [ -z "$BEST_GPU" ]; then
    echo "警告: 无法找到合适的GPU，使用备选GPU 2"
    BEST_GPU=2
    BEST_FREE_MEM=0
fi

echo ""
echo "✓ 选择 GPU $BEST_GPU (可用内存: ${BEST_FREE_MEM}MB)"

# 对于测试，只使用单个GPU（避免OOM）
# 如果用户需要多GPU，可以手动修改
NUM_GPUS=1
MULTI_GPU_LIST="$BEST_GPU"
echo "✓ 使用单GPU模式: GPU $BEST_GPU (避免OOM)"

# 设置 CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$MULTI_GPU_LIST
echo "✓ 设置 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo ""
echo "=== 检查数据文件 ==="
# 检查是否需要重新处理数据（生成 edge_family_avg_counts.pickle）
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
echo "=== 开始测试（小规模配置） ==="
# 使用最小的batch_size和配置避免OOM
# 测试修复后的异质图查询边采样和真实边数统计
python3 sparse_diffusion/main.py \
  +experiment=debug.yaml \
  dataset.name=acm_subgraphs \
  dataset.datadir=data/ACM_subgraphs \
  train.n_epochs=2 \
  train.batch_size=1 \
  train.lr=0.0002 \
  general.gpus=$NUM_GPUS \
  general.wandb=online \
  general.name=test_heterogeneous_fix \
  model.diffusion_steps=1000 \
  train.num_workers=2 \
  general.sample_every_val=1 \
  general.samples_to_generate=1 \
  train.save_model=False \
  model.extra_features=null

echo ""
echo "=== 测试完成 ==="
