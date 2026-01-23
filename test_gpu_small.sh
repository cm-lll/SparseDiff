#!/bin/bash
# 小规模GPU测试（最小配置避免OOM）

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
# 解析GPU信息，选择内存使用最少的GPU
BEST_GPU=""
BEST_FREE_MEM=0

while IFS=',' read -r idx mem_used mem_total util; do
    # 清理空格
    idx=$(echo $idx | xargs)
    mem_used=$(echo $mem_used | xargs)
    mem_total=$(echo $mem_total | xargs)
    util=$(echo $util | xargs)
    
    # 计算可用内存
    mem_free=$((mem_total - mem_used))
    
    # 如果这个GPU的可用内存更多，选择它
    if [ $mem_free -gt $BEST_FREE_MEM ]; then
        BEST_GPU=$idx
        BEST_FREE_MEM=$mem_free
    fi
    
    echo "GPU $idx: 已用 ${mem_used}MB / ${mem_total}MB, 可用 ${mem_free}MB, 利用率 ${util}%"
done <<< "$GPU_INFO"

if [ -z "$BEST_GPU" ]; then
    echo "警告: 无法检测GPU，使用默认GPU 0"
    BEST_GPU=0
    BEST_FREE_MEM=0
fi

echo ""
echo "✓ 选择 GPU $BEST_GPU (可用内存: ${BEST_FREE_MEM}MB)"

# 支持多GPU：如果可用内存足够，可以选择多个GPU
# 检查是否有其他GPU也有足够的可用内存（至少5GB）
MULTI_GPU_LIST="$BEST_GPU"
THRESHOLD_MB=5120  # 5GB阈值

while IFS=',' read -r idx mem_used mem_total util; do
    idx=$(echo $idx | xargs)
    mem_used=$(echo $mem_used | xargs)
    mem_total=$(echo $mem_total | xargs)
    
    mem_free=$((mem_total - mem_used))
    
    # 如果这个GPU不是已选择的，且有足够内存，添加到列表
    if [ "$idx" != "$BEST_GPU" ] && [ $mem_free -ge $THRESHOLD_MB ]; then
        MULTI_GPU_LIST="$MULTI_GPU_LIST,$idx"
    fi
done <<< "$GPU_INFO"

# 统计可用GPU数量
NUM_GPUS=$(echo $MULTI_GPU_LIST | tr ',' '\n' | wc -l)
echo "✓ 检测到 $NUM_GPUS 个可用GPU: $MULTI_GPU_LIST"

# 设置CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$MULTI_GPU_LIST
echo "✓ 设置 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo ""
echo "=== 开始GPU测试（最小配置） ==="
# 使用最小的batch_size和配置避免OOM
python3 sparse_diffusion/main.py \
  +experiment=debug.yaml \
  dataset.name=acm_subgraphs \
  dataset.datadir=data/ACM_subgraphs \
  train.n_epochs=2 \
  train.batch_size=1 \
  train.lr=0.0002 \
  general.gpus=$NUM_GPUS \
  general.wandb=online \
  general.name=test_heterogeneous_sampling_improved \
  model.diffusion_steps=1000 \
  train.num_workers=2 \
  general.sample_every_val=1 \
  general.samples_to_generate=1 \
  train.save_model=False \
  model.extra_features=null
