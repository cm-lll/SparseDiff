#!/bin/bash
echo "=========================================="
echo "GPU内存使用情况检查"
echo "=========================================="
echo ""
echo "=== nvidia-smi 显示 ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{used=$2; total=$3; free=$4; util=$5; pct=int(used*100/total); printf "GPU %d: %d/%d MB (%.1f%% used, %d MB free, %d%% util)\n", $1, used, total, pct, free, util}'
echo ""
echo "=== 占用GPU的进程数量 ==="
count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
echo "当前有 $count 个进程占用GPU"
if [ $count -gt 0 ]; then
    echo ""
    echo "=== 占用GPU的进程详情 ==="
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -10
fi
