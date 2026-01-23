#!/bin/bash
echo "=========================================="
echo "GPU占用情况汇总"
echo "=========================================="
echo ""
echo "=== 当前GPU使用情况 ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -8
echo ""
echo "=== 占用GPU的进程（按内存排序） ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sort -t',' -k3 -rn | head -10
echo ""
echo "=== 主要进程详情 ==="
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -5); do
    if [ ! -z "$pid" ]; then
        echo "--- PID: $pid ---"
        ps -p $pid -o user,pid,etime,start,cmd --no-headers 2>/dev/null | head -1
        echo ""
    fi
done
