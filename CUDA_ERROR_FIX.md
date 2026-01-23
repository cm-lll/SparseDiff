# CUDA 错误修复

## 问题分析

1. **多个训练任务同时运行**：检测到有多个训练进程在运行，包括：
   - 一个8 GPU的训练任务（batch_size=512）
   - 之前的训练任务仍在运行
   - 所有GPU使用率都是100%

2. **CUDA错误类型**：`CUDA error: device-side assert triggered`
   - 通常由索引越界引起
   - 也可能是多个进程竞争GPU资源

## 修复措施

### 1. 修复索引越界问题

在 `HeterogeneousGraphFeatures` 中添加索引边界检查：
- 确保 `offset` 和 `next_offset` 在有效范围内 `[0, de]`
- 防止切片操作越界

### 2. 建议

**停止其他训练任务**，只运行一个测试任务：
```bash
# 查看正在运行的训练任务
ps aux | grep "main.py" | grep -v grep

# 如果需要，可以停止之前的训练任务（谨慎操作）
# kill <PID>
```

**使用 CPU 模式进行测试**：
```bash
general.gpus=0  # 使用CPU，避免GPU竞争
```

