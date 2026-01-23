#!/usr/bin/env python3
"""
检查实际使用的GPU设备
"""
import torch
import os

# 模拟设置 CUDA_VISIBLE_DEVICES=2
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

print("=" * 50)
print("GPU 设备检查")
print("=" * 50)
print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"PyTorch 可见 GPU 数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    print(f"当前设备索引 (PyTorch): {current_device}")
    print(f"设备名称: {torch.cuda.get_device_name(current_device)}")
    
    # 获取设备属性
    props = torch.cuda.get_device_properties(current_device)
    print(f"总内存: {props.total_memory / 1024**3:.2f} GB")
    
    print("\n说明:")
    print("- 当设置 CUDA_VISIBLE_DEVICES=2 时，物理 GPU 2 会被重新映射为 PyTorch 的设备 0")
    print("- 所以 'GR0' (Global Rank 0) 实际上使用的是物理 GPU 2")
    print("- 这是 PyTorch Lightning 的分布式训练中的概念")
else:
    print("CUDA 不可用")
