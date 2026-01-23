# 特征计算问题说明

## 问题位置

特征计算**不是去噪模型的问题**，而是**额外的特征工程**，用于增强模型的输入。

### 定义位置

1. **特征计算类**：`sparse_diffusion/diffusion/extra_features.py`
   - `EigenFeatures` 类：计算图的拉普拉斯矩阵的特征值和特征向量
   - `ExtraFeatures` 类：整合多种特征（邻接特征、位置编码、特征值特征等）

2. **调用位置**：`sparse_diffusion/diffusion_model_sparse.py`
   - `compute_extra_data()` 方法（第 1836 行）
   - 在 `forward()` 方法中调用（第 1363 行）

### 错误原因

错误发生在 `EigenFeatures.compute_features()` 中：
```python
eigvals, eigvectors = torch.linalg.eigh(L.cpu())
```

这是计算拉普拉斯矩阵特征值时出现的数值稳定性问题：
- 某些图的拉普拉斯矩阵条件数很差（ill-conditioned）
- 或者有太多重复的特征值
- 导致特征值分解算法无法收敛

### 配置方式

在配置文件中通过 `model.extra_features` 控制：
- `'all'`：使用所有特征（包括特征值特征）
- `None` 或其他值：使用 `DummyExtraFeatures`（不计算任何特征）

### 解决方案

1. **禁用特征值特征**：将 `model.extra_features` 设置为 `None`
2. **改进数值稳定性**：在 `EigenFeatures.compute_features()` 中添加错误处理
3. **使用更稳定的算法**：使用其他特征值分解方法

