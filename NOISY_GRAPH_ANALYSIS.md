# 噪声图分析结果

## 噪声图结构

### 1. 稀疏格式 (sparse_noisy_data)
- `node_t`: (num_nodes, num_node_types) - one-hot 编码的节点特征
- `edge_index_t`: (2, num_edges) - 边索引
- `edge_attr_t`: (num_edges, num_edge_types) - one-hot 编码的边特征
- `batch`: (num_nodes,) - 批次信息
- `y_t`: (batch_size, 0) - 图级标签（空）

### 2. 密集格式 (dense_noisy_data)
- `X_t`: (batch_size, max_n_nodes, num_node_types) - 节点特征
- `E_t`: (batch_size, max_n_nodes, max_n_nodes, num_edge_types) - 边特征（密集邻接张量）
- `node_mask`: (batch_size, max_n_nodes) - 节点掩码
- `y_t`: (batch_size, 0) - 图级标签

## 问题分析

### 拉普拉斯矩阵的问题

从调试结果可以看到：

1. **条件数非常大** (3.59e+08 到 6.85e+08)
   - 说明矩阵是 ill-conditioned 的
   - 特征值范围很大，从接近0到20+

2. **大量重复特征值** (114-275个)
   - 图 0: 168个重复特征值
   - 图 1: 137个重复特征值
   - 图 2: 114个重复特征值
   - 图 3: 275个重复特征值
   - **这是导致特征值分解失败的主要原因**

3. **大量接近0的特征值** (47-86个)
   - 可能对应图的连通分量
   - 或者数值误差

### 为什么会出现这些问题？

1. **异质图特性**：
   - 异质图可能有多个连通分量
   - 不同关系族可能形成不同的子图结构

2. **噪声添加**：
   - 在训练过程中，噪声会随机添加/删除边
   - 可能导致图变得更加稀疏
   - 产生更多孤立节点或连通分量

3. **数值稳定性**：
   - 当图有很多连通分量时，拉普拉斯矩阵会有很多0特征值
   - 重复特征值会导致 `torch.linalg.eigh` 算法难以收敛

## 解决方案

1. **禁用特征值特征**（最简单）：
   ```yaml
   model:
     eigenfeatures: False
   ```

2. **改进数值稳定性**：
   - 在 `EigenFeatures.compute_features()` 中添加错误处理
   - 使用更稳定的特征值分解方法
   - 或者跳过有问题的图

3. **使用其他特征**：
   - 只使用邻接特征和位置编码
   - 不使用特征值特征

