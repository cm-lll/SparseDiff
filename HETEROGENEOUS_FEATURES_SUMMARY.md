# 异质图特征实现总结

## 问题

`EigenFeatures` 是为同质图设计的，它会将所有边类型合并，导致异质图丢失重要的异质信息。

## 解决方案

创建了 `HeterogeneousGraphFeatures` 类，专门为异质图设计，保留异质信息。

## 实现特点

### 1. 关系类型特定的度特征（替代特征向量）

- 为每种关系类型（edge family）分别计算：
  - **入度**：节点通过该关系类型接收的边数
  - **出度**：节点通过该关系类型发出的边数
  - **总度**：入度 + 出度
- 保留异质信息，不合并不同关系类型
- 输出维度：`(bs, n, num_eigenvectors)`

### 2. 关系类型统计特征（替代特征值）

- 为每种关系类型计算：
  - **边数量**：该关系类型在图中出现的总次数
  - **平均度**：该关系类型的平均节点度
- 计算连通分量数量（使用全局邻接矩阵）
- 输出维度：`(bs, num_eigenvalues + 1)`

## 优势

1. **保留异质信息**：不合并不同关系类型，每种关系类型单独计算特征
2. **兼容接口**：与 `EigenFeatures` 接口一致，可以直接替换
3. **自动适配**：在 `ExtraFeatures` 中自动检测异质图并使用 `HeterogeneousGraphFeatures`

## 使用方式

无需修改配置，代码会自动：
- 检测是否为异质图（通过 `dataset_info.heterogeneous`）
- 如果是异质图，使用 `HeterogeneousGraphFeatures`
- 如果是同质图，使用 `EigenFeatures`

## 特征维度

- **节点特征**（evector_feat）：`(bs, n, num_eigenvectors)`
- **图特征**（evalue_feat）：`(bs, num_eigenvalues + 1)`

与 `EigenFeatures` 的输出维度一致，确保模型输入维度不变。

