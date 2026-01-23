# 异质图边扩散实现（方式2：类别隔离空间）

## 概述

实现了两种边扩散模式：
- **异质图模式（heterogeneous=True）**：每条边在自己的关系族（family）空间内扩散子类别，通过 offset 机制实现类别隔离
- **同质图模式（heterogeneous=False）**：所有边在全局空间内扩散（原始实现）

## 配置参数

在 `configs/dataset/acm_subgraphs.yaml` 中添加了 `heterogeneous` 参数：

```yaml
heterogeneous: True  # True: 异质图（类别隔离空间）
                     # False: 同质图（全局空间）
```

## 实现细节

### 1. 词汇表构建（`_build_vocab_from_meta`）

**异质图模式**：
- 为每个关系族计算 offset（类似节点的 type_offsets）
- 每个关系族的子类别 ID 范围是独立的
- 例如：
  - `affiliated_with`: offset=1, 范围 [1, 1]（1 个子类别：`__none__`）
  - `author_of`: offset=2, 范围 [2, 4]（3 个子类别：first_author, second_author, co_author）
  - `cites`: offset=5, 范围 [5, 5]（1 个子类别：`__none__`）

**同质图模式**：
- 所有边标签共享一个全局 ID 空间
- 从 1 开始顺序分配（0 保留给"无边"）

### 2. 边构建（`_build_edges_for_graph`）

**异质图模式**：
- 使用 `edge_family_offsets[fam] + local_id` 计算全局边标签 ID
- 添加 `edge_family` 字段到 Data 对象，记录每条边所属的关系族 ID（用于约束）

**同质图模式**：
- 使用 `edge_label2id[label_str]` 直接映射（原始实现）

### 3. Data 对象

异质图模式下，Data 对象包含：
- `edge_attr`: 全局边标签 ID（使用 offset 机制）
- `edge_family`: 关系族 ID（用于约束，确保扩散时不会跨关系族）

## 使用示例

```python
# 异质图模式（默认）
datamodule = ACMSubgraphsDataModule(cfg)  # cfg.dataset.heterogeneous = True

# 同质图模式
cfg.dataset.heterogeneous = False
datamodule = ACMSubgraphsDataModule(cfg)
```

## 注意事项

1. **缓存兼容性**：如果更改 `heterogeneous` 设置，需要删除 `processed/` 目录下的缓存文件
2. **meta.json 无需修改**：实现直接从现有的 `meta.json` 结构计算所需信息
3. **向后兼容**：同质图模式保持与原始实现一致

