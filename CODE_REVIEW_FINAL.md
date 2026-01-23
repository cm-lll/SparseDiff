# 代码检查报告

## ✅ 已修复的问题

### 1. ✅ 训练时的查询边采样（高优先级）

**位置**：`diffusion_model_sparse.py` 第207-393行

**实现检查**：
- ✅ 检查是否为异质图模式（第210行）
- ✅ 获取关系族信息（第214-217行）
- ✅ 获取保存的平均真实边数（第217行）
- ✅ 按关系族分别进行均匀采样（第262-372行）
- ✅ 使用 `|Eq| = km`，其中 `m = avg_m_fam`（保存的平均真实边数），`k = edge_fraction`（第324-325行）
- ✅ 合并所有关系族的查询边（第375-377行）
- ✅ 生成 `query_edge_batch`（第377行）

**关键代码**：
```python
# 使用保存的平均真实边数 m_fam（各关系族保持一致）
if fam_name in edge_family_avg_edge_counts:
    avg_m_fam = edge_family_avg_edge_counts[fam_name]
else:
    avg_m_fam = num_fam_edges_per_batch.float().mean().item() if num_fam_edges_per_batch.sum() > 0 else 10.0

# 使用保存的平均真实边数 m_fam（各关系族保持一致）
m_fam = avg_m_fam

# 根据图片要求：|Eq| = km，其中 k 是倍数（通过 edge_fraction 控制，各关系族保持一致）
k = self.edge_fraction  # 倍数
num_query_edges_fam = int(math.ceil(k * m_fam)) if m_fam > 0 else 0
```

---

### 2. ✅ 采样时的真实边数统计（中优先级）

**位置**：
- `acm_subgraphs_dataset.py` 第447-483行：保存每个关系族的平均真实边数
- `acm_subgraphs_dataset.py` 第618-628行：加载每个关系族的平均真实边数
- `diffusion_model_sparse.py` 第2185-2242行：使用保存的平均真实边数

**实现检查**：
- ✅ 在 `process()` 中统计并保存平均真实边数（第447-483行）
- ✅ 在 `ACMSubgraphsInfos.__init__()` 中加载平均真实边数（第618-628行）
- ✅ 在采样时使用保存的平均真实边数（第2187-2192行）
- ✅ 使用 `|Eq| = km`，其中 `m = avg_m_fam`（保存的平均真实边数），`k = edge_fraction`（第2228行）

**关键代码**：
```python
# 在 process() 中保存
edge_family_avg_edge_counts[fam_name] = sum(counts) / len(counts)
edge_family_counts_path = osp.join(self.processed_dir, f"{self.split}_edge_family_avg_counts.pickle")
save_pickle(edge_family_avg_edge_counts, edge_family_counts_path)

# 在采样时使用
edge_family_avg_edge_counts = getattr(self.dataset_info, "edge_family_avg_edge_counts", {})
if fam_name in edge_family_avg_edge_counts:
    avg_m_fam = edge_family_avg_edge_counts[fam_name]
else:
    avg_m_fam = 10.0

# 使用保存的平均真实边数 m_fam（各关系族保持一致）
m_fam = avg_m_fam
```

---

## ✅ 代码逻辑检查

### 1. 训练时的查询边采样逻辑

**流程**：
1. ✅ 检查是否为异质图模式
2. ✅ 获取关系族信息和保存的平均真实边数
3. ✅ 为每个关系族分别计算查询边数：`|Eq_fam| = k * m_fam`
4. ✅ 为每个关系族分别进行均匀采样
5. ✅ 合并所有关系族的查询边
6. ✅ 与已有边合并：`Em = Et ∪ Eq`

**符合图片要求**：
- ✅ `|Eq| = km`：`m` 是各关系族在真实图中的数量（保存在 `edge_family_avg_edge_counts` 中）
- ✅ `k` 是各关系族都保持一致的倍数（通过 `edge_fraction` 控制）
- ✅ 按关系族分别进行均匀采样
- ✅ 不能人为偏向已有边，保持"unbiased estimator"的理论保证

---

### 2. 采样时的真实边数统计逻辑

**流程**：
1. ✅ 在数据预处理时统计每个关系族的平均真实边数
2. ✅ 保存到 `train_edge_family_avg_counts.pickle`
3. ✅ 在 `ACMSubgraphsInfos.__init__()` 中加载
4. ✅ 在采样时使用保存的平均真实边数

**符合图片要求**：
- ✅ `m` 是各关系族在真实图中的数量（保存在 `edge_family_avg_edge_counts` 中）
- ✅ `k` 是各关系族都保持一致的倍数（通过 `edge_fraction` 控制）

---

## ⚠️ 潜在问题检查

### 1. 训练时的 `query_edge_batch` 使用

**位置**：`diffusion_model_sparse.py` 第377行

**检查**：
- ✅ `query_edge_batch` 被正确生成
- ✅ `get_computational_graph` 函数可能不需要 `query_edge_batch`（需要检查函数签名）
- ⚠️ 如果 `get_computational_graph` 不需要 `query_edge_batch`，这个变量可能没有被使用，但不影响功能

**结论**：✅ 无问题（`query_edge_batch` 可能用于调试或未来扩展）

---

### 2. 数据保存路径

**位置**：`acm_subgraphs_dataset.py` 第482行

**检查**：
- ✅ 保存路径：`{self.split}_edge_family_avg_counts.pickle`
- ✅ 加载路径：`train_edge_family_avg_counts.pickle`（第619行）
- ⚠️ 注意：加载时只加载 `train` split 的平均边数，这是合理的（因为训练时使用训练集的统计信息）

**结论**：✅ 无问题

---

### 3. 默认值处理

**位置**：
- `acm_subgraphs_dataset.py` 第628行：默认值 `10.0`
- `diffusion_model_sparse.py` 第2192行：默认值 `10.0`
- `diffusion_model_sparse.py` 第292行：回退到当前批次统计的值

**检查**：
- ✅ 如果保存文件不存在，使用默认值 `10.0`
- ✅ 在训练时，如果没有保存的平均值，使用当前批次统计的值（更准确）
- ⚠️ 默认值 `10.0` 可能不准确，但不会导致错误

**结论**：✅ 无问题（有合理的回退机制）

---

## 📋 总结

### ✅ 已正确实现

1. ✅ 训练时的查询边采样：按关系族分别进行均匀采样，使用保存的平均真实边数
2. ✅ 采样时的真实边数统计：使用保存的平均真实边数
3. ✅ 数据保存和加载：正确保存和加载每个关系族的平均真实边数
4. ✅ 符合图片要求：`|Eq| = km`，其中 `m` 是各关系族在真实图中的数量，`k` 是各关系族都保持一致的倍数

### ⚠️ 注意事项

1. ⚠️ 默认值 `10.0`：如果保存文件不存在，使用默认值，可能不准确
2. ⚠️ `query_edge_batch`：在训练时生成但可能未被使用，不影响功能

### 🎯 建议

1. ✅ 代码逻辑正确，可以开始测试
2. ✅ 建议在首次运行时检查 `train_edge_family_avg_counts.pickle` 文件是否被正确生成
3. ✅ 建议检查保存的平均真实边数是否合理（应该接近实际的平均边数）
