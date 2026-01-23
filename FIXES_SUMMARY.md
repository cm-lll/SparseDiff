# 修复总结

## 已修复的问题

### 1. ✅ 训练时的查询边采样（高优先级）

**位置**：`diffusion_model_sparse.py` 第207-393行

**修复内容**：
- 在异质图模式下，按关系族分别进行均匀采样
- 使用 `|Eq| = km`，其中 `m` 是该关系族的平均真实边数（从 `edge_family_avg_edge_counts` 获取），`k = edge_fraction`（各关系族保持一致）
- 参考原项目实现（`sample_query_edges`），但区分关系族
- 为每个关系族分别计算可能的边数（基于端点类型）
- 使用类似 `sampled_condensed_indices_uniformly` 的方式均匀采样
- 合并所有关系族的查询边

**关键代码**：
```python
# 使用保存的平均真实边数 m_fam（各关系族保持一致）
m_fam = avg_m_fam

# 根据图片要求：|Eq| = km，其中 k 是倍数（通过 edge_fraction 控制，各关系族保持一致）
k = self.edge_fraction  # 倍数
num_query_edges_fam = int(math.ceil(k * m_fam)) if m_fam > 0 else 0
```

---

### 2. ✅ 采样时的真实边数统计（中优先级）

**位置**：
- `acm_subgraphs_dataset.py` 第447-475行：保存每个关系族的平均真实边数
- `acm_subgraphs_dataset.py` 第618-628行：加载每个关系族的平均真实边数
- `diffusion_model_sparse.py` 第2185-2192行：使用保存的平均真实边数

**修复内容**：
- 在 `process()` 方法中，统计每个关系族的平均真实边数并保存到 `train_edge_family_avg_counts.pickle`
- 在 `ACMSubgraphsInfos.__init__()` 中，加载保存的平均真实边数到 `edge_family_avg_edge_counts`
- 在采样时，使用保存的平均真实边数 `m_fam`，而不是从噪声后的边中统计
- `k` 是各关系族都保持一致的倍数（通过 `edge_fraction` 控制）

**关键代码**：
```python
# 在 process() 中保存
edge_family_avg_edge_counts[fam_name] = sum(counts) / len(counts)
save_pickle(edge_family_avg_edge_counts, edge_family_counts_path)

# 在采样时使用
edge_family_avg_edge_counts = getattr(self.dataset_info, "edge_family_avg_edge_counts", {})
if fam_name in edge_family_avg_edge_counts:
    avg_m_fam = edge_family_avg_edge_counts[fam_name]
```

---

## 实现细节

### 训练时的查询边采样

1. **按关系族分别采样**：
   - 为每个关系族分别计算可能的边数（基于端点类型 `src_type`, `dst_type`）
   - 使用保存的平均真实边数 `m_fam`
   - 计算 `|Eq_fam| = k * m_fam`，其中 `k = edge_fraction`

2. **均匀采样**：
   - 使用类似 `sampled_condensed_indices_uniformly` 的方式
   - 生成所有可能的边对，然后随机采样 `num_query_edges_fam` 条边

3. **合并查询边**：
   - 合并所有关系族的查询边
   - 与已有边合并：`Em = Et ∪ Eq`

### 采样时的真实边数统计

1. **保存平均真实边数**：
   - 在 `process()` 方法中，统计每个关系族的真实边数
   - 计算平均值并保存到 `train_edge_family_avg_counts.pickle`

2. **加载平均真实边数**：
   - 在 `ACMSubgraphsInfos.__init__()` 中，加载保存的平均真实边数
   - 如果没有保存的文件，使用默认值

3. **使用平均真实边数**：
   - 在采样时，使用保存的平均真实边数 `m_fam`
   - 计算 `|Eq_fam| = k * m_fam`，其中 `k = edge_fraction`（各关系族保持一致）

---

## 符合图片要求

✅ **|Eq| = km**：
- `m` 是各关系族在真实图中的数量（保存在 `edge_family_avg_edge_counts` 中）
- `k` 是各关系族都保持一致的倍数（通过 `edge_fraction` 控制）

✅ **按关系族分别采样**：
- 训练时和采样时都按关系族分别进行均匀采样
- 最后合并所有关系族的查询边

✅ **均匀采样**：
- 使用类似 `sampled_condensed_indices_uniformly` 的方式
- 不能人为偏向已有边，保持"unbiased estimator"的理论保证
