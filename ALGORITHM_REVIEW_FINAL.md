# 算法和模型实现检查报告

## 1. 训练时的查询边采样（`training_step`）

**位置**：第207-217行

**当前实现**：
```python
triu_query_edge_index, _ = sample_query_edges(
    num_nodes_per_graph=data.ptr.diff(), edge_proportion=self.edge_fraction
)
```

**问题**：
- ❌ **对于异质图，应该按关系族分别进行均匀采样，但当前实现是全局采样**
- 这会导致训练和采样时的不一致

**建议修复**：
- 在训练时也按关系族分别进行均匀采样
- 使用 `|Eq| = km`，其中 `m` 是该关系族的真实边数，`k = edge_fraction`

---

## 2. 采样时的真实边数统计（`sample_p_zs_given_zt`）

**位置**：第1986-2020行

**当前实现**：
```python
edge_attr_discrete = edge_attr.argmax(dim=-1) if edge_attr.dim() > 1 else edge_attr  # (E,)
# 统计每个批次中该关系族的真实边数 m_fam
fam_edge_mask = (edge_attr_discrete >= offset) & (edge_attr_discrete < next_offset)
fam_edge_index = edge_index[:, fam_edge_mask]
```

**问题**：
- ⚠️ **使用 `edge_index` 和 `edge_attr`（这是 `data.edge_index` 和 `data.edge_attr`，即噪声后的边）**
- 根据图片要求，`|Eq| = km` 中的 `m` 应该是**真实的边数**，而不是噪声后的边数
- 但在采样时，我们只有噪声后的边，没有真实的边

**分析**：
- 在采样过程中，我们确实只有噪声后的边
- 但根据图片要求，`|Eq| = km` 中的 `m` 应该是真实的边数
- 这可能需要在训练时预先计算每个关系族的平均真实边数，或者使用其他方法

**建议**：
- 可以考虑使用噪声后的边数作为近似（因为噪声是逐步添加的，早期噪声较小）
- 或者使用全局平均边数作为参考

---

## 3. 采样时的循环处理（`sample_p_zs_given_zt`）

**位置**：第2149-2151行

**当前实现**：
```python
num_edges_per_loop = torch.ceil(self.edge_fraction * num_edges)  # (bs, )
len_loop = math.ceil(1.0 / self.edge_fraction)
```

**问题**：
- ⚠️ **在异质图模式下，`num_edges` 被设置为 `num_edges_per_batch.max()`（第2126行）**
- 这可能导致循环处理时边数计算不准确

**分析**：
- 在异质图模式下，`num_edges` 应该是所有关系族的查询边总数，而不是每个批次的最大值
- 但循环处理时，需要确保每个循环处理的边数是一致的

**建议**：
- 使用所有关系族的查询边总数作为 `num_edges`
- 或者为每个批次分别计算 `num_edges_per_loop`

---

## 4. 采样时的查询边分批处理（`sample_p_zs_given_zt`）

**位置**：第2178-2202行

**当前实现**：
```python
if i == 0:
    # 第一次循环，打乱所有查询边（实现均匀采样）
    perm = torch.randperm(num_query_edges_total, device=self.device)
    all_query_edge_index = all_query_edge_index[:, perm]
    all_query_edge_batch = all_query_edge_batch[perm]

# 计算当前循环要采样的边索引范围
num_query_edges_per_loop = int(math.ceil(num_query_edges_total * self.edge_fraction))
start_idx = i * num_query_edges_per_loop
end_idx = min((i + 1) * num_query_edges_per_loop, num_query_edges_total)
```

**问题**：
- ⚠️ **`num_query_edges_per_loop` 的计算可能不准确**
- 如果 `num_query_edges_total` 很大，`num_query_edges_per_loop` 可能会超过每个批次的实际查询边数

**建议**：
- 需要按批次分别计算 `num_query_edges_per_loop`
- 或者确保每个循环处理的边数不超过每个批次的实际查询边数

---

## 5. 损失计算中的边选择（`compute_Lt`）

**位置**：第1375-1377行

**当前实现**：
```python
# 判断哪些边属于这个关系族（使用真实的 E，而不是噪声后的 E_t）
fam_mask = (E_discrete == 0) | ((E_discrete >= offset) & (E_discrete < next_offset))
```

**状态**：
- ✅ **已修复**：使用真实的 `E` 来判断关系族，而不是噪声后的 `E_t`

---

## 6. 采样过程中的边族判断（`sample_p_zs_given_zt`）

**位置**：第2263-2297行

**当前实现**：
```python
comp_edge_attr_discrete = comp_edge_attr.argmax(dim=-1)  # (NE,)
# 根据边的全局 ID 推断 edge_family
fam_mask = (comp_edge_attr_discrete == 0) | ((comp_edge_attr_discrete >= offset) & (comp_edge_attr_discrete < next_offset))
```

**状态**：
- ✅ **逻辑正确**：在采样过程中，使用 `comp_edge_attr`（噪声后的边属性）来判断关系族是合理的
- 因为 `p(z_s | z_t)` 应该只依赖于 `z_t`（噪声后的状态）

---

## 7. 非存在边采样（`apply_sparse_noise`）

**位置**：约第946-1087行

**状态**：
- ✅ **已实现**：按关系族分别进行伯努利采样
- ✅ **已实现**：考虑端点类型约束
- ✅ **已实现**：排除所有已存在的边

---

## 总结

### 高优先级问题

1. **训练时的查询边采样**（第207-217行）：
   - ❌ 对于异质图，应该按关系族分别进行均匀采样
   - 需要修复

2. **采样时的真实边数统计**（第1986-2020行）：
   - ⚠️ 使用噪声后的边数作为真实边数的近似
   - 可能需要优化

### 中优先级问题

3. **采样时的循环处理**（第2149-2151行）：
   - ⚠️ `num_edges` 的计算可能不准确
   - 需要优化

4. **采样时的查询边分批处理**（第2178-2202行）：
   - ⚠️ `num_query_edges_per_loop` 的计算可能不准确
   - 需要优化

### 已修复/正确实现

5. ✅ 损失计算中的边选择（使用真实的 `E`）
6. ✅ 采样过程中的边族判断（使用 `comp_edge_attr`）
7. ✅ 非存在边采样（按关系族分别进行）
