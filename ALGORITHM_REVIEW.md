# 算法和模型问题审查

## 1. 损失计算中使用噪声边判断关系族 ⚠️ **重要问题**

### 问题位置
`diffusion_model_sparse.py` 的 `compute_Lt` 方法（约第1373行）

### 当前实现
```python
E_t = noisy_data["E_t"]  # (bs, n, n, de) - 噪声后的边
E_t_discrete = E_t.argmax(dim=-1)  # (bs, n, n) - 全局ID
# ...
fam_mask = (E_t_discrete == 0) | ((E_t_discrete >= offset) & (E_t_discrete < next_offset))
```

### 问题描述
- 使用 `E_t`（噪声后的边）来判断哪些边属于哪个关系族
- 噪声可能会改变边的全局ID，导致错误地将边归类到错误的关系族
- 应该使用真实的 `E`（未加噪声的边）来判断关系族

### 影响
- **严重**：可能导致损失计算时使用错误的转移矩阵
- 训练不稳定或收敛慢

### 建议修复
```python
# 使用真实的 E 来判断关系族
E_discrete = E.argmax(dim=-1)  # (bs, n, n) - 全局ID
fam_mask = (E_discrete == 0) | ((E_discrete >= offset) & (E_discrete < next_offset))
```

---

## 2. 非存在边采样时使用了所有关系族的已存在边 ⚠️ **重要问题**

### 问题位置
`diffusion_model_sparse.py` 的 `apply_sparse_noise` 方法（约第1059行）

### 当前实现
```python
neg_edge_index_fam = sample_non_existing_edges_batched_heterogeneous(
    num_edges_to_sample=num_emerge_edges_fam,
    existing_edge_index=dir_edge_index,  # 所有关系族的已存在边
    ...
)
```

### 问题描述
- 对于每个关系族，传入的 `existing_edge_index` 是所有关系族的已存在边的合并
- 应该只传入当前关系族的已存在边（`fam_dir_edge_index`）
- 否则，可能会错误地排除其他关系族的边，导致采样不准确

### 影响
- **中等**：可能导致非存在边采样不准确
- 影响训练数据的分布

### 建议修复
```python
# 只传入当前关系族的已存在边
neg_edge_index_fam = sample_non_existing_edges_batched_heterogeneous(
    num_edges_to_sample=num_emerge_edges_fam,
    existing_edge_index=fam_dir_edge_index,  # 只包含当前关系族的边
    ...
)
```

---

## 3. 采样过程中使用 comp_edge_attr 判断关系族 ⚠️ **潜在问题**

### 问题位置
`diffusion_model_sparse.py` 的 `sample_p_zs_given_zt` 方法（约第2030行）

### 当前实现
```python
comp_edge_attr_discrete = comp_edge_attr.argmax(dim=-1)  # (NE,)
# ...
fam_mask = (comp_edge_attr_discrete == 0) | ((comp_edge_attr_discrete >= offset) & (comp_edge_attr_discrete < next_offset))
```

### 问题描述
- `comp_edge_attr` 包含查询边（query edges），这些边的默认属性是 no-edge (0)
- 使用 `comp_edge_attr_discrete` 来判断关系族时，查询边会被归类为 no-edge，这是正确的
- 但如果有其他边也被错误地归类，可能会导致问题

### 影响
- **较小**：查询边默认是 no-edge，这是正确的
- 但需要确保逻辑的一致性

### 建议
- 当前实现应该是正确的，因为查询边的默认属性是 no-edge
- 但可以考虑添加注释说明这一点

---

## 4. 损失计算中的边选择逻辑 ⚠️ **需要确认**

### 问题位置
`diffusion_model_sparse.py` 的 `compute_Lt` 方法（约第1391行）

### 当前实现
```python
E_b_fam = E[b][batch_fam_mask]  # (num_edges_b_fam, de)
E_t_b_fam = E_t[b][batch_fam_mask]  # (num_edges_b_fam, de)
pred_E_b_fam = pred_probs_E[b][batch_fam_mask]  # (num_edges_b_fam, de)
```

### 问题描述
- 使用 `batch_fam_mask`（基于 `E_t_discrete`）来选择边
- 但 `E` 和 `pred_probs_E` 应该使用相同的 mask
- 如果 `fam_mask` 是基于 `E_t` 的，而 `E` 的真实关系族可能不同，这会导致问题

### 影响
- **严重**：如果 `fam_mask` 基于错误的判断，会导致使用错误的边计算损失

### 建议修复
```python
# 使用真实的 E 来判断关系族
E_discrete = E.argmax(dim=-1)  # (bs, n, n)
fam_mask = (E_discrete == 0) | ((E_discrete >= offset) & (E_discrete < next_offset))
# 然后使用这个 mask 来选择边
```

---

## 5. 采样过程中的边族判断可能不准确 ⚠️ **需要确认**

### 问题位置
`diffusion_model_sparse.py` 的 `sample_p_zs_given_zt` 方法（约第2044-2067行）

### 当前实现
```python
comp_edge_attr_discrete = comp_edge_attr.argmax(dim=-1)  # (NE,)
# ...
fam_mask = (comp_edge_attr_discrete == 0) | ((comp_edge_attr_discrete >= offset) & (comp_edge_attr_discrete < next_offset))
```

### 问题描述
- 在采样过程中，`comp_edge_attr` 包含查询边和已存在边
- 查询边的默认属性是 no-edge (0)，这是正确的
- 但已存在边的属性可能已经被噪声改变，使用 `comp_edge_attr_discrete` 来判断关系族可能不准确

### 影响
- **中等**：如果已存在边的属性被噪声改变，可能会导致使用错误的转移矩阵

### 建议
- 在采样过程中，应该使用原始边的属性（如果有的话）来判断关系族
- 或者，需要确保 `comp_edge_attr` 中的已存在边属性是正确的

---

## 6. 非存在边采样函数的效率问题 ⚠️ **性能问题**

### 问题位置
`sample_edges.py` 的 `sample_non_existing_edges_batched_heterogeneous` 方法

### 当前实现
```python
# 使用向量化比较，但可能效率不高
src_match = (all_possible_edges[0:1, :].T == existing_edges_b[0:1, :])  # (num_possible, E_b)
dst_match = (all_possible_edges[1:2, :].T == existing_edges_b[1:2, :])  # (num_possible, E_b)
```

### 问题描述
- 对于每个批次，需要比较所有可能的边与所有已存在的边
- 如果 `num_possible` 很大（例如，1000个src节点 × 1000个dst节点 = 1,000,000个可能的边），这会导致内存和计算开销很大

### 影响
- **性能**：对于大型图，可能导致内存不足或计算缓慢

### 建议优化
- 使用更高效的算法，例如使用哈希表或集合来存储已存在的边
- 或者，限制采样的边数，避免创建所有可能的边对

---

## 优先级总结

### 高优先级 🔴
1. **问题1**：损失计算中使用噪声边判断关系族
2. **问题2**：非存在边采样时使用了所有关系族的已存在边
3. **问题4**：损失计算中的边选择逻辑

### 中优先级 🟡
4. **问题5**：采样过程中的边族判断可能不准确

### 低优先级 🟢
5. **问题3**：采样过程中使用 comp_edge_attr 判断关系族（当前实现应该是正确的）
6. **问题6**：非存在边采样函数的效率问题（性能优化）

---

## 建议修复顺序

1. 首先修复问题1和问题4（损失计算中的关系族判断）
2. 然后修复问题2（非存在边采样时的 existing_edge_index）
3. 最后优化问题6（性能问题）
