# 采样逻辑验证

## 用户需求总结

### 第一种情况：噪声模型（加噪过程）- `apply_sparse_noise`
- **目的**：除了已有边外，单独采样不存在的边（分关系族进行）
- **方法**：每条None转为类别边的情况服从伯努利分布
- **要求**：按关系族进行采样

### 第二种情况：去噪模型中的采样 - `sample_p_zs_given_zt`
- **目的**：在去噪时，需要添加额外的None边来增强信息传递
- **消息传递边**：Em = Et ∪ Eq
  - Et：已有的噪声边
  - Eq：对所有边均匀采样得到的（可能会采样到已有的噪声边，但没问题）
- **要求**：在每个关系族中进行

---

## 当前实现检查

### 第一种情况：噪声模型（加噪过程）

**位置**：`apply_sparse_noise` 方法（约第946-1087行）

**当前实现**：
```python
for fam_id, fam_name in id2edge_family.items():
    # ... 处理已存在的边 ...
    
    # Step2: 计算该关系族的非存在边数并采样（按关系族隔离的伯努利采样）
    # 计算该关系族的 emerge_prob（qt_fam）
    emerge_prob_fam = Qtb_fam.E[:, 0, 1:].sum(-1)  # (bs, )
    
    # 使用二项分布采样：k_fam ~ B(m̄_fam, qt_fam)
    num_emerge_edges_fam = (
        torch.distributions.binomial.Binomial(num_fam_neg_edge.long(), emerge_prob_fam)
        .sample()
        .int()
    )
    
    # 采样非存在的边（需要考虑端点类型约束）
    neg_edge_index_fam = sample_non_existing_edges_batched_heterogeneous(
        num_edges_to_sample=num_emerge_edges_fam,
        existing_edge_index=dir_edge_index,  # 所有关系族的已存在边
        ...
    )
```

**验证**：
- ✅ 按关系族进行循环处理
- ✅ 使用伯努利分布（二项分布）采样：`k_fam ~ B(m̄_fam, qt_fam)`
- ✅ 考虑端点类型约束（`src_mask`, `dst_mask`）
- ✅ 排除所有已存在的边（`existing_edge_index=dir_edge_index`）

**结论**：✅ **符合要求**

---

### 第二种情况：去噪模型中的采样

**位置**：`sample_p_zs_given_zt` 方法（约第1937-2027行）

**当前实现**：
```python
# 对所有边均匀采样
(
    all_condensed_index,
    all_edge_batch,
    all_edge_mask,
) = sampled_condensed_indices_uniformly(
    max_condensed_value=num_edges,
    num_edges_to_sample=num_edges,
    return_mask=True,
)

# 在循环中处理
for i in range(len_loop):
    # 获取查询边
    triu_query_edge_index = all_condensed_index[edges_to_consider_mask]
    
    # 合并查询边和已有边
    query_mask, comp_edge_index, comp_edge_attr = get_computational_graph(
        triu_query_edge_index=triu_query_edge_index,
        clean_edge_index=sparse_noisy_data["edge_index_t"],  # Et：已有的噪声边
        clean_edge_attr=sparse_noisy_data["edge_attr_t"],
    )
    # comp_edge_index = Et ∪ Eq（查询边）
```

**验证**：
- ✅ 对所有边均匀采样（`sampled_condensed_indices_uniformly`）
- ✅ 合并查询边和已有边（`get_computational_graph`）
- ✅ 可能会采样到已有的噪声边（`get_computational_graph` 会去重）
- ❌ **问题**：没有按关系族进行采样

**问题分析**：
- 当前实现是对所有边进行全局均匀采样
- 用户要求：在每个关系族中进行采样
- 这意味着应该为每个关系族分别采样查询边，而不是全局采样

**需要修复**：
1. 为每个关系族分别计算可能的边数
2. 为每个关系族分别进行均匀采样
3. 合并所有关系族的查询边

---

## 修复建议

### 第二种情况的修复方案

在 `sample_p_zs_given_zt` 方法中，需要：

1. **按关系族分组**：
   - 识别每个关系族的可能边（基于端点类型）
   - 为每个关系族分别计算 `num_edges_fam`

2. **按关系族采样**：
   - 为每个关系族分别调用 `sampled_condensed_indices_uniformly`
   - 采样数量：`num_edges_fam`（每个关系族的所有可能边）

3. **合并查询边**：
   - 合并所有关系族的查询边
   - 然后与已有边合并：`comp_edge_index = Et ∪ Eq`

4. **计算转移概率**：
   - 当前已经按关系族计算转移概率（第2031-2137行）
   - 这部分是正确的

---

## 总结

### ✅ 第一种情况（噪声模型）：符合要求
- 按关系族进行伯努利采样
- 排除所有已存在的边
- 考虑端点类型约束

### ❌ 第二种情况（去噪模型）：需要修复
- 当前是对所有边全局均匀采样
- 应该按关系族分别进行均匀采样
- 需要修改 `sample_p_zs_given_zt` 方法
