# 算法和模型潜在问题分析

## 1. 非存在边采样精度问题 ⚠️ **重要**

### 问题位置
`diffusion_model_sparse.py` 的 `apply_sparse_noise` 方法（约第950行）

### 当前实现
```python
# 简化处理：使用全局的非存在边数按比例分配
# TODO: 更精确的计算需要根据节点类型统计
num_fam_neg_edge = num_neg_edge  # 简化：使用全局值
```

### 问题描述
1. **非存在边数计算不精确**：每个关系族的非存在边数应该根据端点类型计算
   - 例如：`author_of` 关系族只能连接 `Author` 和 `Paper` 节点
   - 当前使用全局 `num_neg_edge`，没有考虑节点类型约束

2. **采样非存在边时没有端点类型约束**：
   - `sample_non_existing_edges_batched` 函数采样时没有考虑端点类型
   - 可能采样到不符合关系族端点类型的边

### 影响
- **训练阶段**：可能学习到错误的边分布
- **采样阶段**：可能生成不符合异质图约束的边

### 建议修复
```python
# 根据端点类型计算该关系族的非存在边数
if fam_name in fam_endpoints:
    src_type = fam_endpoints[fam_name]["src_type"]
    dst_type = fam_endpoints[fam_name]["dst_type"]
    # 计算该关系族可能的边数
    src_nodes = (data.node.argmax(-1) == src_type_id).sum()  # 源节点数
    dst_nodes = (data.node.argmax(-1) == dst_type_id).sum()  # 目标节点数
    num_fam_possible_edges = src_nodes * dst_nodes  # 可能的边数
    num_fam_existing_edges = fam_mask.sum()  # 已存在的边数
    num_fam_neg_edge = num_fam_possible_edges - num_fam_existing_edges
```

---

## 2. KL 先验计算使用全局转移矩阵 ⚠️ **次要**

### 问题位置
`diffusion_model_sparse.py` 的 `kl_prior` 方法（约第1125行）

### 当前实现
```python
# 对于异质图，KL 先验计算使用全局转移矩阵作为近似
# 因为 KL 先验通常值很小，对训练影响不大
Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
```

### 问题描述
- 对于异质图，应该按关系族分别计算 KL 先验
- 当前使用全局转移矩阵，理论上不够精确

### 影响
- **较小**：KL 先验值通常很小，对训练影响不大
- 但理论上应该按关系族计算

### 建议修复
```python
if self.heterogeneous:
    # 按关系族计算 KL 先验
    all_family_qtb = self.transition_model.get_all_family_Qt_bar(alpha_t_bar, self.device)
    # 为每个关系族分别计算 KL，然后求和
else:
    # 同质图模式：使用全局转移矩阵
    Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
```

---

## 3. limit_dist 的构建 ⚠️ **需要确认**

### 问题位置
`diffusion_model_sparse.py` 的 `__init__` 方法（约第183行）

### 当前实现
```python
self.limit_dist = utils.PlaceHolder(
    X=x_marginals,
    E=e_marginals,  # 全局边类型分布
    y=torch.ones(self.out_dims.y) / self.out_dims.y,
    charge=charge_marginals,
)
```

### 问题描述
- `limit_dist.E` 使用全局 `e_marginals`
- 对于异质图，`limit_dist.E` 应该是全局均匀分布（所有边类型等概率）
- 需要确认 `e_marginals` 是否正确反映了异质图的全局分布

### 影响
- **较小**：如果 `e_marginals` 是正确的全局分布，则没问题
- 需要确认数据加载时 `e_marginals` 的计算是否正确

---

## 4. 采样过程中的非存在边处理 ⚠️ **需要检查**

### 问题位置
`diffusion_model_sparse.py` 的 `sample_p_zs_given_zt` 方法（约第1825行）

### 问题描述
- 采样过程中，非存在边的处理可能也需要考虑关系族
- 当前实现主要关注已存在边的处理

### 影响
- **较小**：采样时非存在边的处理相对简单
- 但需要确认是否完全正确

---

## 优先级建议

### 高优先级 🔴
1. **非存在边采样精度问题**（问题1）
   - 影响训练和采样的正确性
   - 需要根据端点类型精确计算非存在边数

### 中优先级 🟡
2. **KL 先验计算**（问题2）
   - 虽然影响较小，但理论上应该修复
   - 可以后续优化

### 低优先级 🟢
3. **limit_dist 构建**（问题3）
   - 需要确认数据加载时的计算是否正确
   - 如果正确，则无需修改

4. **采样过程中的非存在边处理**（问题4）
   - 需要进一步检查
   - 如果当前实现正确，则无需修改

---

## 总结

**最需要修复的是问题1（非存在边采样精度问题）**，因为它直接影响模型学习到的边分布，可能导致生成的图不符合异质图约束。

其他问题相对次要，可以在后续迭代中优化。
