# 异质图噪声模型实现进展

## 已完成

### 1. 创建 `HeterogeneousMarginalUniformTransition` 类 ✅
- 位置: `sparse_diffusion/diffusion/heterogeneous_transition.py`
- 功能: 支持关系族隔离的边转移矩阵
- 关键方法:
  - `get_Qt()`: 单步转移矩阵（可指定关系族）
  - `get_Qt_bar()`: t 步累积转移矩阵（可指定关系族）
  - `get_all_family_Qt_bar()`: 为所有关系族返回转移矩阵（用于批处理）

### 2. 修改 `ACMSubgraphsInfos` ✅
- 计算每个关系族的边类型分布（`edge_family_marginals`）
- 存储关系族映射信息（`edge_family2id`, `edge_family_offsets`）

## 待完成

### 3. 修改 `apply_sparse_noise` 方法 ⏳
需要修改 `sparse_diffusion/diffusion_model_sparse.py` 中的 `apply_sparse_noise` 方法：

**关键修改点**：

1. **边的扩散（Step 1）**：
   - 当前：对所有已存在的边使用统一的 `Qtb.E`
   - 需要：按 `edge_family` 分组，为每个关系族使用对应的转移矩阵

2. **非存在边的采样（Step 2）**：
   - 当前：`emerge_prob = Qtb.E[:, 0, 1:].sum(-1)` （全局）
   - 需要：为每个关系族独立计算 `emerge_prob`
   - 当前：`num_emerge_edges = Binomial(num_neg_edge, emerge_prob)` （全局）
   - 需要：为每个关系族独立计算 `num_neg_edge` 和 `num_emerge_edges`
   - **注意**：非存在边的计算需要考虑端点类型（从 `meta.json` 的 `fam_endpoints` 获取）

3. **边属性采样**：
   - 当前：使用全局的 `Qtb.E[:, 0, 1:]` 采样新边的属性
   - 需要：根据关系族使用对应的转移矩阵采样

## 实现思路

### 方案 1: 按关系族循环处理（简单但可能较慢）
```python
# 对每个关系族分别处理
for fam_name, fam_id in edge_family2id.items():
    # 1. 获取该关系族的已存在边
    fam_mask = (edge_family == fam_id)
    fam_existing_edges = ...
    
    # 2. 计算该关系族的非存在边数（需要考虑端点类型）
    fam_num_neg_edge = ...
    
    # 3. 使用关系族特定的转移矩阵
    Qtb_fam = transition_model.get_Qt_bar(alpha_t_bar, device, edge_family_name=fam_name)
    
    # 4. 扩散已存在的边
    # 5. 采样非存在的边（伯努利分布）
    # 6. 合并结果
```

### 方案 2: 批处理优化（更高效）
- 使用 `get_all_family_Qt_bar()` 一次性获取所有关系族的转移矩阵
- 使用向量化操作处理所有关系族

## 关键挑战

1. **非存在边数的计算**：
   - 需要根据关系族的端点类型（`src_type`, `dst_type`）计算
   - 例如：`author_of` 关系族的非存在边 = (Author 节点数) × (Paper 节点数) - 已存在的 `author_of` 边数

2. **端点类型信息**：
   - 需要从 `meta.json` 的 `fam_endpoints` 获取
   - 或者在 `ACMSubgraphsInfos` 中存储

3. **批处理兼容性**：
   - 需要处理批次中不同图的关系族分布
   - 需要处理不同关系族的边数差异

## 下一步

1. 在 `ACMSubgraphsInfos` 中存储 `fam_endpoints` 信息
2. 实现关系族隔离的边扩散逻辑
3. 实现关系族隔离的非存在边采样（伯努利分布）
4. 测试验证

