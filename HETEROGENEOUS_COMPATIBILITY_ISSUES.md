# 去噪模型中不适配异质图的问题

## 发现的问题

### 1. 采样过程 (`sample_p_zs_given_zt`, 1586行) ⚠️ **关键问题**

**位置**：
- 第1604-1606行：获取转移矩阵
- 第1728-1736行：计算 `p_s_and_t_given_0_E`

**问题**：
```python
# 第1604-1606行
Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)  # 全局转移矩阵
Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)  # 全局转移矩阵
Qt = self.transition_model.get_Qt(beta_t, self.device)  # 全局转移矩阵

# 第1728-1736行
p_s_and_t_given_0_E = diffusion_utils.compute_sparse_batched_over0_posterior_distribution(
    input_data=comp_edge_attr,
    batch=batch[comp_edge_index[0]],
    Qt=Qt.E,      # 使用全局转移矩阵
    Qsb=Qsb.E,    # 使用全局转移矩阵
    Qtb=Qtb.E,    # 使用全局转移矩阵
)
```

**影响**：
- 采样时，所有边类型使用相同的转移矩阵
- 对于异质图，这会导致不同关系族的边使用错误的转移概率
- **可能导致采样结果不符合关系族隔离的要求**

### 2. 损失计算 (`compute_val_loss` / `compute_Lt`, 1053行, 1187行) ⚠️ **关键问题**

**位置**：
- `compute_val_loss`: 第1108行, 第1112行
- `compute_Lt`: 第1197-1199行, 第1203-1214行

**问题**：
```python
# compute_val_loss 第1108行
Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)  # 全局转移矩阵
probE = E @ Qtb.E.unsqueeze(1)  # 使用全局转移矩阵

# compute_Lt 第1197-1199行
Qtb = self.transition_model.get_Qt_bar(noisy_data["alpha_t_bar"], self.device)
Qsb = self.transition_model.get_Qt_bar(noisy_data["alpha_s_bar"], self.device)
Qt = self.transition_model.get_Qt(noisy_data["beta_t"], self.device)

# 第1203-1214行：posterior_distributions 使用全局转移矩阵
prob_true = diffusion_utils.posterior_distributions(
    ...
    Qt=Qt,   # 全局转移矩阵
    Qsb=Qsb, # 全局转移矩阵
    Qtb=Qtb, # 全局转移矩阵
)
```

**影响**：
- 损失计算时，所有边类型使用相同的转移矩阵
- 对于异质图，这会导致损失计算不准确
- **可能影响模型训练的效果**

### 3. KL 先验计算 (`kl_prior`, 1098行) ⚠️ **中等问题**

**位置**：
- 第1108行, 第1112行

**问题**：
```python
Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)  # 全局转移矩阵
probE = E @ Qtb.E.unsqueeze(1)  # 使用全局转移矩阵
```

**影响**：
- KL 先验计算不准确
- 但通常这个值很小，对训练影响可能不大

### 4. 采样边 (`sample_sparse_edge`, 1549行) ⚠️ **关键问题**

**位置**：
- 第1549行：使用 `p_s_and_t_given_0_E`
- 这个值在 `sample_p_zs_given_zt` 中计算（第1728行）

**问题**：
- `p_s_and_t_given_0_E` 使用了全局转移矩阵
- 采样时没有考虑 `edge_family` 信息

**影响**：
- 采样边时，所有边类型使用相同的转移概率
- **可能导致采样结果不符合关系族隔离的要求**

## 修复优先级

1. **高优先级**：
   - 采样过程 (`sample_p_zs_given_zt`) - 直接影响采样结果
   - 损失计算 (`compute_Lt`) - 影响训练效果

2. **中优先级**：
   - KL 先验 (`kl_prior`) - 影响较小但需要修复

3. **低优先级**：
   - `compute_val_loss` - 主要用于验证，影响较小

## 修复方案

### 方案1：为每个关系族计算独立的转移概率（推荐）

在采样和损失计算时：
1. 获取所有关系族的转移矩阵
2. 根据 `edge_family` 选择对应的转移矩阵
3. 为每个关系族独立计算转移概率

### 方案2：使用全局转移矩阵作为近似（临时方案）

如果修复复杂，可以暂时使用全局转移矩阵，但需要：
1. 添加警告信息
2. 记录这个限制
3. 后续再优化

