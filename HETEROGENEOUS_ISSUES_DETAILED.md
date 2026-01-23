# 去噪模型中不适配异质图的详细分析

## 核心问题

在异质图模式下，以下函数使用了**全局转移矩阵**，而没有考虑**关系族隔离**：

### 1. 采样过程 (`sample_p_zs_given_zt`, 1586行) ⚠️ **关键**

**问题代码**：
```python
# 第1604-1606行：获取转移矩阵
Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
Qt = self.transition_model.get_Qt(beta_t, self.device)

# 第1728-1736行：计算 p_s_and_t_given_0_E
p_s_and_t_given_0_E = diffusion_utils.compute_sparse_batched_over0_posterior_distribution(
    input_data=comp_edge_attr,
    batch=batch[comp_edge_index[0]],
    Qt=Qt.E,      # ❌ 使用全局转移矩阵
    Qsb=Qsb.E,    # ❌ 使用全局转移矩阵
    Qtb=Qtb.E,    # ❌ 使用全局转移矩阵
)
```

**问题分析**：
- `HeterogeneousMarginalUniformTransition.get_Qt_bar()` 如果不传 `edge_family_name`，会返回**全局转移矩阵**
- 对于异质图，应该为每个关系族使用不同的转移矩阵
- **当前实现会导致所有关系族的边使用相同的转移概率**

**修复方案**：
- 需要根据 `edge_family` 信息，为每个关系族计算独立的 `p_s_and_t_given_0_E`
- 或者在 `compute_sparse_batched_over0_posterior_distribution` 中支持关系族隔离

### 2. 损失计算 (`compute_Lt`, 1187行) ⚠️ **关键**

**问题代码**：
```python
# 第1197-1199行：获取转移矩阵
Qtb = self.transition_model.get_Qt_bar(noisy_data["alpha_t_bar"], self.device)
Qsb = self.transition_model.get_Qt_bar(noisy_data["alpha_s_bar"], self.device)
Qt = self.transition_model.get_Qt(noisy_data["beta_t"], self.device)

# 第1203-1214行：计算后验分布
prob_true = diffusion_utils.posterior_distributions(
    ...
    Qt=Qt,   # ❌ 使用全局转移矩阵
    Qsb=Qsb, # ❌ 使用全局转移矩阵
    Qtb=Qtb, # ❌ 使用全局转移矩阵
)
```

**问题分析**：
- `posterior_distributions` 函数使用全局转移矩阵计算所有边的后验分布
- 对于异质图，应该为每个关系族使用不同的转移矩阵
- **当前实现会导致损失计算不准确**

**修复方案**：
- 需要修改 `posterior_distributions` 或创建新的函数，支持关系族隔离
- 或者为每个关系族独立计算损失，然后求和

### 3. KL 先验 (`kl_prior`, 1098行) ⚠️ **中等**

**问题代码**：
```python
# 第1108行
Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)  # ❌ 全局转移矩阵
probE = E @ Qtb.E.unsqueeze(1)  # ❌ 使用全局转移矩阵
```

**问题分析**：
- KL 先验通常值很小，影响可能不大
- 但为了准确性，应该使用关系族隔离的转移矩阵

### 4. 验证损失 (`compute_val_loss`, 1053行) ⚠️ **中等**

**问题代码**：
```python
# 第1108行
Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)  # ❌ 全局转移矩阵
probE = E @ Qtb.E.unsqueeze(1)  # ❌ 使用全局转移矩阵
```

**问题分析**：
- 主要用于验证，影响相对较小
- 但为了准确性，应该修复

## 当前状态

✅ **已适配**：
- `apply_sparse_noise` - 噪声应用已支持关系族隔离

❌ **未适配**：
- 采样过程 - 使用全局转移矩阵
- 损失计算 - 使用全局转移矩阵
- KL 先验 - 使用全局转移矩阵

## 影响评估

1. **训练阶段**：
   - 损失计算不准确 → 可能影响训练效果
   - 但噪声应用正确 → 训练应该仍能进行

2. **采样阶段**：
   - 采样时使用错误的转移概率 → **可能导致采样结果不符合关系族隔离的要求**
   - 这是最严重的问题

## 修复建议

### 优先级1：采样过程（最重要）

需要修改 `sample_p_zs_given_zt`，使其：
1. 获取所有关系族的转移矩阵
2. 根据 `edge_family` 选择对应的转移矩阵
3. 为每个关系族独立计算 `p_s_and_t_given_0_E`

### 优先级2：损失计算

需要修改 `compute_Lt`，使其：
1. 为每个关系族独立计算后验分布
2. 分别计算 KL 散度
3. 求和得到总损失

### 优先级3：KL 先验

可以暂时使用全局转移矩阵作为近似，影响较小。

