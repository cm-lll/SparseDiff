# 状态空间映射说明

## 问题背景

在异质图中，每个关系族有自己的**局部状态空间**，但模型使用**全局状态空间**。

### 局部状态空间 vs 全局状态空间

**例子**：
- 关系族A（paper-cites-paper）：有3个子类型（cite, ref, similar）
- 关系族B（author-writes-paper）：有2个子类型（first_author, co_author）

**局部状态空间**（每个关系族独立）：
- 关系族A：状态 0（no-edge），状态 1（cite），状态 2（ref），状态 3（similar）
- 关系族B：状态 0（no-edge），状态 1（first_author），状态 2（co_author）

**全局状态空间**（所有关系族统一）：
- 状态 0：no-edge（所有关系族共享）
- 状态 1-3：关系族A的子类型（offset=1）
- 状态 4-5：关系族B的子类型（offset=4）

## 为什么需要映射？

### 1. 转移矩阵使用局部状态空间

每个关系族的转移矩阵是基于其**局部状态空间**构建的：
- 关系族A的转移矩阵：4x4（0, 1, 2, 3）
- 关系族B的转移矩阵：3x3（0, 1, 2）

### 2. 模型输出使用全局状态空间

模型的输出和后续处理都使用**全局状态空间**：
- 模型预测：`pred.edge_attr` 的形状是 `(num_edges, num_global_states)`
- 采样结果：需要使用全局ID

## 映射过程

### 步骤1：计算局部转移概率

```python
# 为关系族A计算转移概率（使用局部状态空间）
p_s_and_t_given_0_E_fam_A = compute_posterior(
    input_data=fam_A_local_attr_onehot,  # 局部ID: 0, 1, 2, 3
    Qt=Qt_fam_A,  # 4x4 转移矩阵
    ...
)  # 输出: (num_edges_A, 4, 4) - 局部状态空间
```

### 步骤2：映射到全局状态空间

```python
# 将局部状态映射到全局状态
# 局部状态 0 -> 全局状态 0 (no-edge)
# 局部状态 1 -> 全局状态 1 (offset=1)
# 局部状态 2 -> 全局状态 2 (offset=1)
# 局部状态 3 -> 全局状态 3 (offset=1)

p_s_and_t_given_0_E_global = torch.zeros(
    (num_edges_A, num_global_states, num_global_states)
)

for local_from in range(4):
    global_from = 0 if local_from == 0 else offset_A + local_from - 1
    for local_to in range(4):
        global_to = 0 if local_to == 0 else offset_A + local_to - 1
        p_s_and_t_given_0_E_global[:, global_from, global_to] = \
            p_s_and_t_given_0_E_fam_A[:, local_from, local_to]
```

### 步骤3：合并所有关系族

```python
# 为所有关系族创建全局转移概率矩阵
p_s_and_t_given_0_E = torch.zeros(
    (num_all_edges, num_global_states, num_global_states)
)

# 填充关系族A的结果
p_s_and_t_given_0_E[fam_A_mask] = p_s_and_t_given_0_E_global_A

# 填充关系族B的结果
p_s_and_t_given_0_E[fam_B_mask] = p_s_and_t_given_0_E_global_B
```

## 代码实现

在 `sample_p_zs_given_zt` 中的实现：

```python
# 1. 为每个关系族计算局部转移概率
for fam_name in edge_families:
    # 获取该关系族的边（使用局部ID）
    fam_local_attr_onehot = ...  # 局部状态空间
    
    # 使用关系族特定的转移矩阵
    Qt_fam = all_family_qt[fam_name].E  # 局部状态空间的转移矩阵
    
    # 计算局部转移概率
    p_s_and_t_given_0_E_fam = compute_posterior(
        input_data=fam_local_attr_onehot,
        Qt=Qt_fam,  # 局部状态空间
        ...
    )  # (num_edges_fam, num_fam_states, num_fam_states)
    
    # 2. 映射到全局状态空间
    for local_from in range(num_fam_states):
        global_from = 0 if local_from == 0 else offset + local_from - 1
        for local_to in range(num_fam_states):
            global_to = 0 if local_to == 0 else offset + local_to - 1
            p_s_and_t_given_0_E[fam_mask, global_from, global_to] = \
                p_s_and_t_given_0_E_fam[:, local_from, local_to]
```

## 总结

**"直接为每个关系族计算转移概率，并正确映射到全局状态空间"** 的意思是：

1. **为每个关系族独立计算**：使用该关系族的局部状态空间和转移矩阵
2. **映射到全局状态空间**：将局部状态ID转换为全局状态ID
3. **合并结果**：所有关系族的结果都映射到同一个全局状态空间，便于后续处理

这样既保证了每个关系族使用正确的转移矩阵（关系族隔离），又保证了结果与模型输出兼容（全局状态空间）。

