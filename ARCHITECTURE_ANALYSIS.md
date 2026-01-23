# 网络架构分析报告

## 当前实现 vs 图片描述的架构

### 1. 嵌入时子类别和类别的融合后计算QK ❌ **未实现**

**图片要求：**
```
Q = (1 + γ_Q(S)) ⊙ W^Q_t X + β_Q(S)
```
- `X`: 类别嵌入（从 `τ^t` 通过 `E^T` 得到）
- `S`: 子类别嵌入（从 `s^t` 通过 `E_{sub}^T` 得到）
- `W^Q_t`: 不同类别对应不同的权重矩阵
- `γ_Q(S)`, `β_Q(S)`: 从子类别S生成的调制参数

**当前实现：**
```python
# transconv_layer.py:117-119
query = self.lin_query(x).view(-1, H, C)  # 直接从x计算
key = self.lin_key(x).view(-1, H, C)      # 直接从x计算
value = self.lin_value(x).view(-1, H, C)  # 直接从x计算
```
- ❌ 没有区分类别和子类别
- ❌ 没有使用不同的权重矩阵 `W^Q_t` 对不同类别
- ❌ 没有子类别调制机制

**需要的修改：**
- 需要从 `node_t` 中提取类别（`τ^t`）和子类别（`s^t`）
- 需要类别嵌入表 `E^T` 和子类别嵌入表 `E_{sub}^T`
- 需要为每个类别类型创建不同的 `W^Q_t`, `W^K_t`, `W^V_t`
- 需要实现FiLM调制：`Q = (1 + γ_Q(S)) ⊙ W^Q_t X + β_Q(S)`

---

### 2. 融入边时需要relation-aware ❌ **部分实现（但不是relation-aware）**

**图片要求：**
```
luv = (1 + γ_α(e)) ⊙ d(<Q_u, K_v>) + β_α(e)
```
其中：
- `e = e'_{i,j} E^{s_i, s_j}`: 关系嵌入，基于源节点类型 `s_i` 和目标节点类型 `s_j`
- `E^{s_i, s_j}`: 关系类型嵌入（从源类型到目标类型）
- `γ_α(e)`, `β_α(e)`: 从关系嵌入 `e` 生成的调制参数

**当前实现：**
```python
# transconv_layer.py:194-198
Y = (query_i * key_j) / math.sqrt(self.df)  # M, H, C
edge_attr_mul = self.e_mul(edge_attr)  # M, H
edge_attr_add = self.e_add(edge_attr)  # M, H
Y = Y * (edge_attr_mul.unsqueeze(-1) + 1) + edge_attr_add.unsqueeze(-1)
```
- ✅ 有FiLM机制（`edge_attr_mul` 和 `edge_attr_add`）
- ❌ 但 `edge_attr` 不是基于源节点类型和目标节点类型的关系嵌入
- ❌ 没有使用 `E^{s_i, s_j}` 来生成关系特定的嵌入
- ❌ 没有考虑关系族（relation family）的概念

**需要的修改：**
- 需要从 `edge_index` 和 `node_t` 中提取源节点类型和目标节点类型
- 需要关系类型嵌入表 `E^{s_i, s_j}`（基于 `(src_type, dst_type)` 对）
- 需要修改FiLM生成：`e = e'_{i,j} E^{s_i, s_j}`，然后 `γ_α(e)`, `β_α(e) = FiLM(e)`

---

### 3. Softmax时不同大类关系分别softmax然后再外层再套一层softmax ❌ **未实现**

**图片要求：**
1. 对每个关系大类（relation family）单独做 Softmax，避免不同关系之间的注意力竞争
2. 然后再做一个 outer softmax，避免不同关系的总贡献大小不可控

**当前实现：**
```python
# transconv_layer.py:200
alpha = softmax(Y.sum(-1), index, ptr, size_i)  # M, H
```
- ❌ 全局softmax，没有按关系族分组
- ❌ 没有两层softmax机制

**需要的修改：**
- 需要从 `edge_attr` 中提取关系族信息（通过 `edge_family_offsets`）
- 需要按关系族分组进行softmax：
  ```python
  # 伪代码
  for fam_id in relation_families:
      fam_mask = (edge_family == fam_id)
      alpha_fam = softmax(Y[fam_mask].sum(-1), index[fam_mask], ...)
  ```
- 需要outer softmax：
  ```python
  # 伪代码
  alpha_outer = softmax(alpha_fam.sum(), ...)  # 对每个节点的所有关系族聚合结果
  ```

---

## 总结

当前实现**没有**实现图片中描述的异质图架构。主要缺失：

1. ❌ **类别和子类别的融合机制**：Q/K/V的计算没有考虑类别和子类别的分离与融合
2. ❌ **Relation-aware的边嵌入**：边的FiLM没有基于源节点类型和目标节点类型的关系嵌入
3. ❌ **Per-relation-family softmax + outer softmax**：没有按关系族分组进行softmax，也没有outer softmax

## 建议

要实现图片中描述的架构，需要：

1. **修改 `TransformerConv` 类**：
   - 添加类别和子类别的嵌入表
   - 修改Q/K/V的计算方式，使用FiLM调制
   - 添加关系类型嵌入表 `E^{s_i, s_j}`
   - 修改边的FiLM，使其基于关系类型嵌入

2. **修改 `message` 方法**：
   - 实现per-relation-family softmax
   - 实现outer softmax

3. **修改模型输入**：
   - 需要将节点类型、子类别、关系族信息传递给模型层
   - 可能需要修改 `forward_sparse` 方法，传递额外的元数据
