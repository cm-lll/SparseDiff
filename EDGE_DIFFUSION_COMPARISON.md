# 边扩散方式对比：全局空间 vs 类别隔离空间

## 当前实现（方式1）：全局空间 + Mask约束

### 数据结构
```python
# 全局边标签 id（所有关系族共享一个空间）
edge_label2id = {
    "affiliated_with:__none__": 1,      # 全局id=1
    "author_of:first_author": 2,         # 全局id=2
    "author_of:second_author": 3,        # 全局id=3
    "author_of:co_author": 4,            # 全局id=4
    "cites:__none__": 5                  # 全局id=5
}
# 无边状态: 0
```

### 扩散过程
- 所有边在同一个全局空间（0-5）内扩散
- 需要在扩散时添加 mask，限制每条边只能扩散到其所属关系族的子类别
- 例如：`author_of` 边只能扩散到 [2, 3, 4]，不能扩散到 [1, 5]

### 优点
- ✅ 实现简单，不需要修改数据结构
- ✅ 模型输出维度固定（所有边共享相同的输出维度）
- ✅ 边际分布计算简单（所有边类型在一个分布中）

### 缺点
- ❌ 需要在扩散时添加 mask 约束（增加计算开销）
- ❌ 模型可能学习到跨关系族的无效转移（需要 mask 来纠正）
- ❌ 不符合"在各自类别空间内扩散"的语义

---

## 提议实现（方式2）：类别隔离空间（类似节点）

### 数据结构
```python
# 每个关系族有独立的 id 空间（通过 offset 隔离）
edge_family2id = {
    "affiliated_with": 0,
    "author_of": 1,
    "cites": 2
}

# 每个关系族的子类别数量
edge_family_sizes = {
    "affiliated_with": 1,  # 只有 __none__（1个子类别）
    "author_of": 3,         # first_author, second_author, co_author（3个子类别）
    "cites": 1              # 只有 __none__（1个子类别）
}

# 计算 offset（类似节点的 type_offsets）
edge_family_offsets = {
    "affiliated_with": 1,  # 全局id范围: [1, 1]（无边=0，所以从1开始）
    "author_of": 2,         # 全局id范围: [2, 4]
    "cites": 5              # 全局id范围: [5, 5]
}

# 边标签到全局id的映射（保持向后兼容）
edge_label2id = {
    "affiliated_with:__none__": 1,      # offset=1 + local_id=0 = 1
    "author_of:first_author": 2,         # offset=2 + local_id=0 = 2
    "author_of:second_author": 3,        # offset=2 + local_id=1 = 3
    "author_of:co_author": 4,            # offset=2 + local_id=2 = 4
    "cites:__none__": 5                  # offset=5 + local_id=0 = 5
}
# 无边状态: 0（保持不变）
```

### 扩散过程
- 每条边在自己的关系族空间内扩散（通过 offset 机制自动隔离）
- 不需要 mask，因为 id 空间已经隔离
- 例如：`author_of` 边的 edge_attr 只能是 [2, 3, 4]，物理上无法扩散到其他关系族

### 优点
- ✅ 语义清晰：完全符合"在各自类别空间内扩散"的要求
- ✅ 不需要 mask 约束（减少计算开销）
- ✅ 与节点处理方式一致（代码风格统一）
- ✅ 模型不会学习到跨关系族的无效转移

### 缺点
- ⚠️ 需要修改数据加载代码（但改动不大）
- ⚠️ 需要确保扩散模型支持这种隔离机制（可能需要检查）

---

## 实现对比

### 方式1（当前）：全局空间 + Mask
```python
# 扩散时
edge_attr = [2, 3, 4]  # author_of 边的当前状态
# 需要 mask：只允许扩散到 [2, 3, 4]，mask 掉 [0, 1, 5]
mask = create_edge_family_mask(edge_family="author_of", all_edge_labels)
probE_masked = probE * mask  # 应用 mask
```

### 方式2（提议）：类别隔离空间
```python
# 扩散时
edge_attr = [2, 3, 4]  # author_of 边的当前状态（已经是隔离空间）
# 不需要 mask：因为 [2, 3, 4] 已经是 author_of 的完整空间
# 模型输出维度也是针对 author_of 的（3个类别）
probE  # 直接使用，无需 mask
```

---

## 建议

**推荐使用方式2（类别隔离空间）**，原因：
1. **语义一致性**：与节点处理方式完全一致
2. **实现简洁**：不需要 mask，代码更清晰
3. **性能更好**：减少 mask 计算开销
4. **更符合设计**：完全实现"在各自类别空间内扩散"

### 需要修改的地方
1. `_build_vocab_from_meta()`: 计算每个关系族的 offset
2. `_build_edges_for_graph()`: 使用 offset 机制构建边标签 id
3. 添加 `edge_family` 信息到 Data 对象（用于约束，类似 `node_type`）

### 检查点
- 确认扩散模型是否支持不同边有不同的输出维度（如果不支持，可能需要统一维度但通过 offset 隔离）
