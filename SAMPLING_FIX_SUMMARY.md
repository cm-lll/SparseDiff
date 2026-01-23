# 采样逻辑修复总结

## 修复内容

### ✅ 第一种情况（噪声模型 - `apply_sparse_noise`）：已符合要求
- **位置**：约第946-1087行
- **实现**：
  - 按关系族进行循环处理
  - 使用伯努利分布（二项分布）采样：`k_fam ~ B(m̄_fam, qt_fam)`
  - 考虑端点类型约束（`src_mask`, `dst_mask`）
  - 排除所有已存在的边（`existing_edge_index=dir_edge_index`）
- **状态**：✅ **无需修改**

---

### ✅ 第二种情况（去噪模型 - `sample_p_zs_given_zt`）：已修复
- **位置**：约第1937-2134行
- **修复前**：
  - 对所有边进行全局均匀采样
  - 没有按关系族分别采样

- **修复后**：
  1. **按关系族生成查询边**（约第1975-2039行）：
     - 为每个关系族分别计算可能的边数（基于端点类型）
     - 为每个关系族分别生成所有可能的边对（src_type -> dst_type）
     - 对每个关系族的边进行随机打乱（实现均匀采样）
     - 合并所有关系族的查询边

  2. **在循环中分批处理**（约第2112-2133行）：
     - 第一次循环时，打乱所有查询边（实现均匀采样）
     - 每个循环选择一部分边（按 `edge_fraction` 比例）
     - 这些查询边与已有边合并：`Em = Et ∪ Eq`

- **关键代码**：
  ```python
  # 为每个关系族分别生成查询边
  for fam_id, fam_name in id2edge_family.items():
      # 计算端点类型
      src_type = fam_endpoints[fam_name]["src_type"]
      dst_type = fam_endpoints[fam_name]["dst_type"]
      
      # 为每个批次生成该关系族的所有可能边
      for b in range(bs):
          # 生成所有可能的边对（src_type -> dst_type）
          # 随机打乱实现均匀采样
          perm = torch.randperm(num_fam_possible_edges, device=self.device)
          fam_query_edge_index = all_fam_edges[:, perm]
  
  # 合并所有关系族的查询边
  all_query_edge_index = torch.cat(all_query_edge_index_list, dim=1)
  
  # 在循环中分批处理
  for i in range(len_loop):
      if i == 0:
          # 打乱所有查询边
          perm = torch.randperm(num_query_edges_total, device=self.device)
          all_query_edge_index = all_query_edge_index[:, perm]
      
      # 选择当前循环的边
      triu_query_edge_index = all_query_edge_index[:, start_idx:end_idx]
      
      # 与已有边合并：Em = Et ∪ Eq
      comp_edge_index, comp_edge_attr = get_computational_graph(
          triu_query_edge_index=triu_query_edge_index,
          clean_edge_index=sparse_noisy_data["edge_index_t"],  # Et
          ...
      )
  ```

- **状态**：✅ **已修复**

---

## 验证要点

### 第一种情况（噪声模型）
- ✅ 按关系族进行伯努利采样
- ✅ 排除所有已存在的边
- ✅ 考虑端点类型约束

### 第二种情况（去噪模型）
- ✅ 按关系族分别进行均匀采样
- ✅ 合并所有关系族的查询边
- ✅ 查询边与已有边合并：`Em = Et ∪ Eq`
- ✅ 可能会采样到已有的噪声边（`get_computational_graph` 会去重）

---

## 代码变更位置

1. **`sample_p_zs_given_zt` 方法**（约第1937-2134行）：
   - 添加了异质图模式的检查
   - 为每个关系族分别生成查询边
   - 修改了循环逻辑，支持按关系族采样的查询边

2. **变量初始化**（约第1937行）：
   - 添加了 `all_query_edge_index` 和 `all_query_edge_batch` 的初始化

---

## 测试建议

1. **验证第一种情况**：
   - 检查噪声模型是否正确按关系族进行伯努利采样
   - 验证非存在边的采样是否符合端点类型约束

2. **验证第二种情况**：
   - 检查去噪模型是否正确按关系族进行均匀采样
   - 验证查询边是否包含所有关系族的边
   - 验证查询边与已有边的合并是否正确
