# SparseDiff 数据流函数调用链

## 📦 阶段 1: 数据加载 (初始化)

### 1.1 数据集初始化
```
main.py::main()
  └─> ACMSubgraphsDataModule.__init__(cfg)
      ├─> ACMSubgraphsDataset.__init__(split, root, pre_transform)
      │   ├─> InMemoryDataset.__init__()  # PyG基类
      │   ├─> ACMSubgraphsDataset.process()  # 首次加载时触发
      │   │   ├─> _list_subgraph_dirs(root_dir)
      │   │   ├─> _load_or_create_vocab(subgraph_dirs)
      │   │   │   ├─> _load_meta(subdir)  # 读取 meta.json
      │   │   │   └─> _build_vocab_from_meta(meta)  # 构建词表
      │   │   ├─> _load_or_create_splits(n_graphs)  # train/val/test划分
      │   │   └─> 对每个 subgraph_XXX/:
      │   │       ├─> _load_meta(sd)  # 读取 meta.json
      │   │       ├─> torch.load(nodes.pt)  # 加载节点
      │   │       ├─> torch.load(edges.pt)  # 加载边
      │   │       ├─> _build_node_state_id()  # 构建节点扩散状态
      │   │       │   └─> _concat_node_fields()  # 拼接节点字段
      │   │       └─> _build_edges_for_graph()  # 构建有向边
      │   └─> torch.load(processed/{split}.pt)  # 加载处理后的数据
      └─> AbstractDataModule.__init__(cfg, datasets)
          ├─> dataset_stat()  # 打印统计信息
          └─> prepare_dataloader()  # 创建 DataLoader
```

### 1.2 数据集信息初始化
```
main.py::main()
  └─> ACMSubgraphsInfos.__init__(datamodule)
      ├─> AbstractDatasetInfos.complete_infos(statistics, node_types)
      │   └─> DistributionNodes(n_nodes)  # 节点数量分布
      └─> AbstractDatasetInfos.compute_input_dims(datamodule, extra_features, domain_features)
          ├─> to_one_hot(data)  # 离散 → one-hot
          ├─> utils.to_dense()  # 稀疏 → 密集
          │   ├─> to_dense_node()  # 节点
          │   └─> to_dense_edge()  # 边
          ├─> extra_features(example_data)  # 计算额外特征
          └─> domain_features(example_data)  # 计算领域特征
```

---

## 🎯 阶段 2: 训练阶段 (training_step)

### 2.1 数据预处理
```
DiscreteDenoisingDiffusion.training_step(data, i)
  ├─> dataset_info.to_one_hot(data)
  │   └─> F.one_hot(data.x, num_classes)  # 节点: 离散 → one-hot
  │   └─> F.one_hot(data.edge_attr, num_classes)  # 边: 离散 → one-hot
  │
  └─> apply_sparse_noise(data)  # 添加扩散噪声
      ├─> 随机采样时间步 t ∈ [1, T]
      ├─> noise_schedule(t_normalized)  # 计算噪声调度
      ├─> transition_model.get_Qt_bar(alpha_t_bar)  # 获取转移矩阵
      ├─> 节点加噪:
      │   └─> data.x @ Qtb.X → probN → multinomial() → node_t
      ├─> 边加噪:
      │   ├─> utils.undirected_to_directed()  # 转为有向边
      │   ├─> 对已有边加噪: edge_attr @ Qtb.E → probE → multinomial()
      │   └─> sample_non_existing_edge_attr()  # 采样非存在边的噪声
      └─> 返回 sparse_noisy_data (字典格式)
```

### 2.2 构建计算图
```
DiscreteDenoisingDiffusion.training_step()
  ├─> sample_query_edges(num_nodes, edge_proportion)
  │   └─> sampled_condensed_indices_uniformly()  # 均匀采样边对
  │
  └─> get_computational_graph(triu_query_edge_index, clean_edge_index, clean_edge_attr)
      ├─> 合并: 已有噪声边 ∪ 查询边
      ├─> coalesce()  # 去重
      └─> 返回: query_mask, comp_edge_index, comp_edge_attr
```

### 2.3 模型前向传播
```
DiscreteDenoisingDiffusion.training_step()
  └─> forward(sparse_noisy_data)
      ├─> forward_sparse(sparse_noisy_data)
      │   ├─> compute_extra_data(sparse_noisy_data)  # 计算额外特征
      │   │   ├─> extra_features(sparse_noisy_data)
      │   │   └─> domain_features(sparse_noisy_data)
      │   ├─> compute_sparse_extra_data()  # 稀疏版本
      │   └─> model(node, edge_attr, edge_index, y, batch)  # GraphTransformerConv
      │       └─> GraphTransformerConv.forward()
      │           ├─> embedding lookup (节点/边)
      │           ├─> HGT layers (异质图Transformer)
      │           ├─> relation-aware FiLM
      │           └─> 输出预测分布
      └─> 返回: SparsePlaceHolder(node, edge_attr, edge_index, ...)
```

### 2.4 损失计算
```
DiscreteDenoisingDiffusion.training_step()
  ├─> mask_query_graph_from_comp_graph()  # 获取查询边的真实标签
  │   └─> 合并查询边和真实边，生成 mask
  │
  └─> train_loss.forward(pred, true_data)
      └─> compute_Lt()  # 计算扩散损失
          ├─> compute_posterior_distribution()  # 计算后验分布
          ├─> KL散度: KL(pred || posterior)
          └─> 返回损失值
```

---

## 🎨 阶段 3: 推理/采样阶段 (sample_batch)

### 3.1 初始化
```
DiscreteDenoisingDiffusion.sample_batch()
  ├─> DistributionNodes.sample_n()  # 采样节点数量
  ├─> 初始化节点: 从边际分布采样
  │   └─> sample_discrete_node_features(probX, node_mask)
  └─> 初始化边: 从边际分布采样（仅候选边对）
    └─> sample_discrete_edge_features(probE, node_mask)
```

### 3.2 迭代去噪循环
```
DiscreteDenoisingDiffusion.sample_batch()
  └─> for s_int in reversed(time_range):  # t=T → 1
      └─> sample_p_zs_given_zt(s_float, t_float, data)
          ├─> noise_schedule.get_alpha_bar()  # 计算累积噪声
          ├─> transition_model.get_Qt_bar()  # 获取转移矩阵
          ├─> compute_sparse_batched_over0_posterior_distribution()  # 计算后验
          │   └─> compute_posterior_distribution()  # p(z_s | z_t, z_0)
          │
          └─> 分 K 次迭代处理边 (K = 1/edge_fraction):
              for i in range(len_loop):
                  ├─> sampled_condensed_indices_uniformly()  # 采样查询边
                  ├─> condensed_to_matrix_index_batch()  # 压缩索引 → 矩阵索引
                  ├─> get_computational_graph()  # 构建计算图
                  ├─> forward_sparse()  # 模型预测
                  │   └─> GraphTransformerConv.forward()
                  │
                  ├─> sample_sparse_node_edge()  # 采样节点和边
                  │   ├─> sample_sparse_node()  # 先采样节点子类别
                  │   │   └─> prob_X.multinomial(1)
                  │   └─> sample_sparse_edge()  # 再采样边子类别
                  │       └─> prob_E.multinomial(1)
                  │
                  ├─> 只保留 edge_attr != 0 的边
                  └─> 更新图状态
```

### 3.3 后处理
```
DiscreteDenoisingDiffusion.sample_batch()
  ├─> delete_repeated_twice_edges()  # 删除重复边
  ├─> to_undirected()  # 转为无向（如果需要）
  └─> 转换为离散类别
      └─> argmax()  # one-hot → 离散类别
```

---

## 🔧 关键辅助函数

### 数据格式转换
- `to_one_hot()`: 离散类别 → one-hot 编码
- `to_dense()`: 稀疏图 → 密集表示 (用于某些计算)
- `to_sparse()`: 密集表示 → 稀疏图
- `undirected_to_directed()`: 无向边 → 有向边
- `to_undirected()`: 有向边 → 无向边

### 边索引转换
- `condensed_to_matrix_index()`: 压缩索引 → (i,j) 矩阵索引
- `matrix_to_condensed_index()`: (i,j) → 压缩索引
- `condensed_to_matrix_index_batch()`: batch 版本
- `matrix_to_condensed_index_batch()`: batch 版本

### 扩散相关
- `compute_posterior_distribution()`: 计算后验分布 p(z_s | z_t, z_0)
- `sample_discrete_features()`: 从分布采样离散特征
- `sample_discrete_node_features()`: 采样节点特征
- `sample_discrete_edge_features()`: 采样边特征

### 边采样
- `sample_query_edges()`: 采样查询边对
- `sampled_condensed_indices_uniformly()`: 均匀采样压缩索引
- `get_computational_graph()`: 构建计算图（已有边 ∪ 查询边）
- `mask_query_graph_from_comp_graph()`: 生成查询边的 mask

### 特征计算
- `ExtraFeatures()`: 计算额外特征（特征值、度、距离等）
- `ExtraMolecularFeatures()`: 分子特定特征
- `compute_extra_data()`: 计算并拼接额外特征

---

## 📊 数据格式变化

```
原始数据 (nodes.pt, edges.pt)
  ↓
PyG Data (离散类别)
  - x: [N] (节点子类别 id)
  - edge_index: [2, M] (有向边)
  - edge_attr: [M] (边子类别 id)
  ↓
to_one_hot() → one-hot 编码
  - x: [N, num_node_types] (one-hot)
  - edge_attr: [M, num_edge_types] (one-hot)
  ↓
apply_sparse_noise() → 加噪
  - node_t: [N, num_node_types] (one-hot, 已加噪)
  - edge_attr_t: [M, num_edge_types] (one-hot, 已加噪)
  ↓
forward() → 模型预测
  - pred.node: [N, num_node_types] (logits)
  - pred.edge_attr: [M, num_edge_types] (logits)
  ↓
采样 → 离散类别
  - node: [N] (节点子类别 id)
  - edge_attr: [M] (边子类别 id)
```
