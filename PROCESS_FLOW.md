# 数据 Process 流程详解

## 整体流程（`ACMSubgraphsDataset.process()`）

### 阶段 1: 发现子图目录
```python
subgraph_dirs = _list_subgraph_dirs(self.root)
# 例如: ['subgraph_000', 'subgraph_001', ...]
```

### 阶段 2: 构建/加载词汇表（`_load_or_create_vocab`）

#### 2.1 检查缓存
- 如果 `processed/vocab.json` 存在，直接加载
- **重要**：检查缓存的 `heterogeneous` 设置是否与当前设置一致
  - 不一致则删除缓存，重新生成

#### 2.2 构建词汇表（`_build_vocab_from_meta`）

**异质图模式（heterogeneous=True）**：
1. **节点类型映射**：
   - `node_type_names`: ['Author', 'Organization', 'Paper']
   - `node_type2id`: {'Author': 0, 'Organization': 1, 'Paper': 2}

2. **边关系族映射**：
   - `edge_family2id`: {'affiliated_with': 0, 'author_of': 1, 'cites': 2}
   - `edge_family_offsets`: 为每个关系族计算 offset
     - `affiliated_with`: offset=1, 范围 [1, 1]（1 个子类别）
     - `author_of`: offset=2, 范围 [2, 4]（3 个子类别）
     - `cites`: offset=5, 范围 [5, 5]（1 个子类别）

3. **边标签映射（类别隔离）**：
   - `edge_label2id`: 使用 `offset + local_id` 计算全局 ID
     - `affiliated_with:__none__` -> 1
     - `author_of:first_author` -> 2
     - `author_of:second_author` -> 3
     - `author_of:co_author` -> 4
     - `cites:__none__` -> 5

**同质图模式（heterogeneous=False）**：
- 所有边标签共享全局 ID 空间，从 1 开始顺序分配

#### 2.3 保存词汇表
- 保存到 `processed/vocab.json`，包含所有映射关系和 `heterogeneous` 标志

### 阶段 3: 计算扩散空间大小

```python
# 节点扩散空间：所有节点子类别的总数
type_sizes = [len(meta0["schema_by_type"][t]) for t in node_type_names]
num_node_types = sum(type_sizes)  # 例如: 4 + 4 + 6 = 14

# 边扩散空间：所有边标签数 + 1（无边状态）
num_edge_types = 1 + len(edge_label2id)  # 例如: 1 + 5 = 6
```

### 阶段 4: 数据集划分（`_load_or_create_splits`）

- 如果 `processed/splits.pt` 存在，直接加载
- 否则随机划分：train (80%), val (10%), test (10%)
- 保存划分结果到 `processed/splits.pt`

### 阶段 5: 处理每个子图

对每个选中的子图目录（根据 split）：

#### 5.1 加载原始数据
```python
meta = _load_meta(sd)  # 加载 meta.json
nodes = torch.load(osp.join(sd, "nodes.pt"))  # 加载节点数据
edges = torch.load(osp.join(sd, "edges.pt"))  # 加载边数据
```

#### 5.2 构建节点扩散状态（`_build_node_state_id`）

1. **计算节点类型 offset**：
   - Author: offset=0, 范围 [0, 3]（4 个子类别）
   - Organization: offset=4, 范围 [4, 7]（4 个子类别）
   - Paper: offset=8, 范围 [8, 13]（6 个子类别）

2. **构建 `node_state`**（`data.x`）：
   - 全局节点子类别 ID，使用 `type_offsets[node_type] + local_subtype_id`
   - 例如：Author 节点的 HighImpact 子类别 -> 全局 ID 0

3. **构建 `node_type_id` 和 `node_subtype_local`**：
   - `node_type_id`: 粗粒度节点类型 ID（0=Author, 1=Organization, 2=Paper）
   - `node_subtype_local`: 节点类型内的局部子类别 ID（用于约束）

#### 5.3 构建边（`_build_edges_for_graph`）

**异质图模式**：
1. 对每个关系族（family）：
   - 获取端点类型，计算全局节点 ID（使用 `offsets[src_type] + src_local`）
   - 使用 `edge_family_offsets[fam] + local_id` 计算全局边标签 ID
   - 记录 `edge_family` ID（用于约束）

2. 合并所有关系族的边：
   - `edge_index`: [2, M] 有向边索引
   - `edge_attr`: [M] 全局边标签 ID（类别隔离）
   - `edge_family`: [M] 关系族 ID

**同质图模式**：
- 使用 `edge_label2id[label_str]` 直接映射（全局空间）

#### 5.4 创建 PyG Data 对象

```python
data = Data(
    x=node_state,              # [N] 节点扩散状态（全局子类别 ID）
    edge_index=edge_index,     # [2, M] 有向边索引
    edge_attr=edge_attr,       # [M] 边扩散状态（全局边标签 ID，类别隔离）
    node_type=node_type_id,   # [N] 粗粒度节点类型 ID（用于约束）
    node_subtype=node_subtype_local,  # [N] 局部子类别 ID（用于约束）
    edge_family=edge_family,   # [M] 关系族 ID（用于约束，仅异质图）
    y=torch.zeros((1, 0)),    # 图级别标签（空）
)
```

### 阶段 6: 统计和保存

1. **计算统计信息**：
   - `num_nodes`: 节点数量分布
   - `node_types`: 节点类型分布
   - `bond_types`: 边类型分布

2. **保存处理后的数据**：
   - `processed/{split}.pt`: 所有图的 Data 对象（collated）
   - `processed/{split}_num_nodes.pkl`: 节点数量统计
   - `processed/{split}_node_types.npy`: 节点类型分布
   - `processed/{split}_bond_types.npy`: 边类型分布

## 关键设计点

### 1. 节点类别隔离
- 每个节点类型有独立的子类别 ID 范围
- 通过 `type_offsets` 实现类别隔离
- 扩散时，Author 节点只能扩散到 [0, 3]，不会扩散到其他类别

### 2. 边关系族隔离（异质图模式）
- 每个关系族有独立的子类别 ID 范围
- 通过 `edge_family_offsets` 实现类别隔离
- 扩散时，author_of 边只能扩散到 [2, 4]，不会扩散到其他关系族

### 3. 缓存机制
- 词汇表和数据集划分会被缓存
- 如果 `heterogeneous` 设置改变，会自动重新生成词汇表

### 4. 有向边保留
- 保留原始边的方向（src -> dst）
- 不自动添加反向边

