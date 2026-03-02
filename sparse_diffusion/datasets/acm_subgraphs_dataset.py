import json
import os
import os.path as osp
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import get_original_cwd
from torch_geometric.data import Data, InMemoryDataset

from sparse_diffusion.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
from sparse_diffusion.datasets.dataset_utils import RemoveYTransform, Statistics, load_pickle, save_pickle
from sparse_diffusion.metrics.metrics_utils import atom_type_counts, edge_counts, node_counts


def _touch(path: str) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8"):
        pass


def _list_subgraph_dirs(root_dir: str) -> List[str]:
    # root_dir is expected to contain subgraph_*/ folders
    subdirs = []
    for name in os.listdir(root_dir):
        p = osp.join(root_dir, name)
        if osp.isdir(p) and name.startswith("subgraph_"):
            subdirs.append(p)
    return sorted(subdirs)


def _load_meta(subdir: str) -> Dict:
    with open(osp.join(subdir, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _build_vocab_from_meta(meta: Dict, heterogeneous: bool = True) -> Tuple[List[str], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Returns:
      - node_type_names: list of node type names in stable order
      - node_type2id: mapping node_type -> 0..T-1
      - edge_label2id: mapping edge label string -> 1..R (0 reserved for 'no-edge')

    Notes:
      - Your diffusion target is the *subtype*, not the coarse node type. We therefore build the node
        diffusion state from node subtypes (see _build_node_state_id()).
      - For directed relations, we preserve the original direction (u->v) without adding reverse edges.
        Edge labels are mapped directly from meta.json's fam_id2label (e.g., "author_of:first_author").
        If a label is "__none__", it means no specific subtype (only presence/absence is diffused).
    """
    node_type_names = list(meta["node_types"])
    node_type2id = {t: i for i, t in enumerate(node_type_names)}

    fam_id2label = meta.get("fam_id2label", {})
    edge_family_names = list(fam_id2label.keys()) if heterogeneous else []
    edge_family2id = {fam: i for i, fam in enumerate(edge_family_names)} if heterogeneous else {}
    
    if heterogeneous:
        # 方式2：类别隔离空间（类似节点）
        # 计算每个关系族的子类别数量和 offset
        edge_family_offsets: Dict[str, int] = {}
        cur_offset = 1  # 从1开始，0保留给"无边"
        
        for fam in edge_family_names:
            id2label = fam_id2label[fam]
            # 计算该关系族的子类别数量（通过 id2label 的键数量）
            num_subtypes = len(id2label)
            edge_family_offsets[fam] = cur_offset
            cur_offset += num_subtypes
        
        # 构建 edge_label2id：使用 offset + local_id
        edge_label2id: Dict[str, int] = {}
        for fam in edge_family_names:
            id2label = fam_id2label[fam]
            offset = edge_family_offsets[fam]
            for k in sorted(id2label.keys(), key=lambda x: int(x)):
                lbl = id2label[k]
                # 跳过None或空字符串标签
                if lbl is None or (isinstance(lbl, str) and len(lbl.strip()) == 0):
                    continue
                local_id = int(k) - 1  # meta.json 中 id 从 1 开始，转换为从 0 开始
                global_id = offset + local_id
                edge_label2id[lbl] = global_id
        
        # 调试信息：记录edge_label2id的内容
        print(f"[DEBUG] _build_vocab_from_meta: edge_label2id数量={len(edge_label2id)}, 内容={list(edge_label2id.keys())}")
        print(f"[DEBUG] _build_vocab_from_meta: edge_family_offsets={edge_family_offsets}")
        
        return node_type_names, node_type2id, edge_label2id, edge_family2id, edge_family_offsets
    else:
        # 方式1：全局空间（原始实现，用于同质图）
        edge_labels: List[str] = []
        seen_labels = set()
        for fam, id2label in fam_id2label.items():
            for k in sorted(id2label.keys(), key=lambda x: int(x)):
                lbl = id2label[k]
                if lbl not in seen_labels:
                    edge_labels.append(lbl)
                    seen_labels.add(lbl)
        
        # global ids start at 1; 0 is reserved for non-edges in this codebase
        edge_label2id = {lbl: i + 1 for i, lbl in enumerate(edge_labels)}
        return node_type_names, node_type2id, edge_label2id, {}, {}


def _concat_node_fields(
    nodes_dict: Dict, node_type_names: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    nodes.pt structure:
      nodes_dict[node_type]["subtype"] : LongTensor[num_nodes_of_type]
    Returns:
      - node_type_id: LongTensor[N]  (coarse type id)
      - node_subtype_local: LongTensor[N]  (subtype id within that type)
      - offsets: mapping node_type -> global offset
    """
    offsets: Dict[str, int] = {}
    total = 0
    for t in node_type_names:
        offsets[t] = total
        total += int(nodes_dict[t]["subtype"].shape[0])

    node_type_id = torch.empty(total, dtype=torch.long)
    node_subtype_local = torch.empty(total, dtype=torch.long)
    for t in node_type_names:
        off = offsets[t]
        n = int(nodes_dict[t]["subtype"].shape[0])
        node_type_id[off : off + n] = 0  # will be overwritten by caller
        node_subtype_local[off : off + n] = nodes_dict[t]["subtype"].long()

    return node_type_id, node_subtype_local, offsets


def _build_node_state_id(
    node_type_names: List[str], nodes_dict: Dict, node_type2id: Dict[str, int]
) -> Tuple[torch.Tensor, Dict[str, int], List[int]]:
    """
    Builds the diffusion target for nodes: a single categorical variable over ALL node subtypes
    (paper subtype, author subtype, org subtype, ...), using disjoint ranges per node type.

    Important:
      - 每个节点在自己的类别空间内扩散子类别（通过 offset 机制实现类别隔离）
      - Author 节点：全局 id 范围 [0, 3]（4 个子类别）
      - Organization 节点：全局 id 范围 [4, 7]（4 个子类别）
      - Paper 节点：全局 id 范围 [8, 13]（6 个子类别）
      - 扩散时，Author 节点只能扩散到 [0, 3]，不会扩散到其他类别的子类别

    Returns:
      - node_state: LongTensor[N] in 0..(sum(A_t)-1)，全局子类别 id
      - type_offsets: mapping node_type -> starting offset in the global subtype space
      - type_sizes: list of A_t in node_type_names order（每个类别的子类别数量）
    """
    type_sizes = [int(nodes_dict[t]["A"]) for t in node_type_names]
    type_offsets: Dict[str, int] = {}
    cur = 0
    for t, a in zip(node_type_names, type_sizes):
        type_offsets[t] = cur
        cur += a

    # global node ordering is by concatenating node types in node_type_names order
    node_state = torch.empty(sum(int(nodes_dict[t]["subtype"].shape[0]) for t in node_type_names), dtype=torch.long)
    node_type_id, node_subtype_local, offsets = _concat_node_fields(nodes_dict, node_type_names)
    for t in node_type_names:
        off = offsets[t]
        n = int(nodes_dict[t]["subtype"].shape[0])
        node_type_id[off : off + n] = int(node_type2id[t])
        node_state[off : off + n] = int(type_offsets[t]) + node_subtype_local[off : off + n]

    return node_state, type_offsets, type_sizes


def _build_edges_for_graph(
    edges_dict: Dict,
    meta: Dict,
    offsets: Dict[str, int],
    edge_label2id: Dict[str, int],
    edge_family_offsets: Dict[str, int] = None,
    edge_family2id: Dict[str, int] = None,
    heterogeneous: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    edges.pt structure:
      edges_dict["families"][family]["src_local"], ["dst_local"], ["y"]
    meta.json:
      fam_endpoints[family] -> src_type, dst_type
      fam_id2label[family][str(local_id)] -> label_str
    Returns:
      - edge_index: LongTensor[2, M] (directed edges: src -> dst, NO reverse edges added)
      - edge_attr: LongTensor[M] with values in 1..R (0 reserved for non-edges)
      - edge_family: LongTensor[M] with edge family ids (only for heterogeneous, else None)

    Important:
      - **边扩散**：
        * 异质图（heterogeneous=True）：每条边在自己的关系族（family）空间内扩散子类别，通过 offset 机制隔离。
          例如：author_of 边只能在 author_of 的 3 个子类别内扩散，不会扩散到 cites 的子类别。
        * 同质图（heterogeneous=False）：所有边在全局空间内扩散（原始实现）。
      
      - **无子类别情况**：如果 label 是 "__none__"（如 "affiliated_with:__none__"），
        表示该关系族没有更细的子类别，子类别就是类别本身（只有一种状态）。
      
      - **无边状态**：edge_attr=0 表示无边，这是额外的状态，不属于任何关系族。
      
      - **有向边**：保留原始方向 (u->v)，不自动添加反向边。
    """
    fam_endpoints = meta["fam_endpoints"]
    fam_id2label = meta["fam_id2label"]

    edge_src_all = []
    edge_dst_all = []
    edge_attr_all = []
    edge_family_all = [] if heterogeneous else None

    for fam, fd in edges_dict["families"].items():
        ep = fam_endpoints[fam]
        src_type = ep["src_type"]
        dst_type = ep["dst_type"]

        src = fd["src_local"].long() + int(offsets[src_type])
        dst = fd["dst_local"].long() + int(offsets[dst_type])
        y_local = fd["y"].long()

        # Map local edge label ids -> global edge label ids
        id2label = fam_id2label[fam]
        uniq = torch.unique(y_local).tolist()
        local_to_global: Dict[int, int] = {}
        
        if heterogeneous and edge_family_offsets is not None:
            # 方式2：使用 offset 机制（类别隔离）
            offset = edge_family_offsets[fam]
            for u in uniq:
                local_id = int(u) - 1  # meta.json 中 id 从 1 开始，转换为从 0 开始
                global_id = offset + local_id
                local_to_global[int(u)] = global_id
        else:
            # 方式1：全局映射（同质图）
            for u in uniq:
                label_str = id2label[str(int(u))]
                if label_str in edge_label2id:
                    local_to_global[int(u)] = int(edge_label2id[label_str])
                else:
                    local_to_global[int(u)] = 0  # Fallback to no-edge

        # Preserve original directed edges (src -> dst)
        y_list = y_local.tolist()
        edge_attr_directed = torch.tensor(
            [local_to_global[int(y_list[k])] for k in range(len(y_list))],
            dtype=torch.long,
        )

        edge_src_all.append(src)
        edge_dst_all.append(dst)
        edge_attr_all.append(edge_attr_directed)
        
        if heterogeneous and edge_family2id is not None:
            # 记录每条边所属的关系族 id（用于约束）
            fam_id = edge_family2id.get(fam, 0)  # 获取关系族 id，如果不存在则默认为 0
            edge_family_all.append(torch.full((len(edge_attr_directed),), fam_id, dtype=torch.long))

    if len(edge_src_all) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0,), dtype=torch.long)
        edge_family = torch.zeros((0,), dtype=torch.long) if heterogeneous else None
        return edge_index, edge_attr, edge_family

    edge_src = torch.cat(edge_src_all, dim=0)
    edge_dst = torch.cat(edge_dst_all, dim=0)
    edge_attr = torch.cat(edge_attr_all, dim=0)
    edge_family = torch.cat(edge_family_all, dim=0) if heterogeneous else None

    edge_index = torch.stack([edge_src, edge_dst], dim=0)
    return edge_index, edge_attr, edge_family


class ACMSubgraphsDataset(InMemoryDataset):
    """
    InMemoryDataset是 PyTorch Geometric 里的一个数据集基类
    InMemoryDataset 负责「raw → process → 存盘 → 一次性加载进内存 + 按索引返回单个图」这一套流程；
    ACMSubgraphsDataset 只负责实现 raw_file_names、processed_file_names、download、process(),
    以及用 self.data, self.slices 把处理结果存/读到基类约定的那个 .pt 里。

    Loads a dataset of heterogeneous citation ego subgraphs stored as:
      subgraph_XXX/
        - nodes.pt
        - edges.pt
        - meta.json

    Each subgraph folder is treated as one graph sample.
    """

    def __init__(self, split: str, root: str, transform=None, pre_transform=None, heterogeneous: bool = True):
        assert split in {"train", "val", "test"}
        self.split = split
        self.heterogeneous = heterogeneous  # True: 异质图（类别隔离空间），False: 同质图（全局空间）
        super().__init__(root, transform, pre_transform)  # 按需调用 download()、process()，并设置 processed_dir、processed_paths 等。
        self.data, self.slices = torch.load(self.processed_paths[0])  # 用 processed_paths[0]（即当前 split 的 .pt）加载整份图到 self.data 和 self.slices，这样 len(self) 和 self[i] 就都由基类按切片正确工作。

        # 加载按类型/关系族分组的子类别分布（如果存在）
        node_subtype_by_type = {}
        edge_subtype_by_family = {}
        node_type_distribution = {}
        edge_family_distribution = {}
        if self.heterogeneous:
            node_subtype_path = osp.join(self.processed_dir, f"{self.split}_node_subtype_by_type.pickle")
            edge_subtype_path = osp.join(self.processed_dir, f"{self.split}_edge_subtype_by_family.pickle")
            node_type_dist_path = osp.join(self.processed_dir, f"{self.split}_node_type_distribution.pickle")
            edge_family_dist_path = osp.join(self.processed_dir, f"{self.split}_edge_family_distribution.pickle")
            if osp.exists(node_subtype_path):
                node_subtype_by_type = load_pickle(node_subtype_path)
                print(f"[DEBUG] ACMSubgraphsDataset.__init__: 已加载node_subtype_by_type从 {node_subtype_path}, 包含{len(node_subtype_by_type)}个节点类型")
            if osp.exists(edge_subtype_path):
                edge_subtype_by_family = load_pickle(edge_subtype_path)
                print(f"[DEBUG] ACMSubgraphsDataset.__init__: 已加载edge_subtype_by_family从 {edge_subtype_path}, 包含{len(edge_subtype_by_family)}个关系族")
            if osp.exists(node_type_dist_path):
                node_type_distribution = load_pickle(node_type_dist_path)
                print(f"[DEBUG] ACMSubgraphsDataset.__init__: 已加载node_type_distribution从 {node_type_dist_path}, 包含{len(node_type_distribution)}个节点类型")
            if osp.exists(edge_family_dist_path):
                edge_family_distribution = load_pickle(edge_family_dist_path)
                print(f"[DEBUG] ACMSubgraphsDataset.__init__: 已加载edge_family_distribution从 {edge_family_dist_path}, 包含{len(edge_family_distribution)}个类型对")
        
        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            node_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
            charge_types=None,
            valencies=None,
            node_subtype_by_type=node_subtype_by_type,
            edge_subtype_by_family=edge_subtype_by_family,
            node_type_distribution=node_type_distribution,
            edge_family_distribution=edge_family_distribution,
        )

    @property
    def raw_file_names(self):
        # We treat the existing subgraph_* folders (under root) as raw data, not root/raw.
        # This marker file is created to satisfy PyG's raw-file checks.
        return ["_acm_subgraphs_ready.txt"]

    @property
    def processed_file_names(self):
        files = [
            f"{self.split}.pt",  # 存储处理后的图数据(data, slices)；pt文件即PyTorch对象的"工程格式（本质是 pickle）"
            f"{self.split}_n.pickle",  # 存储节点数量分布统计，字典，键为节点数，值为出现频率；pickle文件即Python对象的二进制快照，支持各种对象
            f"{self.split}_node_types.npy",  # 存储节点类型分布统计，numpy 数组，键为节点类型，值为出现频率
            f"{self.split}_bond_types.npy",  # 存储边类型分布统计，numpy 数组，键为边类型，值为出现频率
            "splits.pt",  # 存储训练、验证、测试集的索引，字典 {"train": [...], "val": [...], "test": [...]}
            "vocab.json",  # 存储节点类型和边类型映射关系，字典 {"node_type_names": [...], "node_type2id": {...}, "edge_label2id": {...}}
        ]
        # 如果是异质图，添加关系族平均边数文件和子类别分布文件
        if self.heterogeneous:
            files.append(f"{self.split}_edge_family_avg_counts.pickle")
            files.append(f"{self.split}_node_subtype_by_type.pickle")  # 按节点类型分组的子类别分布
            files.append(f"{self.split}_edge_subtype_by_family.pickle")  # 按关系族分组的边子类别分布
            files.append(f"{self.split}_node_type_distribution.pickle")  # 节点类型分布
            files.append(f"{self.split}_edge_family_distribution.pickle")  # 关系族分布
        return files

    def download(self):
        _touch(osp.join(self.raw_dir, self.raw_file_names[0]))

    def _load_or_create_splits(self, n_graphs: int, seed: int = 1234, val_ratio: float = 0.1, test_ratio: float = 0.1):
        split_path = osp.join(self.processed_dir, "splits.pt")
        if osp.exists(split_path):
            return torch.load(split_path)

        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(n_graphs, generator=g)

        # 当图很少时 round(ratio) 可能为 0，导致 val/test 为空、collate 报错；保证每份至少 1 个（若 n>=3）
        n_test = int(round(n_graphs * test_ratio))
        n_val = int(round(n_graphs * val_ratio))
        if n_graphs >= 3:
            if n_test < 1:
                n_test = 1
            if n_val < 1:
                n_val = 1
        n_train = n_graphs - n_val - n_test
        if n_train < 1:
            n_train = 1
            n_test = min(n_test, n_graphs - 2)
            n_val = n_graphs - n_train - n_test

        splits = {
            "train": perm[:n_train],
            "val": perm[n_train : n_train + n_val],
            "test": perm[n_train + n_val :],
        }
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(splits, split_path)
        return splits

    def _load_or_create_vocab(self, subgraph_dirs: List[str]) -> Tuple[List[str], Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
        vocab_path = osp.join(self.processed_dir, "vocab.json")
        if osp.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            node_type_names = vocab["node_type_names"]
            node_type2id = {k: int(v) for k, v in vocab["node_type2id"].items()}
            # Support both old format (edge_label_dir2id) and new format (edge_label2id)
            edge_label2id = vocab.get("edge_label2id", vocab.get("edge_label_dir2id", {}))
            edge_label2id = {k: int(v) for k, v in edge_label2id.items()}
            edge_family2id = vocab.get("edge_family2id", {})
            edge_family2id = {k: int(v) for k, v in edge_family2id.items()}
            edge_family_offsets = vocab.get("edge_family_offsets", {})
            edge_family_offsets = {k: int(v) for k, v in edge_family_offsets.items()}
            # 调试信息：记录从缓存加载的edge_label2id
            print(f"[DEBUG] _load_or_create_vocab: 从缓存加载 edge_label2id数量={len(edge_label2id)}, 内容={list(edge_label2id.keys())}")
            # 检查是否与当前的 heterogeneous 设置一致
            vocab_heterogeneous = len(edge_family2id) > 0
            if vocab_heterogeneous != self.heterogeneous:
                # 如果设置不一致，删除缓存重新生成
                print(f"Warning: heterogeneous setting ({self.heterogeneous}) differs from cached vocab ({vocab_heterogeneous}). Regenerating vocab...")
                os.remove(vocab_path)
                return self._load_or_create_vocab(subgraph_dirs)
            return node_type_names, node_type2id, edge_label2id, edge_family2id, edge_family_offsets

        meta0 = _load_meta(subgraph_dirs[0])
        node_type_names, node_type2id, edge_label2id, edge_family2id, edge_family_offsets = _build_vocab_from_meta(meta0, self.heterogeneous)

        os.makedirs(self.processed_dir, exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "node_type_names": node_type_names,
                    "node_type2id": node_type2id,
                    "edge_label2id": edge_label2id,
                    "edge_family2id": edge_family2id,
                    "edge_family_offsets": edge_family_offsets,
                    "heterogeneous": self.heterogeneous,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return node_type_names, node_type2id, edge_label2id, edge_family2id, edge_family_offsets

    def process(self):
        # NOTE: raw data lives directly under self.root, not self.raw_dir
        subgraph_dirs = _list_subgraph_dirs(self.root)
        if len(subgraph_dirs) == 0:
            raise RuntimeError(
                f"No subgraph_* directories found under {self.root}. Expected data/ACM_subgraphs/subgraph_XXX/..."
            )

        node_type_names, node_type2id, edge_label2id, edge_family2id, edge_family_offsets = self._load_or_create_vocab(subgraph_dirs)
        
        # node diffusion state is over node subtypes (disjoint ranges per node type)
        meta0 = _load_meta(subgraph_dirs[0])
        type_sizes = [len(meta0["schema_by_type"][t]) for t in node_type_names]
        num_node_types = int(sum(type_sizes))
        
        # edge diffusion state is over directed edge subtypes (label strings from meta.json)
        num_edge_types = 1 + len(edge_label2id)  # include 'no-edge' class at index 0

        splits = self._load_or_create_splits(n_graphs=len(subgraph_dirs))
        idx = splits[self.split].tolist()
        chosen_dirs = [subgraph_dirs[i] for i in idx]

        # 计算全局的type_offsets（所有图共享，因为schema相同）
        # 使用第一个图来计算type_offsets（所有图的schema应该相同）
        meta0 = _load_meta(chosen_dirs[0])
        nodes0 = torch.load(osp.join(chosen_dirs[0], "nodes.pt"), map_location="cpu")
        _, global_type_offsets, global_type_sizes_list = _build_node_state_id(node_type_names, nodes0, node_type2id)
        # 将type_sizes_list转换为字典，方便使用
        global_type_sizes = {t: size for t, size in zip(node_type_names, global_type_sizes_list)}
        
        data_list: List[Data] = []
        for sd in chosen_dirs:
            meta = _load_meta(sd)
            nodes = torch.load(osp.join(sd, "nodes.pt"), map_location="cpu")
            edges = torch.load(osp.join(sd, "edges.pt"), map_location="cpu")

            # Node diffusion target: global subtype id (disjoint ranges per node type)
            node_state, type_offsets, _ = _build_node_state_id(node_type_names, nodes, node_type2id)

            # Keep coarse node type id + local subtype for downstream constraints (future)
            node_type_id, node_subtype_local, offsets = _concat_node_fields(nodes, node_type_names)
            for t in node_type_names:
                off = offsets[t]
                n = int(nodes[t]["subtype"].shape[0])
                node_type_id[off : off + n] = int(node_type2id[t])

            edge_index, edge_attr, edge_family = _build_edges_for_graph(
                edges, meta, offsets, edge_label2id, 
                edge_family_offsets=edge_family_offsets if self.heterogeneous else None,
                edge_family2id=edge_family2id if self.heterogeneous else None,
                heterogeneous=self.heterogeneous
            )

            data = Data(
                x=node_state,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_type=node_type_id,
                node_subtype=node_subtype_local,
                edge_family=edge_family,  # 关系族 id，用于约束（仅异质图）
                y=torch.zeros((1, 0), dtype=torch.float),
            )

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        # statistics for this split (node/edge distributions)
        num_nodes = node_counts(data_list)
        node_types = atom_type_counts(data_list, num_classes=num_node_types)
        # 调试信息：记录num_edge_types的值
        print(f"[DEBUG] ACMSubgraphsDataset.process: num_edge_types={num_edge_types}, len(edge_label2id)={len(edge_label2id)}")
        bond_types = edge_counts(data_list, num_bond_types=num_edge_types)
        print(f"[DEBUG] ACMSubgraphsDataset.process: bond_types长度={len(bond_types)}")
        
        # 异质图：计算节点类型分布（每个类型有多少节点）
        node_type_distribution = {}
        if self.heterogeneous and len(node_type_names) > 0:
            print(f"[DEBUG] ACMSubgraphsDataset.process: 开始计算节点类型分布")
            node_type_counts_dict = {t: 0 for t in node_type_names}
            total_nodes = 0
            for data in data_list:
                if hasattr(data, 'node_type'):
                    for node_type_name, node_type_id in node_type2id.items():
                        count = (data.node_type == node_type_id).sum().item()
                        node_type_counts_dict[node_type_name] += count
                        total_nodes += count
            
            if total_nodes > 0:
                for node_type_name in node_type_names:
                    node_type_distribution[node_type_name] = node_type_counts_dict[node_type_name] / total_nodes
                print(f"[DEBUG] ACMSubgraphsDataset.process: 节点类型分布: {node_type_distribution}")
            else:
                # 如果没有数据，使用均匀分布
                uniform_prob = 1.0 / len(node_type_names)
                for node_type_name in node_type_names:
                    node_type_distribution[node_type_name] = uniform_prob
                print(f"[DEBUG] ACMSubgraphsDataset.process: 警告！没有节点数据，使用均匀分布: {node_type_distribution}")
        
        # 异质图：计算关系族分布（两个节点类型之间，每个关系族的比例）
        edge_family_distribution = {}
        # 从第一个图的meta中获取fam_endpoints（所有图的schema应该相同）
        fam_endpoints_meta = None
        if self.heterogeneous and len(chosen_dirs) > 0:
            meta0 = _load_meta(chosen_dirs[0])
            fam_endpoints_meta = meta0.get("fam_endpoints", {})
        
        if self.heterogeneous and len(edge_family2id) > 0 and fam_endpoints_meta:
            print(f"[DEBUG] ACMSubgraphsDataset.process: 开始计算关系族分布")
            # 构建 (src_type, dst_type) -> {fam_name: count} 的映射
            type_pair_fam_counts = {}
            for data in data_list:
                if hasattr(data, 'edge_family') and data.edge_family is not None and hasattr(data, 'node_type'):
                    edge_family = data.edge_family  # (E,)
                    edge_attr = data.edge_attr  # (E,)
                    node_type = data.node_type  # (N,)
                    edge_index = data.edge_index  # (2, E)
                    
                    # 只统计非no-edge的边
                    valid_mask = edge_attr != 0
                    if valid_mask.any():
                        valid_edge_index = edge_index[:, valid_mask]  # (2, E_valid)
                        valid_edge_family = edge_family[valid_mask]  # (E_valid,)
                        
                        for i in range(valid_edge_index.shape[1]):
                            src_idx = valid_edge_index[0, i].item()
                            dst_idx = valid_edge_index[1, i].item()
                            src_type_id = node_type[src_idx].item()
                            dst_type_id = node_type[dst_idx].item()
                            fam_id = valid_edge_family[i].item()
                            
                            # 找到对应的节点类型名称
                            src_type_name = None
                            dst_type_name = None
                            for t, tid in node_type2id.items():
                                if tid == src_type_id:
                                    src_type_name = t
                                if tid == dst_type_id:
                                    dst_type_name = t
                            
                            if src_type_name and dst_type_name:
                                type_pair = (src_type_name, dst_type_name)
                                if type_pair not in type_pair_fam_counts:
                                    type_pair_fam_counts[type_pair] = {}
                                
                                # 找到对应的关系族名称
                                fam_name = None
                                for fname, fid in edge_family2id.items():
                                    if fid == fam_id:
                                        fam_name = fname
                                        break
                                
                                if fam_name:
                                    if fam_name not in type_pair_fam_counts[type_pair]:
                                        type_pair_fam_counts[type_pair][fam_name] = 0
                                    type_pair_fam_counts[type_pair][fam_name] += 1
            
            # 归一化每个类型对的关系族分布
            for type_pair, fam_counts in type_pair_fam_counts.items():
                total = sum(fam_counts.values())
                if total > 0:
                    edge_family_distribution[type_pair] = {fam: count / total for fam, count in fam_counts.items()}
                else:
                    # 如果没有数据，使用均匀分布
                    fam_names = [fname for fname in edge_family2id.keys() 
                               if fam_endpoints_meta.get(fname, {}).get("src_type") == type_pair[0] 
                               and fam_endpoints_meta.get(fname, {}).get("dst_type") == type_pair[1]]
                    if len(fam_names) > 0:
                        uniform_prob = 1.0 / len(fam_names)
                        edge_family_distribution[type_pair] = {fam: uniform_prob for fam in fam_names}
            
            print(f"[DEBUG] ACMSubgraphsDataset.process: 关系族分布: {edge_family_distribution}")
        
        # 异质图：计算按节点类型分组的子类别分布和按关系族分组的边子类别分布
        node_subtype_by_type = {}
        edge_subtype_by_family = {}
        if self.heterogeneous and global_type_offsets and len(node_type_names) > 0:
            print(f"[DEBUG] ACMSubgraphsDataset.process: 开始计算按类型分组的子类别分布")
            
            # 计算每个节点类型的子类别分布
            for node_type_name in node_type_names:
                if node_type_name not in global_type_offsets:
                    continue
                offset = global_type_offsets[node_type_name]
                type_size = global_type_sizes.get(node_type_name, 0)
                if type_size <= 0:
                    continue
                
                # 统计该节点类型的所有子类别
                subtype_counts = torch.zeros(type_size, dtype=torch.float)
                for data in data_list:
                    if hasattr(data, 'node_type') and hasattr(data, 'x'):
                        node_type_id = node_type2id.get(node_type_name, -1)
                        if node_type_id < 0:
                            continue
                        # 找到属于该节点类型的节点
                        node_type_mask = (data.node_type == node_type_id)
                        if node_type_mask.any():
                            # 获取这些节点的全局子类别ID
                            # data.x 是离散的全局子类别ID（LongTensor）
                            node_subtypes = data.x[node_type_mask]  # (N_type,)
                            if node_subtypes.dim() == 0:
                                node_subtypes = node_subtypes.unsqueeze(0)
                            # 转换为局部子类别ID（减去offset）
                            local_subtypes = node_subtypes.long() - offset
                            # 统计（只统计在有效范围内的）
                            valid_mask = (local_subtypes >= 0) & (local_subtypes < type_size)
                            if valid_mask.any():
                                local_subtypes_valid = local_subtypes[valid_mask].long()
                                unique, counts = torch.unique(local_subtypes_valid, return_counts=True)
                                for u, c in zip(unique, counts):
                                    if 0 <= u < type_size:
                                        subtype_counts[u] += c.float()
                
                # 归一化
                if subtype_counts.sum() > 0:
                    subtype_counts = subtype_counts / subtype_counts.sum()
                else:
                    # 如果没有数据，使用均匀分布
                    subtype_counts = torch.ones(type_size, dtype=torch.float) / type_size
                
                node_subtype_by_type[node_type_name] = subtype_counts
                print(f"[DEBUG] ACMSubgraphsDataset.process: {node_type_name}的子类别分布: {subtype_counts.tolist()}")
            
            # 计算每个关系族的边子类别分布
            if edge_family_offsets and len(edge_family2id) > 0:
                print(f"[DEBUG] ACMSubgraphsDataset.process: 开始计算按关系族分组的边子类别分布")
                for fam_name, fam_id in edge_family2id.items():
                    if fam_name not in edge_family_offsets:
                        continue
                    offset = edge_family_offsets[fam_name]
                    # 计算该关系族的子类别数量
                    next_offset = num_edge_types
                    for other_fam_name, other_offset in edge_family_offsets.items():
                        if other_offset > offset and other_offset < next_offset:
                            next_offset = other_offset
                    num_subtypes = next_offset - offset
                    if num_subtypes <= 0:
                        continue
                    
                    # 统计该关系族的边子类别
                    subtype_counts = torch.zeros(num_subtypes, dtype=torch.float)
                    for data in data_list:
                        if hasattr(data, 'edge_family') and data.edge_family is not None:
                            # 找到属于该关系族的边
                            fam_mask = (data.edge_family == fam_id) & (data.edge_attr != 0)
                            if fam_mask.any():
                                edge_attrs = data.edge_attr[fam_mask]  # (E_fam,)
                                # 转换为局部子类别ID（减去offset）
                                local_subtypes = edge_attrs - offset
                                # 统计（只统计在有效范围内的）
                                valid_mask = (local_subtypes >= 0) & (local_subtypes < num_subtypes)
                                if valid_mask.any():
                                    local_subtypes_valid = local_subtypes[valid_mask].long()
                                    unique, counts = torch.unique(local_subtypes_valid, return_counts=True)
                                    for u, c in zip(unique, counts):
                                        if 0 <= u < num_subtypes:
                                            subtype_counts[u] += c.float()
                    
                    # 归一化
                    if subtype_counts.sum() > 0:
                        subtype_counts = subtype_counts / subtype_counts.sum()
                    else:
                        # 如果没有数据，使用均匀分布
                        subtype_counts = torch.ones(num_subtypes, dtype=torch.float) / num_subtypes
                    
                    edge_subtype_by_family[fam_name] = subtype_counts
                    print(f"[DEBUG] ACMSubgraphsDataset.process: {fam_name}的边子类别分布: {subtype_counts.tolist()}")
        
        # 计算每个关系族的平均真实边数（用于采样时的 |Eq| = km 计算）
        edge_family_edge_counts = {}
        if self.heterogeneous and len(edge_family2id) > 0:
            print(f"[DEBUG] ACMSubgraphsDataset.process: 开始计算edge_family_avg_edge_counts, edge_family2id={list(edge_family2id.keys())}")
            num_data_with_edge_family = 0
            for data in data_list:
                if hasattr(data, 'edge_family') and data.edge_family is not None:
                    num_data_with_edge_family += 1
                    edge_family = data.edge_family  # (E,) - 每条边所属的关系族 ID
                    edge_attr = data.edge_attr  # (E,) - 边属性（全局 ID）
                    
                    # 统计每个关系族的边数
                    for fam_name, fam_id in edge_family2id.items():
                        # 找到属于该关系族的边（edge_family == fam_id 且 edge_attr != 0）
                        fam_mask = (edge_family == fam_id) & (edge_attr != 0)
                        num_fam_edges = fam_mask.sum().item()
                        
                        if fam_name not in edge_family_edge_counts:
                            edge_family_edge_counts[fam_name] = []
                        edge_family_edge_counts[fam_name].append(num_fam_edges)
            
            print(f"[DEBUG] ACMSubgraphsDataset.process: 有edge_family的数据数量={num_data_with_edge_family}/{len(data_list)}")
            print(f"[DEBUG] ACMSubgraphsDataset.process: edge_family_edge_counts统计={[(k, len(v)) for k, v in edge_family_edge_counts.items()]}")
            
            # 计算每个关系族的平均边数
            edge_family_avg_edge_counts = {}
            for fam_name, counts in edge_family_edge_counts.items():
                if len(counts) > 0:
                    edge_family_avg_edge_counts[fam_name] = sum(counts) / len(counts)
                else:
                    edge_family_avg_edge_counts[fam_name] = 0.0
            
            print(f"[DEBUG] ACMSubgraphsDataset.process: edge_family_avg_edge_counts={edge_family_avg_edge_counts}")
        else:
            edge_family_avg_edge_counts = {}
            if not self.heterogeneous:
                print(f"[DEBUG] ACMSubgraphsDataset.process: 非异质图模式，不计算edge_family_avg_edge_counts")
            elif len(edge_family2id) == 0:
                print(f"[DEBUG] ACMSubgraphsDataset.process: edge_family2id为空，不计算edge_family_avg_edge_counts")

        torch.save(self.collate(data_list), self.processed_paths[0])
        save_pickle(num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], node_types)
        np.save(self.processed_paths[3], bond_types)
        
        # 保存每个关系族的平均真实边数（即使为空也保存，以便后续加载时知道情况）
        edge_family_counts_path = osp.join(self.processed_dir, f"{self.split}_edge_family_avg_counts.pickle")
        if edge_family_avg_edge_counts:
            save_pickle(edge_family_avg_edge_counts, edge_family_counts_path)
            print(f"[DEBUG] ACMSubgraphsDataset.process: 已保存edge_family_avg_edge_counts到 {edge_family_counts_path}")
            print(f"[DEBUG] ACMSubgraphsDataset.process: 内容={edge_family_avg_edge_counts}")
        else:
            # 即使为空也保存，以便后续加载时知道情况（而不是使用默认值）
            save_pickle(edge_family_avg_edge_counts, edge_family_counts_path)
            print(f"[DEBUG] ACMSubgraphsDataset.process: 警告！edge_family_avg_edge_counts为空，已保存空字典到 {edge_family_counts_path}")
            print(f"[DEBUG] ACMSubgraphsDataset.process: 这可能导致初始化时无法生成边，请检查edge_family属性是否正确添加")
        
        # 保存按节点类型分组的子类别分布和按关系族分组的边子类别分布
        if self.heterogeneous:
            node_subtype_path = osp.join(self.processed_dir, f"{self.split}_node_subtype_by_type.pickle")
            edge_subtype_path = osp.join(self.processed_dir, f"{self.split}_edge_subtype_by_family.pickle")
            node_type_dist_path = osp.join(self.processed_dir, f"{self.split}_node_type_distribution.pickle")
            edge_family_dist_path = osp.join(self.processed_dir, f"{self.split}_edge_family_distribution.pickle")
            save_pickle(node_subtype_by_type, node_subtype_path)
            save_pickle(edge_subtype_by_family, edge_subtype_path)
            save_pickle(node_type_distribution, node_type_dist_path)
            save_pickle(edge_family_distribution, edge_family_dist_path)
            print(f"[DEBUG] ACMSubgraphsDataset.process: 已保存node_subtype_by_type到 {node_subtype_path}, 包含{len(node_subtype_by_type)}个节点类型")
            print(f"[DEBUG] ACMSubgraphsDataset.process: 已保存edge_subtype_by_family到 {edge_subtype_path}, 包含{len(edge_subtype_by_family)}个关系族")
            print(f"[DEBUG] ACMSubgraphsDataset.process: 已保存node_type_distribution到 {node_type_dist_path}, 包含{len(node_type_distribution)}个节点类型")
            print(f"[DEBUG] ACMSubgraphsDataset.process: 已保存edge_family_distribution到 {edge_family_dist_path}, 包含{len(edge_family_distribution)}个类型对")


class ACMSubgraphsDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset.name
        self.datadir = cfg.dataset.datadir
        # 数据路径：绝对路径直接用；相对路径则先尝试 (cwd / datadir)，否则 (cwd 上一级 / datadir)，以兼容 SparseDiff/data 与 项目父级/data 两种布局
        if osp.isabs(self.datadir):
            root_path = self.datadir
        else:
            cwd = pathlib.Path(get_original_cwd())
            cand = cwd / self.datadir
            if cand.exists():
                root_path = str(cand)
            else:
                base_path = cwd.parents[0]
                root_path = osp.join(base_path, self.datadir)
        pre_transform = RemoveYTransform() # 将 data.y（图级别标签）清空，设置为空的零张量

        # 从配置读取 heterogeneous 参数，默认为 True（异质图）
        heterogeneous = getattr(cfg.dataset, "heterogeneous", True)
        
        datasets = {
            "train": ACMSubgraphsDataset(split="train", root=root_path, pre_transform=pre_transform, heterogeneous=heterogeneous),
            "val": ACMSubgraphsDataset(split="val", root=root_path, pre_transform=pre_transform, heterogeneous=heterogeneous),
            "test": ACMSubgraphsDataset(split="test", root=root_path, pre_transform=pre_transform, heterogeneous=heterogeneous),
        }

        self.statistics = {
            "train": datasets["train"].statistics,
            "val": datasets["val"].statistics,
            "test": datasets["test"].statistics,
        }

        super().__init__(cfg, datasets)
        super().prepare_dataloader()
        self.inner = self.train_dataset


class ACMSubgraphsInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.is_molecular = False
        # Set to False to avoid importing SPECTRE-specific metrics (which depend on DGL).
        # For this hetero citation dataset we can rely on the generic sampling metrics
        # (node/edge distributions, connected components, etc.) first.
        self.spectre = False
        self.use_charge = False
        self.dataset_name = datamodule.dataset_name
        self.node_types = datamodule.inner.statistics.node_types
        self.bond_types = datamodule.inner.statistics.bond_types
        super().complete_infos(
            datamodule.statistics, len(datamodule.inner.statistics.node_types)
        )

        # For non-molecular datasets, y and charge are not used.
        # input_dims will be updated later by compute_input_dims() after extra features are attached.
        from sparse_diffusion.utils import PlaceHolder

        # 先设置heterogeneous属性，再使用
        self.heterogeneous = getattr(datamodule.cfg.dataset, "heterogeneous", True)
        
        # 对于异质图，output_dims.X应该使用从schema计算的num_node_types（所有可能的子类别数）
        # 而不是len(self.node_types)（只统计实际出现的子类别数）
        # 这样可以确保模型维度包含所有可能的子类别，即使训练数据中某些子类别没有出现
        num_node_subtypes = len(self.node_types)  # 默认使用实际出现的类别数
        if self.heterogeneous:
            # 从第一个子图的meta.json获取schema信息，计算所有可能的子类别数
            subgraph_dirs = _list_subgraph_dirs(datamodule.inner.root)
            if len(subgraph_dirs) > 0:
                meta0 = _load_meta(subgraph_dirs[0])
                schema_by_type = meta0.get("schema_by_type", {})
                # 从vocab.json或meta.json获取node_type_names
                vocab_path = osp.join(datamodule.inner.processed_dir, "vocab.json")
                if osp.exists(vocab_path):
                    import json
                    with open(vocab_path, "r", encoding="utf-8") as f:
                        vocab = json.load(f)
                    node_type_names = vocab.get("node_type_names", [])
                else:
                    node_type_names = list(meta0.get("node_types", []))
                
                if node_type_names and schema_by_type:
                    type_sizes = [len(schema_by_type.get(t, [])) for t in node_type_names]
                    num_node_subtypes = sum(type_sizes)  # 所有可能的子类别总数

        # 对于异质图，output_dims.E应该使用1+len(edge_label2id)，而不是len(self.bond_types)
        # 因为bond_types的长度取决于传入edge_counts的num_bond_types参数
        num_edge_types = len(self.bond_types)  # 默认使用bond_types的长度
        if self.heterogeneous:
            # 从vocab.json获取edge_label2id，计算正确的num_edge_types
            vocab_path = osp.join(datamodule.inner.processed_dir, "vocab.json")
            if osp.exists(vocab_path):
                import json
                with open(vocab_path, "r", encoding="utf-8") as f:
                    vocab = json.load(f)
                edge_label2id = vocab.get("edge_label2id", vocab.get("edge_label_dir2id", {}))
                if edge_label2id:
                    num_edge_types = 1 + len(edge_label2id)  # 包括no-edge类型
                    print(f"[DEBUG] ACMSubgraphsInfos.__init__: 从vocab.json获取edge_label2id数量={len(edge_label2id)}, num_edge_types={num_edge_types}")
        
        self.input_dims = PlaceHolder(
            X=num_node_subtypes, E=num_edge_types, y=0, charge=0
        )
        self.output_dims = PlaceHolder(
            X=num_node_subtypes, E=num_edge_types, y=0, charge=0
        )
        self.statistics = {
            "train": datamodule.statistics["train"],
            "val": datamodule.statistics["val"],
            "test": datamodule.statistics["test"],
        }
        
        # 存储关系族信息（用于异质图边扩散）
        if self.heterogeneous:
            # 从 dataset 加载关系族信息
            vocab_path = osp.join(datamodule.inner.processed_dir, "vocab.json")
            if osp.exists(vocab_path):
                import json
                with open(vocab_path, "r", encoding="utf-8") as f:
                    vocab = json.load(f)
                
                # 加载节点类型信息
                self.node_type_names = vocab.get("node_type_names", [])
                self.node_type2id = {k: int(v) for k, v in vocab.get("node_type2id", {}).items()}
                self.edge_family2id = {k: int(v) for k, v in vocab.get("edge_family2id", {}).items()}
                self.edge_family_offsets = {k: int(v) for k, v in vocab.get("edge_family_offsets", {}).items()}
                
                # 计算 type_offsets（从第一个子图的 meta.json）
                subgraph_dirs = _list_subgraph_dirs(datamodule.inner.root)
                if len(subgraph_dirs) > 0:
                    meta0 = _load_meta(subgraph_dirs[0])
                    schema_by_type = meta0.get("schema_by_type", {})
                    type_sizes = [len(schema_by_type.get(t, [])) for t in self.node_type_names]
                    self.type_offsets = {}
                    cur = 0
                    for t, size in zip(self.node_type_names, type_sizes):
                        self.type_offsets[t] = cur
                        cur += size
                    # 子类别名称列表（按全局 id 顺序），用于 wandb / chain / graph 显示「子类别」
                    self.node_subtype_names = []
                    for t in self.node_type_names:
                        for name in schema_by_type.get(t, []):
                            self.node_subtype_names.append(f"{t}_{name}" if name else str(t))
                    self.node_subtype_decoder = {i: n for i, n in enumerate(self.node_subtype_names)}
                    # 每个关系族的边子类别名称（no-edge + 子类别名），用于 wandb 显示「子类别」
                    self.edge_subtype_names_by_family = {}
                    fam_id2label = meta0.get("fam_id2label", {})
                    for fam in sorted(self.edge_family_offsets.keys(), key=lambda f: self.edge_family_offsets[f]):
                        id2label = fam_id2label.get(fam, {})
                        names = ["no-edge"]
                        for k in sorted(id2label.keys(), key=lambda x: int(x)):
                            lbl = id2label[k]
                            if lbl is not None and (not isinstance(lbl, str) or lbl.strip()):
                                names.append(lbl)
                        self.edge_subtype_names_by_family[fam] = names
                else:
                    self.type_offsets = {}
                    self.node_subtype_names = []
                    self.node_subtype_decoder = {}
                    self.edge_subtype_names_by_family = {}
                
                edge_label2id = vocab.get("edge_label2id", {})
                self.edge_label2id = {k: int(v) for k, v in edge_label2id.items()}  # 用于 chain/graph 显示边子标签
                
                # 存储端点类型信息（用于计算非存在边数）
                # 从第一个子图的 meta.json 加载（如果还没加载）
                if not hasattr(self, 'fam_endpoints'):
                    subgraph_dirs = _list_subgraph_dirs(datamodule.inner.root)
                    if len(subgraph_dirs) > 0:
                        meta0 = _load_meta(subgraph_dirs[0])
                        self.fam_endpoints = meta0.get("fam_endpoints", {})
                    else:
                        self.fam_endpoints = {}
                
                # 从statistics中获取节点类型分布、关系族分布和子类别分布
                train_stats = datamodule.statistics.get("train")
                if train_stats:
                    self.node_type_distribution = getattr(train_stats, "node_type_distribution", {}) or {}
                    self.edge_family_distribution = getattr(train_stats, "edge_family_distribution", {}) or {}
                    self.node_subtype_by_type = getattr(train_stats, "node_subtype_by_type", {}) or {}
                    self.edge_subtype_by_family = getattr(train_stats, "edge_subtype_by_family", {}) or {}
                    print(f"[DEBUG] ACMSubgraphsInfos.__init__: 已设置node_type_distribution={len(self.node_type_distribution)}个类型")
                    print(f"[DEBUG] ACMSubgraphsInfos.__init__: 已设置edge_family_distribution={len(self.edge_family_distribution)}个类型对")
                    print(f"[DEBUG] ACMSubgraphsInfos.__init__: 已设置node_subtype_by_type={len(self.node_subtype_by_type)}个类型")
                    print(f"[DEBUG] ACMSubgraphsInfos.__init__: 已设置edge_subtype_by_family={len(self.edge_subtype_by_family)}个关系族")
                else:
                    self.node_type_distribution = {}
                    self.edge_family_distribution = {}
                    self.node_subtype_by_type = {}
                    self.edge_subtype_by_family = {}
                    print(f"[DEBUG] ACMSubgraphsInfos.__init__: 警告！train statistics不存在，使用空字典")
                
                # 计算每个关系族的边类型分布
                self.edge_family_marginals = {}
                # 加载每个关系族的平均真实边数（用于计算 m_fam / W_fam）
                edge_family_counts_path = osp.join(datamodule.inner.processed_dir, "train_edge_family_avg_counts.pickle")
                if osp.exists(edge_family_counts_path):
                    self.edge_family_avg_edge_counts = load_pickle(edge_family_counts_path)
                    print(f"[DEBUG] ACMSubgraphsInfos.__init__: 已加载edge_family_avg_edge_counts从 {edge_family_counts_path}")
                    print(f"[DEBUG] ACMSubgraphsInfos.__init__: edge_family_avg_edge_counts={self.edge_family_avg_edge_counts}")
                    # 检查是否为空字典
                    if not self.edge_family_avg_edge_counts:
                        print(f"[DEBUG] ACMSubgraphsInfos.__init__: 警告！edge_family_avg_edge_counts为空字典，这可能导致初始化时无法生成边")
                    # 检查是否有值为0的关系族
                    zero_count_fams = [fam for fam, count in self.edge_family_avg_edge_counts.items() if count == 0.0]
                    if zero_count_fams:
                        print(f"[DEBUG] ACMSubgraphsInfos.__init__: 警告！以下关系族的平均边数为0: {zero_count_fams}")
                else:
                    # 如果没有保存的文件，使用默认值（从 bond_types 估算）
                    print(f"[DEBUG] ACMSubgraphsInfos.__init__: 警告！未找到edge_family_avg_edge_counts文件: {edge_family_counts_path}")
                    print(f"[DEBUG] ACMSubgraphsInfos.__init__: 使用默认值10.0（这可能导致初始化时无法生成边）")
                    self.edge_family_avg_edge_counts = {}
                    for fam_name in self.edge_family2id.keys():
                        # 估算：使用一个简单的启发式值
                        self.edge_family_avg_edge_counts[fam_name] = 10.0  # 默认值，实际应该从数据中计算
                    print(f"[DEBUG] ACMSubgraphsInfos.__init__: 使用默认值后的edge_family_avg_edge_counts={self.edge_family_avg_edge_counts}")

                # 计算期望节点数（用于按关系族计算 W_fam）
                mean_num_nodes = None
                if train_stats and getattr(train_stats, "num_nodes", None):
                    try:
                        num_nodes_counter = train_stats.num_nodes
                        total_graphs = sum(num_nodes_counter.values())
                        if total_graphs > 0:
                            mean_num_nodes = sum(
                                int(n) * int(c) for n, c in num_nodes_counter.items()
                            ) / float(total_graphs)
                    except Exception as _e:
                        print(f"[DEBUG] ACMSubgraphsInfos.__init__: 计算mean_num_nodes失败: {_e}")
                        mean_num_nodes = None

                for fam_name, fam_id in self.edge_family2id.items():
                    offset = self.edge_family_offsets[fam_name]
                    # 找到该关系族的所有标签
                    fam_labels = [lbl for lbl, gid in edge_label2id.items() 
                                 if lbl.startswith(fam_name + ":")]
                    num_subtypes = len(fam_labels)

                    # 优先使用按关系族的 m/W 计算边际分布
                    fam_endpoints = getattr(self, "fam_endpoints", {}) or {}
                    edge_family_avg_edge_counts = getattr(self, "edge_family_avg_edge_counts", {}) or {}
                    can_compute_mw = (
                        mean_num_nodes is not None
                        and fam_name in fam_endpoints
                        and self.node_type_distribution
                        and edge_family_avg_edge_counts
                    )

                    if can_compute_mw:
                        src_type = fam_endpoints[fam_name].get("src_type")
                        dst_type = fam_endpoints[fam_name].get("dst_type")
                        p_src = self.node_type_distribution.get(src_type, 0.0)
                        p_dst = self.node_type_distribution.get(dst_type, 0.0)
                        n_src = mean_num_nodes * p_src
                        n_dst = mean_num_nodes * p_dst
                        if src_type == dst_type:
                            w_fam = max(n_src * max(n_src - 1.0, 0.0), 0.0)
                        else:
                            w_fam = max(n_src * n_dst, 0.0)
                        m_fam = float(edge_family_avg_edge_counts.get(fam_name, 0.0))
                        u1 = (m_fam / w_fam) if w_fam > 0 else 0.0
                        u1 = max(0.0, min(1.0, u1))
                        u0 = 1.0 - u1

                        if num_subtypes > 0:
                            subtype_dist = self.edge_subtype_by_family.get(fam_name)
                            if subtype_dist is None:
                                subtype_dist = torch.ones(num_subtypes, dtype=torch.float) / num_subtypes
                            else:
                                if not isinstance(subtype_dist, torch.Tensor):
                                    subtype_dist = torch.tensor(subtype_dist, dtype=torch.float)
                                if subtype_dist.numel() != num_subtypes:
                                    subtype_dist = torch.ones(num_subtypes, dtype=torch.float) / num_subtypes
                                else:
                                    if subtype_dist.sum() > 0:
                                        subtype_dist = subtype_dist / subtype_dist.sum()
                                    else:
                                        subtype_dist = torch.ones(num_subtypes, dtype=torch.float) / num_subtypes
                            fam_marginals = torch.zeros(num_subtypes + 1, dtype=torch.float)
                            fam_marginals[0] = u0
                            fam_marginals[1:] = u1 * subtype_dist
                        else:
                            fam_marginals = torch.tensor([u0, u1], dtype=torch.float)

                        self.edge_family_marginals[fam_name] = fam_marginals
                        print(
                            f"[DEBUG] ACMSubgraphsInfos.__init__: fam={fam_name} "
                            f"m_fam={m_fam:.3f}, W_fam={w_fam:.3f}, u1={u1:.6f}"
                        )
                    else:
                        # 回退：使用 bond_types 中的全局分布
                        if num_subtypes > 0:
                            fam_marginals = torch.zeros(num_subtypes + 1)  # +1 for no-edge
                            fam_marginals[0] = self.bond_types[0] if len(self.bond_types) > 0 else 0.0
                            for i, lbl in enumerate(fam_labels, start=1):
                                gid = edge_label2id[lbl]
                                if gid < len(self.bond_types):
                                    fam_marginals[i] = self.bond_types[gid]
                            if fam_marginals.sum() > 0:
                                fam_marginals = fam_marginals / fam_marginals.sum()
                            else:
                                fam_marginals = torch.ones(num_subtypes + 1) / (num_subtypes + 1)
                            self.edge_family_marginals[fam_name] = fam_marginals
                        else:
                            fam_marginals = torch.zeros(2)  # [no-edge, exists]
                            fam_marginals[0] = self.bond_types[0] if len(self.bond_types) > 0 else 0.5
                            if offset < len(self.bond_types):
                                fam_marginals[1] = self.bond_types[offset]
                            else:
                                fam_marginals[1] = 0.5
                            if fam_marginals.sum() > 0:
                                fam_marginals = fam_marginals / fam_marginals.sum()
                            else:
                                fam_marginals = torch.ones(2) / 2.0
                            self.edge_family_marginals[fam_name] = fam_marginals
                
            else:
                self.node_type_names = []
                self.node_type2id = {}
                self.type_offsets = {}
                self.node_subtype_decoder = {}
                self.edge_family2id = {}
                self.edge_family_offsets = {}
                self.edge_label2id = {}
                self.edge_family_marginals = {}
                self.edge_family_avg_edge_counts = {}
    
    def to_one_hot(self, data):
        """
        重写to_one_hot方法，使用output_dims.X和output_dims.E进行one-hot编码
        这样可以确保编码后的维度与模型期望的维度一致
        """
        one_hot_data = data.clone()
        one_hot_data.x = F.one_hot(data.x, num_classes=self.output_dims.X).float()
        one_hot_data.edge_attr = F.one_hot(data.edge_attr, num_classes=self.output_dims.E).float()
        
        if not self.use_charge:
            one_hot_data.charge = data.x.new_zeros((*data.x.shape[:-1], 0))
        else:
            one_hot_data.charge = F.one_hot(data.charge + 1, num_classes=self.num_charge_types).float()
        
        return one_hot_data