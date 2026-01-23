import json
import os
import os.path as osp
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import torch
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
                local_id = int(k) - 1  # meta.json 中 id 从 1 开始，转换为从 0 开始
                global_id = offset + local_id
                edge_label2id[lbl] = global_id
        
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
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            node_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
            charge_types=None,
            valencies=None,
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
        # 如果是异质图，添加关系族平均边数文件
        if self.heterogeneous:
            files.append(f"{self.split}_edge_family_avg_counts.pickle")
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

        n_test = int(round(n_graphs * test_ratio))
        n_val = int(round(n_graphs * val_ratio))
        n_train = n_graphs - n_val - n_test

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
        bond_types = edge_counts(data_list, num_bond_types=num_edge_types)
        
        # 计算每个关系族的平均真实边数（用于采样时的 |Eq| = km 计算）
        edge_family_edge_counts = {}
        if self.heterogeneous and len(edge_family2id) > 0:
            for data in data_list:
                if hasattr(data, 'edge_family') and data.edge_family is not None:
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
            
            # 计算每个关系族的平均边数
            edge_family_avg_edge_counts = {}
            for fam_name, counts in edge_family_edge_counts.items():
                if len(counts) > 0:
                    edge_family_avg_edge_counts[fam_name] = sum(counts) / len(counts)
                else:
                    edge_family_avg_edge_counts[fam_name] = 0.0
        else:
            edge_family_avg_edge_counts = {}

        torch.save(self.collate(data_list), self.processed_paths[0])
        save_pickle(num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], node_types)
        np.save(self.processed_paths[3], bond_types)
        
        # 保存每个关系族的平均真实边数
        if edge_family_avg_edge_counts:
            edge_family_counts_path = osp.join(self.processed_dir, f"{self.split}_edge_family_avg_counts.pickle")
            save_pickle(edge_family_avg_edge_counts, edge_family_counts_path)


class ACMSubgraphsDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset.name
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
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

        self.input_dims = PlaceHolder(
            X=len(self.node_types), E=len(self.bond_types), y=0, charge=0
        )
        self.output_dims = PlaceHolder(
            X=len(self.node_types), E=len(self.bond_types), y=0, charge=0
        )
        self.statistics = {
            "train": datamodule.statistics["train"],
            "val": datamodule.statistics["val"],
            "test": datamodule.statistics["test"],
        }
        
        # 存储关系族信息（用于异质图边扩散）
        self.heterogeneous = getattr(datamodule.cfg.dataset, "heterogeneous", True)
        if self.heterogeneous:
            # 从 dataset 加载关系族信息
            vocab_path = osp.join(datamodule.inner.processed_dir, "vocab.json")
            if osp.exists(vocab_path):
                import json
                with open(vocab_path, "r", encoding="utf-8") as f:
                    vocab = json.load(f)
                self.edge_family2id = {k: int(v) for k, v in vocab.get("edge_family2id", {}).items()}
                self.edge_family_offsets = {k: int(v) for k, v in vocab.get("edge_family_offsets", {}).items()}
                edge_label2id = vocab.get("edge_label2id", {})
                
                # 存储端点类型信息（用于计算非存在边数）
                # 从第一个子图的 meta.json 加载
                subgraph_dirs = _list_subgraph_dirs(datamodule.inner.root)
                if len(subgraph_dirs) > 0:
                    meta0 = _load_meta(subgraph_dirs[0])
                    self.fam_endpoints = meta0.get("fam_endpoints", {})
                else:
                    self.fam_endpoints = {}
                
                # 计算每个关系族的边类型分布
                self.edge_family_marginals = {}
                for fam_name, fam_id in self.edge_family2id.items():
                    offset = self.edge_family_offsets[fam_name]
                    # 找到该关系族的所有标签
                    fam_labels = [lbl for lbl, gid in edge_label2id.items() 
                                 if lbl.startswith(fam_name + ":")]
                    num_subtypes = len(fam_labels)
                    
                    # 从 bond_types 中提取该关系族的分布
                    # bond_types 的索引对应全局 edge_label ID（包括 no-edge=0）
                    # 需要映射到关系族内的局部 ID
                    if num_subtypes > 0:
                        # 有子类别：提取该关系族的边类型分布
                        fam_marginals = torch.zeros(num_subtypes + 1)  # +1 for no-edge
                        # no-edge 的分布（索引 0）在所有关系族中共享
                        fam_marginals[0] = self.bond_types[0] if len(self.bond_types) > 0 else 0.0
                        # 子类别的分布
                        for i, lbl in enumerate(fam_labels, start=1):
                            gid = edge_label2id[lbl]
                            if gid < len(self.bond_types):
                                fam_marginals[i] = self.bond_types[gid]
                        
                        # 归一化（如果总和为0，使用均匀分布）
                        if fam_marginals.sum() > 0:
                            fam_marginals = fam_marginals / fam_marginals.sum()
                        else:
                            fam_marginals = torch.ones(num_subtypes + 1) / (num_subtypes + 1)
                        
                        self.edge_family_marginals[fam_name] = fam_marginals
                    else:
                        # 没有子类别：只有存在性（no-edge 和单一类别）
                        # 为了统一接口，仍然计算分布，但只有两个状态：[no-edge, exists]
                        # 从 bond_types 中提取该关系族的单一类别分布
                        fam_marginals = torch.zeros(2)  # [no-edge, exists]
                        fam_marginals[0] = self.bond_types[0] if len(self.bond_types) > 0 else 0.5
                        # 提取该关系族的单一类别（offset 对应的全局 ID）
                        if offset < len(self.bond_types):
                            fam_marginals[1] = self.bond_types[offset]
                        else:
                            fam_marginals[1] = 0.5
                        
                        # 归一化
                        if fam_marginals.sum() > 0:
                            fam_marginals = fam_marginals / fam_marginals.sum()
                        else:
                            fam_marginals = torch.ones(2) / 2.0
                        
                        self.edge_family_marginals[fam_name] = fam_marginals
                
                # 加载每个关系族的平均真实边数（用于采样时的 |Eq| = km 计算）
                edge_family_counts_path = osp.join(datamodule.inner.processed_dir, "train_edge_family_avg_counts.pickle")
                if osp.exists(edge_family_counts_path):
                    self.edge_family_avg_edge_counts = load_pickle(edge_family_counts_path)
                else:
                    # 如果没有保存的文件，使用默认值（从 bond_types 估算）
                    self.edge_family_avg_edge_counts = {}
                    for fam_name in self.edge_family2id.keys():
                        # 估算：使用一个简单的启发式值
                        # 实际应该从训练数据中统计，但如果没有保存，使用这个估算值
                        self.edge_family_avg_edge_counts[fam_name] = 10.0  # 默认值，实际应该从数据中计算
            else:
                self.edge_family2id = {}
                self.edge_family_offsets = {}
                self.edge_family_marginals = {}
                self.edge_family_avg_edge_counts = {}
        else:
            self.edge_family2id = {}
            self.edge_family_offsets = {}
            self.edge_family_marginals = {}
            self.edge_family_avg_edge_counts = {}