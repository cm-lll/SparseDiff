"""
异质图工具函数：提取元数据
"""
import torch
from typing import Optional, Dict, Tuple, List


def extract_node_metadata(
    node_t: torch.Tensor,  # (N, num_node_subtypes) one-hot 或 (N,) discrete
    type_offsets: Dict[str, int],  # 节点类型offset映射
    node_type_names: list,  # 节点类型名称列表
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从node_t中提取节点类别ID和子类别ID
    
    Args:
        node_t: 节点特征，one-hot编码或离散ID
        type_offsets: 节点类型offset映射，例如 {"Author": 0, "Paper": 4, "Organization": 8}
        node_type_names: 节点类型名称列表，例如 ["Author", "Paper", "Organization"]
    
    Returns:
        node_type_ids: (N,) 节点类别ID，例如 [0, 0, 1, 1, 2, ...]
        node_subtype_ids: (N,) 节点子类别ID（全局），例如 [0, 1, 4, 5, 8, ...]
    """
    # 转换为离散ID
    if node_t.dim() > 1:
        node_subtype_ids = node_t.argmax(dim=-1)  # (N,) 全局子类别ID
    else:
        node_subtype_ids = node_t.long()  # (N,) 已经是离散ID
    
    # 根据type_offsets确定每个节点属于哪个类别
    node_type_ids = torch.zeros_like(node_subtype_ids)  # (N,)
    
    # 按offset范围确定类别
    sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
    for i, (type_name, offset) in enumerate(sorted_types):
        if i + 1 < len(sorted_types):
            next_offset = sorted_types[i + 1][1]
            mask = (node_subtype_ids >= offset) & (node_subtype_ids < next_offset)
        else:
            # 最后一个类型
            mask = node_subtype_ids >= offset
        node_type_ids[mask] = i
    
    return node_type_ids, node_subtype_ids


def compute_relation_type_ids(
    edge_index: torch.Tensor,  # (2, E)
    node_type_ids: torch.Tensor,  # (N,) 节点类别ID
    num_node_types: int,
) -> torch.Tensor:
    """
    计算关系类型ID：从源节点类型和目标节点类型组合
    
    Args:
        edge_index: 边索引 (2, E)
        node_type_ids: 节点类别ID (N,)
        num_node_types: 节点类型数量
    
    Returns:
        relation_type_ids: (E,) 关系类型ID
        计算方式: relation_id = src_type * num_node_types + dst_type
    """
    src_nodes = edge_index[0]  # (E,)
    dst_nodes = edge_index[1]  # (E,)
    
    src_types = node_type_ids[src_nodes]  # (E,)
    dst_types = node_type_ids[dst_nodes]  # (E,)
    
    # 关系类型ID = src_type * num_node_types + dst_type
    relation_type_ids = src_types * num_node_types + dst_types  # (E,)
    
    return relation_type_ids


def extract_edge_family_ids(
    edge_attr: torch.Tensor,  # (E, num_edge_types) one-hot 或 (E,) discrete
    edge_family_offsets: Dict[str, int],  # 关系族offset映射
    edge_index: Optional[torch.Tensor] = None,  # (2, E)
    node_type_ids: Optional[torch.Tensor] = None,  # (N,)
    fam_endpoints: Optional[Dict[str, Dict[str, str]]] = None,
    node_type_names_by_offset: Optional[List[str]] = None,
    unknown_family_id: int = -1,
) -> torch.Tensor:
    """
    从edge_attr中提取关系族ID
    
    Args:
        edge_attr: 边特征，one-hot编码或离散ID
        edge_family_offsets: 关系族offset映射，例如 {"author_of": 0, "cite": 4, "belong_to": 8}
    
    Returns:
        edge_family_ids: (E,) 关系族ID。对于 no-edge：
            - 若可由 (src_type, dst_type) 唯一映射到 family，则使用该 family；
            - 否则尝试 (dst_type, src_type)：to_undirected 引入的反向边归属到其正向关系族
              （如「论文-作者」→ author_of，「论文-论文」→ cite，二者不会混为一谈）；
            - 否则赋值为 unknown_family_id（默认 -1）。
    """
    # 转换为离散ID
    if edge_attr.dim() > 1:
        edge_type_ids = edge_attr.argmax(dim=-1)  # (E,) 全局边类型ID
    else:
        edge_type_ids = edge_attr.long()  # (E,) 已经是离散ID
    
    # 先全部设为 unknown，避免 no-edge 被默认归到 family 0
    edge_family_ids = torch.full_like(edge_type_ids, fill_value=unknown_family_id)  # (E,)
    
    # 按offset范围确定关系族
    sorted_families = sorted(edge_family_offsets.items(), key=lambda x: x[1])
    for i, (fam_name, offset) in enumerate(sorted_families):
        if i + 1 < len(sorted_families):
            next_offset = sorted_families[i + 1][1]
            mask = (edge_type_ids >= offset) & (edge_type_ids < next_offset)
        else:
            # 最后一个关系族
            mask = edge_type_ids >= offset
        edge_family_ids[mask] = i

    # no-edge（0）不属于固定 family：按端点类型推断它应属于哪个 family
    can_infer_no_edge_family = (
        edge_index is not None
        and node_type_ids is not None
        and fam_endpoints is not None
        and node_type_names_by_offset is not None
    )
    if can_infer_no_edge_family:
        # type_name -> type_id（与 extract_node_metadata 的 offset 顺序保持一致）
        type_name_to_id = {t_name: i for i, t_name in enumerate(node_type_names_by_offset)}
        pair_to_family_id = {}
        for fam_id, (fam_name, _) in enumerate(sorted_families):
            endpoint = fam_endpoints.get(fam_name, None)
            if not endpoint:
                continue
            src_type = endpoint.get("src_type")
            dst_type = endpoint.get("dst_type")
            if src_type not in type_name_to_id or dst_type not in type_name_to_id:
                continue
            pair_to_family_id[(type_name_to_id[src_type], type_name_to_id[dst_type])] = fam_id

        no_edge_mask = edge_type_ids == 0
        if no_edge_mask.any():
            src_nodes = edge_index[0, no_edge_mask]
            dst_nodes = edge_index[1, no_edge_mask]
            src_types = node_type_ids[src_nodes]
            dst_types = node_type_ids[dst_nodes]
            no_edge_indices = torch.where(no_edge_mask)[0]

            for i, edge_pos in enumerate(no_edge_indices.tolist()):
                s, d = int(src_types[i].item()), int(dst_types[i].item())
                # 先试 (src_type, dst_type)，再试 (dst_type, src_type)
                # 消息传递时 to_undirected 会引入反向边，(dst, src) 可能不在 fam_endpoints，
                # 但对应同一关系族，用反向对即可映射到正确 family
                fam_id = pair_to_family_id.get((s, d))
                if fam_id is None:
                    fam_id = pair_to_family_id.get((d, s))
                if fam_id is not None:
                    edge_family_ids[edge_pos] = fam_id
    
    return edge_family_ids


def extract_heterogeneous_metadata(
    node_t: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_index: torch.Tensor,
    type_offsets: Optional[Dict[str, int]] = None,
    node_type_names: Optional[list] = None,
    edge_family_offsets: Optional[Dict[str, int]] = None,
    fam_endpoints: Optional[Dict[str, Dict[str, str]]] = None,
    num_node_types: int = 0,
    num_edge_classes: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    提取所有异质图元数据
    
    Args:
        num_edge_classes: 离散边类型维度。若提供且 edge_attr 的最后一维 > num_edge_classes，
            则只用 edge_attr[:, :num_edge_classes] 提取 edge_family_ids，避免 extra 列参与
            argmax 导致关系族错误（compute_extra_data 会把 [one-hot 边类型, extraE] 拼在一起）。
    
    Returns:
        metadata: 包含以下键的字典：
            - node_type_ids: (N,) 节点类别ID
            - node_subtype_ids: (N,) 节点子类别ID（全局）
            - relation_type_ids: (E,) 关系类型ID
            - edge_family_ids: (E,) 关系族ID
    """
    metadata = {}
    
    # 提取节点元数据
    if type_offsets is not None and node_type_names is not None:
        node_type_ids, node_subtype_ids = extract_node_metadata(
            node_t, type_offsets, node_type_names
        )
        metadata["node_type_ids"] = node_type_ids
        metadata["node_subtype_ids"] = node_subtype_ids
        
        # 计算关系类型ID
        if num_node_types > 0:
            relation_type_ids = compute_relation_type_ids(
                edge_index, node_type_ids, num_node_types
            )
            metadata["relation_type_ids"] = relation_type_ids
    
    # 提取边元数据（关系族）
    # edge_attr 在 compute_extra_data 之后可能是 [one-hot 边类型 | extraE]，
    # 只有 one-hot 那一段的 argmax 才是边类型 ID；整段 argmax 会混入 extra，导致族错。
    if edge_family_offsets is not None:
        edge_attr_for_family = edge_attr
        if num_edge_classes is not None and edge_attr.dim() > 1 and edge_attr.shape[-1] > num_edge_classes:
            edge_attr_for_family = edge_attr[:, :num_edge_classes]
        node_type_names_by_offset = None
        if type_offsets is not None:
            node_type_names_by_offset = [t_name for t_name, _ in sorted(type_offsets.items(), key=lambda x: x[1])]
        edge_family_ids = extract_edge_family_ids(
            edge_attr=edge_attr_for_family,
            edge_family_offsets=edge_family_offsets,
            edge_index=edge_index,
            node_type_ids=metadata.get("node_type_ids"),
            fam_endpoints=fam_endpoints,
            node_type_names_by_offset=node_type_names_by_offset,
        )
        metadata["edge_family_ids"] = edge_family_ids
    
    return metadata
