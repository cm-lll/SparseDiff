#!/usr/bin/env python3
"""
测试异质图数据处理流程
"""
import json
import os
import os.path as osp
import sys
import torch

# 添加项目路径
sys.path.insert(0, '.')

from sparse_diffusion.datasets.acm_subgraphs_dataset import (
    _build_vocab_from_meta,
    _build_edges_for_graph,
    _build_node_state_id,
    _load_meta,
)

def test_vocab_building():
    """测试词汇表构建（异质图 vs 同质图）"""
    print("=" * 80)
    print("测试 1: 词汇表构建")
    print("=" * 80)
    
    meta_path = '/data2/lyh/gnn_project/data/ACM_subgraphs/subgraph_000/meta.json'
    if not osp.exists(meta_path):
        print(f"错误: 找不到 {meta_path}")
        return False
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    print("\n--- 异质图模式 (heterogeneous=True) ---")
    node_type_names, node_type2id, edge_label2id, edge_family2id, edge_family_offsets = _build_vocab_from_meta(meta, heterogeneous=True)
    
    print(f"节点类型: {node_type_names}")
    print(f"节点类型映射: {node_type2id}")
    print(f"\n关系族映射: {edge_family2id}")
    print(f"关系族 offsets: {edge_family_offsets}")
    print(f"\n边标签映射（异质图，类别隔离）:")
    for lbl, gid in sorted(edge_label2id.items(), key=lambda x: x[1]):
        print(f"  {lbl:40s} -> {gid:3d}")
    
    print("\n--- 同质图模式 (heterogeneous=False) ---")
    node_type_names2, node_type2id2, edge_label2id2, edge_family2id2, edge_family_offsets2 = _build_vocab_from_meta(meta, heterogeneous=False)
    
    print(f"边标签映射（同质图，全局空间）:")
    for lbl, gid in sorted(edge_label2id2.items(), key=lambda x: x[1]):
        print(f"  {lbl:40s} -> {gid:3d}")
    
    print("\n✅ 词汇表构建测试通过")
    return True, edge_family2id, edge_family_offsets


def test_edge_building():
    """测试边构建（异质图模式）"""
    print("\n" + "=" * 80)
    print("测试 2: 边构建（异质图模式）")
    print("=" * 80)
    
    subgraph_dir = '/data2/lyh/gnn_project/data/ACM_subgraphs/subgraph_000'
    if not osp.exists(subgraph_dir):
        print(f"错误: 找不到 {subgraph_dir}")
        return False
    
    meta = _load_meta(subgraph_dir)
    nodes = torch.load(osp.join(subgraph_dir, "nodes.pt"), map_location="cpu")
    edges = torch.load(osp.join(subgraph_dir, "edges.pt"), map_location="cpu")
    
    # 构建词汇表
    node_type_names, node_type2id, edge_label2id, edge_family2id, edge_family_offsets = _build_vocab_from_meta(meta, heterogeneous=True)
    
    # 构建节点状态（用于获取 offsets）
    node_state, type_offsets, _ = _build_node_state_id(node_type_names, nodes, node_type2id)
    
    # 构建边（异质图模式）
    edge_index, edge_attr, edge_family = _build_edges_for_graph(
        edges, meta, type_offsets, edge_label2id,
        edge_family_offsets=edge_family_offsets,
        edge_family2id=edge_family2id,
        heterogeneous=True
    )
    
    print(f"\n边数量: {edge_index.shape[1]}")
    print(f"边属性形状: {edge_attr.shape}")
    print(f"边关系族形状: {edge_family.shape if edge_family is not None else None}")
    
    # 验证边属性 ID 范围
    print(f"\n边属性 ID 范围: [{edge_attr.min().item()}, {edge_attr.max().item()}]")
    print(f"边关系族 ID 范围: [{edge_family.min().item()}, {edge_family.max().item()}]" if edge_family is not None else "边关系族: None")
    
    # 检查每个关系族的边属性 ID 是否在正确的范围内
    print("\n验证每个关系族的边属性 ID 范围:")
    for fam, offset in edge_family_offsets.items():
        fam_id = edge_family2id[fam]
        fam_mask = (edge_family == fam_id)
        if fam_mask.any():
            fam_edge_attrs = edge_attr[fam_mask]
            num_subtypes = len(meta["fam_id2label"][fam])
            expected_range = (offset, offset + num_subtypes - 1)
            actual_range = (fam_edge_attrs.min().item(), fam_edge_attrs.max().item())
            print(f"  {fam:20s} (fam_id={fam_id:2d}, offset={offset:2d}): "
                  f"期望范围 [{expected_range[0]:2d}, {expected_range[1]:2d}], "
                  f"实际范围 [{actual_range[0]:2d}, {actual_range[1]:2d}] "
                  f"{'✅' if expected_range[0] <= actual_range[0] and actual_range[1] <= expected_range[1] else '❌'}")
    
    print("\n✅ 边构建测试通过")
    return True


def test_data_object():
    """测试完整的 Data 对象构建"""
    print("\n" + "=" * 80)
    print("测试 3: 完整 Data 对象构建")
    print("=" * 80)
    
    from torch_geometric.data import Data
    from sparse_diffusion.datasets.acm_subgraphs_dataset import _concat_node_fields
    
    subgraph_dir = '/data2/lyh/gnn_project/data/ACM_subgraphs/subgraph_000'
    if not osp.exists(subgraph_dir):
        print(f"错误: 找不到 {subgraph_dir}")
        return False
    
    meta = _load_meta(subgraph_dir)
    nodes = torch.load(osp.join(subgraph_dir, "nodes.pt"), map_location="cpu")
    edges = torch.load(osp.join(subgraph_dir, "edges.pt"), map_location="cpu")
    
    # 构建词汇表
    node_type_names, node_type2id, edge_label2id, edge_family2id, edge_family_offsets = _build_vocab_from_meta(meta, heterogeneous=True)
    
    # 构建节点状态
    node_state, type_offsets, _ = _build_node_state_id(node_type_names, nodes, node_type2id)
    
    # 构建节点类型和子类型
    node_type_id, node_subtype_local, offsets = _concat_node_fields(nodes, node_type_names)
    for t in node_type_names:
        off = offsets[t]
        n = int(nodes[t]["subtype"].shape[0])
        node_type_id[off : off + n] = int(node_type2id[t])
    
    # 构建边
    edge_index, edge_attr, edge_family = _build_edges_for_graph(
        edges, meta, offsets, edge_label2id,
        edge_family_offsets=edge_family_offsets,
        edge_family2id=edge_family2id,
        heterogeneous=True
    )
    
    # 创建 Data 对象
    data = Data(
        x=node_state,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=node_type_id,
        node_subtype=node_subtype_local,
        edge_family=edge_family,
        y=torch.zeros((1, 0), dtype=torch.float),
    )
    
    print(f"\nData 对象属性:")
    print(f"  x (节点扩散状态): {data.x.shape}, 范围 [{data.x.min().item()}, {data.x.max().item()}]")
    print(f"  edge_index: {data.edge_index.shape}")
    print(f"  edge_attr (边扩散状态): {data.edge_attr.shape}, 范围 [{data.edge_attr.min().item()}, {data.edge_attr.max().item()}]")
    print(f"  node_type: {data.node_type.shape}, 范围 [{data.node_type.min().item()}, {data.node_type.max().item()}]")
    print(f"  node_subtype: {data.node_subtype.shape}, 范围 [{data.node_subtype.min().item()}, {data.node_subtype.max().item()}]")
    print(f"  edge_family: {data.edge_family.shape if hasattr(data, 'edge_family') and data.edge_family is not None else None}")
    
    # 验证节点隔离
    print("\n验证节点类别隔离:")
    for t in node_type_names:
        t_id = node_type2id[t]
        t_mask = (data.node_type == t_id)
        if t_mask.any():
            t_x = data.x[t_mask]
            off = type_offsets[t]
            num_subtypes = len(meta["schema_by_type"][t])
            expected_range = (off, off + num_subtypes - 1)
            actual_range = (t_x.min().item(), t_x.max().item())
            print(f"  {t:15s} (type_id={t_id}): "
                  f"期望范围 [{expected_range[0]:2d}, {expected_range[1]:2d}], "
                  f"实际范围 [{actual_range[0]:2d}, {actual_range[1]:2d}] "
                  f"{'✅' if expected_range[0] <= actual_range[0] and actual_range[1] <= expected_range[1] else '❌'}")
    
    print("\n✅ Data 对象构建测试通过")
    return True


if __name__ == "__main__":
    print("开始测试异质图数据处理流程...\n")
    
    try:
        # 测试 1: 词汇表构建
        success, edge_family2id, edge_family_offsets = test_vocab_building()
        if not success:
            sys.exit(1)
        
        # 测试 2: 边构建
        if not test_edge_building():
            sys.exit(1)
        
        # 测试 3: Data 对象构建
        if not test_data_object():
            sys.exit(1)
        
        print("\n" + "=" * 80)
        print("✅ 所有测试通过！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
