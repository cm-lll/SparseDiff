#!/usr/bin/env python3
"""
测试异质图噪声模型（关系族隔离的边扩散）
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')

from sparse_diffusion.datasets.acm_subgraphs_dataset import ACMSubgraphsDataset
from sparse_diffusion.datasets.dataset_utils import RemoveYTransform

def test_edge_family_info():
    """测试 edge_family 信息"""
    print("=" * 80)
    print("测试 1: edge_family 信息")
    print("=" * 80)
    
    root_path = "/data2/lyh/gnn_project/data/ACM_subgraphs"
    dataset = ACMSubgraphsDataset(
        split="train",
        root=root_path,
        pre_transform=RemoveYTransform(),
        heterogeneous=True
    )
    
    if len(dataset) == 0:
        print("⚠️  数据集为空")
        return None
    
    data = dataset[0]
    
    print(f"数据属性:")
    print(f"  节点数: {data.x.shape[0]}")
    print(f"  边数: {data.edge_index.shape[1]}")
    print(f"  edge_attr 范围: [{data.edge_attr.min().item()}, {data.edge_attr.max().item()}]")
    
    if hasattr(data, 'edge_family') and data.edge_family is not None:
        print(f"  edge_family 范围: [{data.edge_family.min().item()}, {data.edge_family.max().item()}]")
        
        # 统计每个关系族的边数
        unique_families, counts = torch.unique(data.edge_family, return_counts=True)
        print(f"\n每个关系族的边数:")
        for fam_id, count in zip(unique_families, counts):
            print(f"  关系族 {fam_id.item()}: {count.item()} 条边")
        
        # 验证每个关系族的边属性 ID 范围
        print(f"\n验证每个关系族的边属性 ID 范围:")
        # 需要从 vocab.json 加载关系族信息
        import json
        import os.path as osp
        vocab_path = osp.join(dataset.processed_dir, "vocab.json")
        if osp.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            edge_family2id = {k: int(v) for k, v in vocab.get("edge_family2id", {}).items()}
            edge_family_offsets = {k: int(v) for k, v in vocab.get("edge_family_offsets", {}).items()}
            edge_label2id = vocab.get("edge_label2id", {})
            
            # 计算每个关系族的子类别数量
            for fam_name, fam_id in edge_family2id.items():
                offset = edge_family_offsets[fam_name]
                # 计算该关系族的子类别数量
                fam_labels = [lbl for lbl, gid in edge_label2id.items() 
                             if lbl.startswith(fam_name + ":")]
                num_subtypes = len(fam_labels)
                
                fam_mask = (data.edge_family == fam_id)
                if fam_mask.any():
                    fam_edge_attrs = data.edge_attr[fam_mask]
                    expected_range = (offset, offset + num_subtypes - 1)
                    actual_range = (fam_edge_attrs.min().item(), fam_edge_attrs.max().item())
                    print(f"  {fam_name:20s} (fam_id={fam_id}, offset={offset:2d}): "
                          f"期望范围 [{expected_range[0]:2d}, {expected_range[1]:2d}], "
                          f"实际范围 [{actual_range[0]:2d}, {actual_range[1]:2d}] "
                          f"{'✅' if expected_range[0] <= actual_range[0] and actual_range[1] <= expected_range[1] else '❌'}")
    else:
        print("  ⚠️  edge_family 不存在")
    
    return data


def test_noise_application_structure():
    """测试噪声应用的结构（不实际应用噪声，只检查数据结构）"""
    print("\n" + "=" * 80)
    print("测试 2: 噪声应用结构分析")
    print("=" * 80)
    
    root_path = "/data2/lyh/gnn_project/data/ACM_subgraphs"
    dataset = ACMSubgraphsDataset(
        split="train",
        root=root_path,
        pre_transform=RemoveYTransform(),
        heterogeneous=True
    )
    
    if len(dataset) == 0:
        print("⚠️  数据集为空")
        return
    
    data = dataset[0]
    
    if not (hasattr(data, 'edge_family') and data.edge_family is not None):
        print("⚠️  edge_family 不存在，无法测试")
        return
    
    # 加载关系族信息
    import json
    import os.path as osp
    vocab_path = osp.join(dataset.processed_dir, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    edge_family2id = {k: int(v) for k, v in vocab.get("edge_family2id", {}).items()}
    edge_family_offsets = {k: int(v) for k, v in vocab.get("edge_family_offsets", {}).items()}
    edge_label2id = vocab.get("edge_label2id", {})
    
    print(f"关系族信息:")
    for fam_name, fam_id in edge_family2id.items():
        offset = edge_family_offsets[fam_name]
        fam_labels = [lbl for lbl, gid in edge_label2id.items() 
                     if lbl.startswith(fam_name + ":")]
        num_subtypes = len(fam_labels)
        print(f"  {fam_name:20s}: fam_id={fam_id:2d}, offset={offset:2d}, 子类别数={num_subtypes}")
    
    # 分析边的结构
    print(f"\n边的结构分析:")
    print(f"  总边数: {data.edge_index.shape[1]}")
    
    # 对于每个关系族，计算：
    # 1. 已存在的边数
    # 2. 可能的非存在边数（需要考虑端点类型）
    for fam_name, fam_id in edge_family2id.items():
        fam_mask = (data.edge_family == fam_id)
        num_existing_edges = fam_mask.sum().item()
        print(f"  {fam_name:20s}: 已存在边数 = {num_existing_edges}")
        
        # 计算可能的非存在边数（需要考虑端点类型）
        # 这需要从 meta.json 获取端点类型信息
        # 暂时跳过，后续实现
    
    print(f"\n✅ 噪声应用结构分析完成")
    print(f"\n关键点:")
    print(f"  1. 每个关系族需要独立的转移矩阵")
    print(f"  2. 每个关系族需要独立的伯努利采样（emerge_prob）")
    print(f"  3. 每个关系族的非存在边数需要根据端点类型计算")


if __name__ == "__main__":
    print("开始测试异质图噪声模型...\n")
    
    try:
        # 测试 1: edge_family 信息
        data = test_edge_family_info()
        
        # 测试 2: 噪声应用结构分析
        test_noise_application_structure()
        
        print("\n" + "=" * 80)
        print("✅ 测试完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
