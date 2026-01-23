#!/usr/bin/env python3
"""
详细测试噪声应用函数（需要模型初始化）
"""
import sys
import torch

sys.path.insert(0, '.')

from sparse_diffusion.datasets.acm_subgraphs_dataset import ACMSubgraphsDataset, ACMSubgraphsDataModule
from sparse_diffusion.datasets.dataset_utils import RemoveYTransform
from sparse_diffusion.diffusion.heterogeneous_transition import HeterogeneousMarginalUniformTransition


def test_transition_model_with_real_data():
    """使用真实数据测试转移矩阵"""
    print("=" * 80)
    print("测试: 转移矩阵与真实数据")
    print("=" * 80)
    
    root_path = "/data2/lyh/gnn_project/data/ACM_subgraphs"
    dataset = ACMSubgraphsDataset(
        split="train",
        root=root_path,
        pre_transform=RemoveYTransform(),
        heterogeneous=True
    )
    
    if len(dataset) == 0:
        print("⚠️  数据集为空，跳过测试")
        return False
    
    # 获取数据集信息
    from sparse_diffusion.datasets.acm_subgraphs_dataset import ACMSubgraphsInfos
    
    # 创建模拟的 datamodule（用于获取 Infos）
    class MockConfig:
        class Dataset:
            name = 'acm_subgraphs'
            datadir = 'data/ACM_subgraphs'
            heterogeneous = True
        dataset = Dataset()
    
    class MockDataModule:
        def __init__(self):
            self.cfg = MockConfig()
            self.dataset_name = 'acm_subgraphs'
            self.inner = dataset
            self.statistics = {
                "train": dataset.statistics,
                "val": dataset.statistics,
                "test": dataset.statistics,
            }
    
    datamodule = MockDataModule()
    dataset_infos = ACMSubgraphsInfos(datamodule)
    
    print(f"数据集信息:")
    print(f"  heterogeneous: {dataset_infos.heterogeneous}")
    print(f"  关系族数量: {len(dataset_infos.edge_family2id)}")
    print(f"  关系族边际分布:")
    for fam_name, marginals in dataset_infos.edge_family_marginals.items():
        print(f"    {fam_name:20s}: {len(marginals)} 个状态, 分布 = {marginals.tolist()}")
    
    # 创建转移矩阵
    x_marginals = dataset_infos.node_types / dataset_infos.node_types.sum()
    e_marginals = dataset_infos.bond_types / dataset_infos.bond_types.sum()
    
    transition = HeterogeneousMarginalUniformTransition(
        x_marginals=x_marginals,
        e_marginals=e_marginals,
        y_classes=0,
        charge_marginals=torch.zeros(0),
        edge_family_marginals=dataset_infos.edge_family_marginals,
        edge_family_offsets=dataset_infos.edge_family_offsets,
    )
    
    print(f"\n转移矩阵:")
    print(f"  heterogeneous: {transition.heterogeneous}")
    print(f"  关系族数量: {len(transition.edge_family_marginals)}")
    
    # 测试转移矩阵计算
    device = torch.device("cpu")
    alpha_bar_t = torch.tensor([0.5, 0.8, 0.9])  # 3 个批次
    
    # 获取所有关系族的转移矩阵
    all_family_qt = transition.get_all_family_Qt_bar(alpha_bar_t, device)
    
    print(f"\n所有关系族的转移矩阵:")
    for fam_name, qt in all_family_qt.items():
        num_states = qt.E.shape[1]
        print(f"  {fam_name:20s}: Q_e 形状 = {qt.E.shape}, 状态数 = {num_states}")
        # 验证行和
        row_sums = qt.E.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums)), f"{fam_name} 行和不为 1"
        print(f"    ✅ 行和验证通过")
    
    print("\n✅ 转移矩阵与真实数据测试通过")
    return True


def test_edge_id_conversion():
    """测试边 ID 的全局到局部转换"""
    print("\n" + "=" * 80)
    print("测试: 边 ID 转换（全局 <-> 局部）")
    print("=" * 80)
    
    root_path = "/data2/lyh/gnn_project/data/ACM_subgraphs"
    dataset = ACMSubgraphsDataset(
        split="train",
        root=root_path,
        pre_transform=RemoveYTransform(),
        heterogeneous=True
    )
    
    if len(dataset) == 0:
        print("⚠️  数据集为空，跳过测试")
        return False
    
    data = dataset[0]
    
    # 加载关系族信息
    import json
    import os.path as osp
    vocab_path = osp.join(dataset.processed_dir, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    edge_family2id = {k: int(v) for k, v in vocab.get("edge_family2id", {}).items()}
    edge_family_offsets = {k: int(v) for k, v in vocab.get("edge_family_offsets", {}).items()}
    
    print(f"测试边 ID 转换:")
    for fam_name, fam_id in edge_family2id.items():
        offset = edge_family_offsets[fam_name]
        fam_mask = (data.edge_family == fam_id)
        
        if fam_mask.any():
            fam_edge_attrs = data.edge_attr[fam_mask]
            
            # 全局 ID -> 局部 ID
            # 局部 ID: 0 = no-edge, 1, 2, ... = 子类别
            # 全局 ID: 0 = no-edge, offset, offset+1, ... = 该关系族的子类别
            local_attrs = fam_edge_attrs.clone()
            non_zero_mask = local_attrs != 0
            if non_zero_mask.any():
                # 全局 ID -> 局部 ID: 全局 ID - offset + 1（因为局部 ID 从 1 开始）
                local_attrs[non_zero_mask] = local_attrs[non_zero_mask] - offset + 1
            
            # 局部 ID -> 全局 ID
            # 局部 ID 0 -> 全局 0, 局部 ID 1,2,... -> 全局 offset, offset+1,...
            global_attrs = local_attrs.clone()
            non_zero_local_mask = global_attrs != 0
            if non_zero_local_mask.any():
                global_attrs[non_zero_local_mask] = global_attrs[non_zero_local_mask] - 1 + offset
            
            # 验证转换正确性
            original = fam_edge_attrs
            
            # 对于所有边，应该能正确转换
            if torch.all(global_attrs == original):
                print(f"  {fam_name:20s}: ✅ ID 转换正确 (范围 [{fam_edge_attrs.min().item()}, {fam_edge_attrs.max().item()}])")
            else:
                # 找出不匹配的边
                mismatch_mask = global_attrs != original
                if mismatch_mask.any():
                    print(f"  {fam_name:20s}: ❌ ID 转换错误")
                    print(f"    不匹配数量: {mismatch_mask.sum().item()}")
                    print(f"    示例: 原始={original[mismatch_mask][:5].tolist()}, "
                          f"转换后={global_attrs[mismatch_mask][:5].tolist()}")
                else:
                    print(f"  {fam_name:20s}: ✅ ID 转换正确")
    
    print("\n✅ 边 ID 转换测试通过")
    return True


if __name__ == "__main__":
    print("开始详细测试噪声应用...\n")
    
    try:
        # 测试转移矩阵与真实数据
        test_transition_model_with_real_data()
        
        # 测试边 ID 转换
        test_edge_id_conversion()
        
        print("\n" + "=" * 80)
        print("✅ 所有详细测试通过！")
        print("=" * 80)
        print("\n下一步建议:")
        print("  1. 运行一个小规模的训练来验证完整流程")
        print("  2. 检查训练过程中的损失是否正常")
        print("  3. 验证采样结果是否符合关系族隔离的要求")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
