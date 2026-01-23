#!/usr/bin/env python3
"""
测试异质图转移矩阵类
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')

from sparse_diffusion.diffusion.heterogeneous_transition import HeterogeneousMarginalUniformTransition
from sparse_diffusion.datasets.acm_subgraphs_dataset import ACMSubgraphsDataset, ACMSubgraphsDataModule
from sparse_diffusion.datasets.dataset_utils import RemoveYTransform


def test_transition_matrix_creation():
    """测试转移矩阵的创建"""
    print("=" * 80)
    print("测试 1: 转移矩阵创建")
    print("=" * 80)
    
    # 创建模拟数据
    x_marginals = torch.ones(14) / 14  # 14 个节点子类别
    e_marginals = torch.ones(6) / 6    # 6 个边类型（包括 no-edge）
    y_classes = 0
    charge_marginals = torch.zeros(0)
    
    # 创建关系族边际分布
    edge_family_marginals = {
        "affiliated_with": torch.tensor([0.5, 0.5]),  # [no-edge, exists] - 没有子类别
        "author_of": torch.tensor([0.25, 0.25, 0.25, 0.25]),  # [no-edge, first_author, second_author, co_author]
        "cites": torch.tensor([0.5, 0.5]),  # [no-edge, exists] - 没有子类别
    }
    edge_family_offsets = {
        "affiliated_with": 1,
        "author_of": 2,
        "cites": 5,
    }
    
    # 创建转移矩阵（异质图模式）
    transition = HeterogeneousMarginalUniformTransition(
        x_marginals=x_marginals,
        e_marginals=e_marginals,
        y_classes=y_classes,
        charge_marginals=charge_marginals,
        edge_family_marginals=edge_family_marginals,
        edge_family_offsets=edge_family_offsets,
    )
    
    print(f"转移矩阵属性:")
    print(f"  X_classes: {transition.X_classes}")
    print(f"  E_classes: {transition.E_classes}")
    print(f"  heterogeneous: {transition.heterogeneous}")
    print(f"  关系族数量: {len(transition.edge_family_marginals)}")
    
    print(f"\n关系族信息:")
    for fam_name, marginals in transition.edge_family_marginals.items():
        print(f"  {fam_name:20s}: {len(marginals)} 个状态, 分布 = {marginals.tolist()}")
    
    print("\n✅ 转移矩阵创建测试通过")
    return transition


def test_transition_matrix_computation():
    """测试转移矩阵的计算"""
    print("\n" + "=" * 80)
    print("测试 2: 转移矩阵计算")
    print("=" * 80)
    
    # 创建转移矩阵
    x_marginals = torch.ones(14) / 14
    e_marginals = torch.ones(6) / 6
    edge_family_marginals = {
        "affiliated_with": torch.tensor([0.5, 0.5]),
        "author_of": torch.tensor([0.25, 0.25, 0.25, 0.25]),
        "cites": torch.tensor([0.5, 0.5]),
    }
    edge_family_offsets = {
        "affiliated_with": 1,
        "author_of": 2,
        "cites": 5,
    }
    
    transition = HeterogeneousMarginalUniformTransition(
        x_marginals=x_marginals,
        e_marginals=e_marginals,
        y_classes=0,
        charge_marginals=torch.zeros(0),
        edge_family_marginals=edge_family_marginals,
        edge_family_offsets=edge_family_offsets,
    )
    
    device = torch.device("cpu")
    beta_t = torch.tensor([0.1, 0.5, 0.9])  # 3 个批次，不同的噪声水平
    alpha_bar_t = torch.tensor([0.9, 0.5, 0.1])  # 对应的 alpha_bar
    
    # 测试单步转移矩阵（全局）
    print("\n--- 单步转移矩阵 (get_Qt) ---")
    Qt_global = transition.get_Qt(beta_t, device)
    print(f"  Q_x 形状: {Qt_global.X.shape}")  # (3, 14, 14)
    print(f"  Q_e 形状: {Qt_global.E.shape}")  # (3, 6, 6)
    
    # 验证转移矩阵的行和是否为 1
    assert torch.allclose(Qt_global.X.sum(dim=-1), torch.ones(3, 14)), "Q_x 行和不为 1"
    assert torch.allclose(Qt_global.E.sum(dim=-1), torch.ones(3, 6)), "Q_e 行和不为 1"
    print("  ✅ 转移矩阵行和验证通过")
    
    # 测试关系族特定的转移矩阵
    print("\n--- 关系族特定的转移矩阵 ---")
    for fam_name in edge_family_marginals.keys():
        Qt_fam = transition.get_Qt(beta_t, device, edge_family_name=fam_name)
        num_states = len(edge_family_marginals[fam_name])
        print(f"  {fam_name:20s}: Q_e 形状 = {Qt_fam.E.shape}, 期望 = (3, {num_states}, {num_states})")
        assert Qt_fam.E.shape == (3, num_states, num_states), f"{fam_name} 转移矩阵形状错误"
        assert torch.allclose(Qt_fam.E.sum(dim=-1), torch.ones(3, num_states)), f"{fam_name} 行和不为 1"
    print("  ✅ 关系族特定转移矩阵验证通过")
    
    # 测试累积转移矩阵
    print("\n--- 累积转移矩阵 (get_Qt_bar) ---")
    Qtb_global = transition.get_Qt_bar(alpha_bar_t, device)
    print(f"  Q_x 形状: {Qtb_global.X.shape}")
    print(f"  Q_e 形状: {Qtb_global.E.shape}")
    assert torch.allclose(Qtb_global.X.sum(dim=-1), torch.ones(3, 14)), "Q_x 行和不为 1"
    assert torch.allclose(Qtb_global.E.sum(dim=-1), torch.ones(3, 6)), "Q_e 行和不为 1"
    print("  ✅ 累积转移矩阵验证通过")
    
    # 测试所有关系族的转移矩阵
    print("\n--- 所有关系族的转移矩阵 (get_all_family_Qt_bar) ---")
    all_family_qt = transition.get_all_family_Qt_bar(alpha_bar_t, device)
    print(f"  关系族数量: {len(all_family_qt)}")
    for fam_name, qt in all_family_qt.items():
        num_states = len(edge_family_marginals[fam_name])
        print(f"  {fam_name:20s}: Q_e 形状 = {qt.E.shape}, 期望 = (3, {num_states}, {num_states})")
        assert qt.E.shape == (3, num_states, num_states), f"{fam_name} 转移矩阵形状错误"
        assert torch.allclose(qt.E.sum(dim=-1), torch.ones(3, num_states)), f"{fam_name} 行和不为 1"
    print("  ✅ 所有关系族转移矩阵验证通过")
    
    print("\n✅ 转移矩阵计算测试通过")


def test_homogeneous_mode():
    """测试同质图模式（应该使用全局转移矩阵）"""
    print("\n" + "=" * 80)
    print("测试 3: 同质图模式（保留原逻辑）")
    print("=" * 80)
    
    x_marginals = torch.ones(14) / 14
    e_marginals = torch.ones(6) / 6
    
    # 同质图模式：不提供 edge_family_marginals
    transition = HeterogeneousMarginalUniformTransition(
        x_marginals=x_marginals,
        e_marginals=e_marginals,
        y_classes=0,
        charge_marginals=torch.zeros(0),
        edge_family_marginals=None,  # 同质图模式
        edge_family_offsets=None,
    )
    
    print(f"  heterogeneous: {transition.heterogeneous}")
    assert not transition.heterogeneous, "同质图模式应该设置 heterogeneous=False"
    
    device = torch.device("cpu")
    beta_t = torch.tensor([0.5])
    alpha_bar_t = torch.tensor([0.5])
    
    # 应该使用全局转移矩阵
    Qt = transition.get_Qt(beta_t, device)
    Qtb = transition.get_Qt_bar(alpha_bar_t, device)
    
    print(f"  Q_e 形状: {Qt.E.shape}, 期望 = (1, 6, 6)")
    assert Qt.E.shape == (1, 6, 6), "同质图模式应该使用全局转移矩阵"
    
    # 即使指定 edge_family_name，也应该使用全局转移矩阵
    Qt_fam = transition.get_Qt(beta_t, device, edge_family_name="author_of")
    assert Qt_fam.E.shape == (1, 6, 6), "同质图模式应该忽略 edge_family_name"
    
    print("  ✅ 同质图模式验证通过")


def test_with_real_data():
    """使用真实数据测试"""
    print("\n" + "=" * 80)
    print("测试 4: 使用真实数据")
    print("=" * 80)
    
    # 这里需要实际加载数据集，暂时跳过
    print("  ⚠️  需要实际数据集，暂时跳过")
    print("  ✅ 真实数据测试跳过（正常）")


if __name__ == "__main__":
    print("开始测试异质图转移矩阵类...\n")
    
    try:
        # 测试 1: 转移矩阵创建
        transition = test_transition_matrix_creation()
        
        # 测试 2: 转移矩阵计算
        test_transition_matrix_computation()
        
        # 测试 3: 同质图模式
        test_homogeneous_mode()
        
        # 测试 4: 真实数据（跳过）
        test_with_real_data()
        
        print("\n" + "=" * 80)
        print("✅ 所有测试通过！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
