#!/usr/bin/env python3
"""
调试脚本：查看噪声图的结构，特别是拉普拉斯矩阵
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')

from sparse_diffusion.datasets.acm_subgraphs_dataset import ACMSubgraphsDataset, ACMSubgraphsDataModule
from sparse_diffusion.datasets.dataset_utils import RemoveYTransform
from sparse_diffusion import utils

# 加载数据集
root_path = "/data2/lyh/gnn_project/data/ACM_subgraphs"
dataset = ACMSubgraphsDataset(
    split="train",
    root=root_path,
    pre_transform=RemoveYTransform(),
    heterogeneous=True
)

if len(dataset) == 0:
    print("数据集为空")
    sys.exit(1)

# 获取数据集信息
from sparse_diffusion.datasets.acm_subgraphs_dataset import ACMSubgraphsInfos

class MockConfig:
    class Dataset:
        name = 'acm_subgraphs'
        datadir = 'data/ACM_subgraphs'
        heterogeneous = True
    dataset = Dataset()
    
    class Model:
        n_layers = 2
        diffusion_noise_schedule = 'cosine'
        diffusion_steps = 20
        hidden_mlp_dims = {'X': 17, 'E': 18, 'y': 19}
        hidden_dims = {'dx': 20, 'de': 21, 'dy': 22, 'n_head': 5, 'dim_ffX': 23, 'dim_ffE': 24, 'dim_ffy': 25}
        extra_features = 'all'
        eigenfeatures = True
        edge_features = 'all'
        num_eigenvectors = 8
        num_eigenvalues = 5
        num_degree = 10
        dist_feat = True
        positional_encoding = False
        sign_net = False
        use_charge = False
        output_y = False
        scaling_layer = False
    model = Model()
    
    class Train:
        batch_size = 4
    train = Train()
    
    class General:
        name = 'debug'
    general = General()

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

# 创建模型（简化版，只用于测试噪声应用）
from sparse_diffusion.metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from sparse_diffusion.diffusion.extra_features import DummyExtraFeatures

# 创建数据加载器
from torch_geometric.loader import DataLoader

train_loader = DataLoader(dataset[:10], batch_size=4, shuffle=False)

# 获取一个批次
for batch in train_loader:
    print("=" * 80)
    print("原始批次信息:")
    print(f"  节点数: {batch.x.shape[0]}")
    print(f"  边数: {batch.edge_index.shape[1]}")
    print(f"  batch 信息: {batch.batch.shape}, 唯一值: {torch.unique(batch.batch)}")
    print(f"  edge_attr 形状: {batch.edge_attr.shape}")
    print(f"  edge_attr 范围: [{batch.edge_attr.min().item()}, {batch.edge_attr.max().item()}]")
    
    # 转换为 one-hot
    batch = dataset_infos.to_one_hot(batch)
    print(f"\n转换为 one-hot 后:")
    print(f"  node_t 形状: {batch.x.shape}")
    print(f"  edge_attr_t 形状: {batch.edge_attr.shape}")
    
    # 创建模型（简化，只用于测试噪声应用）
    # 我们需要手动应用噪声来查看结果
    from sparse_diffusion.diffusion.heterogeneous_transition import HeterogeneousMarginalUniformTransition
    
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
    
    # 手动应用噪声（简化版）
    bs = int(batch.batch.max() + 1)
    device = batch.x.device
    t_int = torch.randint(1, 21, size=(bs, 1), device=device).float()
    t_float = t_int / 20.0
    alpha_t_bar = (1 - t_float) ** 2  # 简化的噪声调度
    
    # 转换为有向边
    dir_edge_index, dir_edge_attr = utils.undirected_to_directed(
        batch.edge_index, batch.edge_attr
    )
    
    print(f"\n转换为有向边后:")
    print(f"  dir_edge_index 形状: {dir_edge_index.shape}")
    print(f"  dir_edge_attr 形状: {dir_edge_attr.shape}")
    print(f"  dir_edge_attr 范围: [{dir_edge_attr.min().item()}, {dir_edge_attr.max().item()}]")
    
    # 获取所有关系族的转移矩阵
    all_family_qt = transition.get_all_family_Qt_bar(alpha_t_bar, device=device)
    
    # 转换为密集格式用于特征计算
    from sparse_diffusion.diffusion.extra_features import EigenFeatures
    
    # 创建简化的噪声数据字典
    # 首先需要将边转换为 one-hot
    E_t_attr_onehot = F.one_hot(dir_edge_attr, num_classes=dataset_infos.num_edge_types)
    
    # 转换为无向边
    E_t_index, E_t_attr_onehot = utils.to_undirected(dir_edge_index, E_t_attr_onehot)
    
    # 创建 node_mask
    num_nodes = batch.x.shape[0]
    node_mask = torch.ones(bs, num_nodes, dtype=torch.bool, device=device)
    # 简化：假设所有节点都有效
    
    noisy_data = {
        "E_t": None,  # 需要密集格式
        "node_mask": node_mask,
    }
    
    # 转换为密集格式
    from sparse_diffusion import utils as sparse_utils
    
    # 使用 utils.densify_noisy_data 需要完整的 sparse_noisy_data
    # 让我们手动创建密集格式的邻接矩阵
    max_n_nodes = dataset_infos.max_n_nodes
    A = torch.zeros(bs, max_n_nodes, max_n_nodes, device=device)
    
    # 填充邻接矩阵
    for b in range(bs):
        batch_mask = batch.batch == b
        batch_nodes = torch.where(batch_mask)[0]
        batch_edge_mask = (batch.batch[E_t_index[0]] == b) & (batch.batch[E_t_index[1]] == b)
        batch_edges = E_t_index[:, batch_edge_mask]
        
        # 转换为局部索引
        node_to_local = {n.item(): i for i, n in enumerate(batch_nodes)}
        for e_idx in range(batch_edges.shape[1]):
            src, dst = batch_edges[0, e_idx].item(), batch_edges[1, e_idx].item()
            if src in node_to_local and dst in node_to_local:
                local_src = node_to_local[src]
                local_dst = node_to_local[dst]
                # 边存在（非 no-edge）
                if E_t_attr_onehot[batch_edge_mask][e_idx, 1:].sum() > 0:
                    A[b, local_src, local_dst] = 1.0
                    A[b, local_dst, local_src] = 1.0  # 无向图
    
    print(f"\n密集格式邻接矩阵:")
    print(f"  A 形状: {A.shape}")
    print(f"  A 非零元素数: {(A > 0).sum().item()}")
    print(f"  每个图的边数:")
    for b in range(bs):
        print(f"    图 {b}: {(A[b] > 0).sum().item() // 2} 条边")
    
    # 计算拉普拉斯矩阵
    def compute_laplacian(A, normalize=False):
        """计算拉普拉斯矩阵"""
        D = A.sum(dim=-1)  # 度矩阵
        if normalize:
            D_inv_sqrt = torch.pow(D + 1e-8, -0.5)
            D_inv_sqrt[D == 0] = 0
            L = torch.eye(A.shape[-1], device=A.device).unsqueeze(0) - torch.bmm(
                D_inv_sqrt.unsqueeze(-1) * A, D_inv_sqrt.unsqueeze(-1)
            )
        else:
            L = torch.diag_embed(D) - A
        return L
    
    L = compute_laplacian(A, normalize=False)
    L = L * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    
    print(f"\n拉普拉斯矩阵:")
    print(f"  L 形状: {L.shape}")
    print(f"  L 的条件数:")
    for b in range(bs):
        L_b = L[b]
        # 只取有效部分
        valid_nodes = node_mask[b].sum().item()
        L_b_valid = L_b[:valid_nodes, :valid_nodes]
        
        # 计算条件数
        try:
            eigvals = torch.linalg.eigvals(L_b_valid)
            eigvals_real = eigvals.real
            eigvals_real = eigvals_real[eigvals_real > 1e-8]  # 过滤接近0的特征值
            if len(eigvals_real) > 0:
                cond_num = eigvals_real.max() / (eigvals_real.min() + 1e-8)
                print(f"    图 {b}: 条件数 = {cond_num.item():.2e}, 特征值范围: [{eigvals_real.min().item():.2e}, {eigvals_real.max().item():.2e}]")
            else:
                print(f"    图 {b}: 所有特征值接近0")
        except Exception as e:
            print(f"    图 {b}: 计算特征值失败 - {e}")
            print(f"      L_b_valid 形状: {L_b_valid.shape}")
            print(f"      L_b_valid 是否有 NaN: {L_b_valid.isnan().any()}")
            print(f"      L_b_valid 是否有 Inf: {L_b_valid.isinf().any()}")
    
    # 尝试使用 EigenFeatures 计算特征
    print(f"\n尝试使用 EigenFeatures 计算特征:")
    eigenfeatures = EigenFeatures(num_eigenvectors=8, num_eigenvalues=5)
    
    # 需要完整的 noisy_data
    dense_noisy_data = {
        "E_t": E_t_attr_onehot.unsqueeze(0).expand(bs, -1, -1, -1) if E_t_attr_onehot.dim() == 2 else E_t_attr_onehot,
        "node_mask": node_mask,
    }
    
    try:
        eval_feat, evec_feat = eigenfeatures.compute_features(dense_noisy_data)
        print(f"  ✅ 特征计算成功")
        print(f"  eval_feat 形状: {eval_feat.shape}")
        print(f"  evec_feat 形状: {evec_feat.shape}")
    except Exception as e:
        print(f"  ❌ 特征计算失败: {e}")
        import traceback
        traceback.print_exc()
    
    break  # 只处理第一个批次
