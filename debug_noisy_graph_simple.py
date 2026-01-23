#!/usr/bin/env python3
"""
简化调试脚本：查看噪声图在特征计算时的结构
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')

from sparse_diffusion.datasets.acm_subgraphs_dataset import ACMSubgraphsDataset
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
    
    # 模拟 apply_sparse_noise 的输出（简化版）
    # 转换为有向边
    dir_edge_index, dir_edge_attr = utils.undirected_to_directed(
        batch.edge_index, batch.edge_attr
    )
    
    print(f"\n转换为有向边后:")
    print(f"  dir_edge_index 形状: {dir_edge_index.shape}")
    print(f"  dir_edge_attr 形状: {dir_edge_attr.shape}")
    print(f"  dir_edge_attr 是 one-hot: {dir_edge_attr.dtype == torch.float32}")
    
    # 转换为无向边（用于特征计算）
    E_t_index, E_t_attr = utils.to_undirected(dir_edge_index, dir_edge_attr)
    
    print(f"\n转换为无向边后（用于特征计算）:")
    print(f"  E_t_index 形状: {E_t_index.shape}")
    print(f"  E_t_attr 形状: {E_t_attr.shape}")
    
    # 创建 sparse_noisy_data（模拟 apply_sparse_noise 的输出）
    bs = int(batch.batch.max() + 1)
    device = batch.x.device
    max_n_nodes = dataset_infos.max_n_nodes
    
    sparse_noisy_data = {
        "node_t": batch.x,
        "edge_index_t": E_t_index,
        "edge_attr_t": E_t_attr,
        "batch": batch.batch,
        "y_t": batch.y,
        "charge_t": torch.zeros(batch.x.shape[0], 0, device=batch.x.device),  # 空 charge
    }
    
    print(f"\nsparse_noisy_data 结构:")
    for key, value in sparse_noisy_data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    # 转换为密集格式（模拟 compute_extra_data 中的 densify_noisy_data）
    print(f"\n转换为密集格式:")
    dense_noisy_data = utils.densify_noisy_data(sparse_noisy_data)
    
    print(f"dense_noisy_data 结构:")
    for key, value in dense_noisy_data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")
            if key == "E_t":
                print(f"    E_t 非零元素数: {(value > 0).sum().item()}")
                print(f"    E_t 每个图的边数:")
                for b in range(bs):
                    E_t_b = value[b]
                    # E_t 形状是 (bs, max_n_nodes, max_n_nodes, num_edge_types)
                    # 边存在性 = E_t[..., 1:].sum(dim=-1)
                    A_b = E_t_b[..., 1:].sum(dim=-1)
                    num_edges = (A_b > 0).sum().item() // 2
                    print(f"      图 {b}: {num_edges} 条边")
            elif key == "node_mask":
                print(f"    node_mask 每个图的节点数:")
                for b in range(bs):
                    num_nodes = value[b].sum().item()
                    print(f"      图 {b}: {num_nodes} 个节点")
    
    # 计算拉普拉斯矩阵（模拟 EigenFeatures.compute_features）
    print(f"\n计算拉普拉斯矩阵:")
    E_t = dense_noisy_data["E_t"]
    mask = dense_noisy_data["node_mask"]
    
    # 构建邻接矩阵
    A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
    
    print(f"  邻接矩阵 A 形状: {A.shape}")
    print(f"  每个图的边数:")
    for b in range(bs):
        A_b = A[b]
        num_edges = (A_b > 0).sum().item() // 2
        print(f"    图 {b}: {num_edges} 条边")
    
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
    mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
    mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
    L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag
    
    print(f"  拉普拉斯矩阵 L 形状: {L.shape}")
    print(f"  L 是否有 NaN: {L.isnan().any().item()}")
    print(f"  L 是否有 Inf: {L.isinf().any().item()}")
    
    # 检查每个图的拉普拉斯矩阵
    print(f"\n每个图的拉普拉斯矩阵分析:")
    for b in range(bs):
        valid_nodes = mask[b].sum().item()
        L_b = L[b, :valid_nodes, :valid_nodes]
        
        print(f"\n  图 {b}:")
        print(f"    有效节点数: {valid_nodes}")
        print(f"    L_b 形状: {L_b.shape}")
        print(f"    L_b 是否有 NaN: {L_b.isnan().any().item()}")
        print(f"    L_b 是否有 Inf: {L_b.isinf().any().item()}")
        
        # 检查对称性
        is_symmetric = torch.allclose(L_b, L_b.T, atol=1e-5)
        print(f"    L_b 是否对称: {is_symmetric}")
        
        # 计算条件数
        try:
            eigvals = torch.linalg.eigvals(L_b)
            eigvals_real = eigvals.real
            eigvals_imag = eigvals.imag
            
            print(f"    特征值:")
            print(f"      实部范围: [{eigvals_real.min().item():.2e}, {eigvals_real.max().item():.2e}]")
            print(f"      虚部范围: [{eigvals_imag.min().item():.2e}, {eigvals_imag.max().item():.2e}]")
            print(f"      接近0的特征值数: {(eigvals_real.abs() < 1e-6).sum().item()}")
            print(f"      重复特征值数: {len(eigvals_real) - len(torch.unique(eigvals_real.round(decimals=6)))}")
            
            # 过滤接近0的特征值
            eigvals_positive = eigvals_real[eigvals_real > 1e-8]
            if len(eigvals_positive) > 0:
                cond_num = eigvals_positive.max() / (eigvals_positive.min() + 1e-8)
                print(f"      条件数: {cond_num.item():.2e}")
            else:
                print(f"      条件数: 所有特征值接近0")
            
            # 尝试特征值分解
            try:
                eigvals_full, eigvectors = torch.linalg.eigh(L_b)
                print(f"      ✅ 特征值分解成功")
            except Exception as e:
                print(f"      ❌ 特征值分解失败: {e}")
        except Exception as e:
            print(f"    计算特征值失败: {e}")
    
    break  # 只处理第一个批次
