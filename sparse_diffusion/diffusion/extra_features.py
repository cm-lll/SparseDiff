import time
import math

import torch
import torch.nn.functional as F
from sparse_diffusion import utils


def batch_trace(X):
    """ Expect a matrix of shape B N N, returns the trace in shape B."""
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    return diag.sum(dim=-1)


def batch_diagonal(X):
    """ Extracts the diagonal from the last two dims of a tensor. """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class DummyExtraFeatures:
    """This class does not compute anything, just returns empty tensors."""
    def __call__(self, noisy_data, sparse=True):
        # compute_extra_data 期望返回 PlaceHolder（密集格式），即使输入是稀疏的
        # 所以我们需要将稀疏数据转换为密集格式
        
        if "X_t" in noisy_data:
            # 已经是密集格式
            X = noisy_data["X_t"]  # (bs, n, dx) 或 (N, dx)
            E = noisy_data["E_t"]  # (bs, n, n, de) 或 (bs, n, n) 或其他
            y = noisy_data["y_t"]  # (bs, dy)
            
            # 处理X：确保是 (bs, n, dx) 格式
            if X.dim() == 2:  # (N, dx) - 需要reshape
                # 需要从batch信息推断bs和n
                if "batch" in noisy_data:
                    batch = noisy_data["batch"]
                    bs = batch.max().item() + 1
                    n = X.shape[0] // bs
                    X = X.reshape(bs, n, -1)
                else:
                    # 假设是单个图
                    X = X.unsqueeze(0)  # (1, n, dx)
            
            # 处理E：确保是 (bs, n, n, de) 格式
            if E.dim() == 3:  # (bs, n, n) - 需要添加de维度
                E = E.unsqueeze(-1)  # (bs, n, n, 1)
            elif E.dim() == 2:  # (n, n) - 需要添加bs和de维度
                E = E.unsqueeze(0).unsqueeze(-1)  # (1, n, n, 1)
            
            empty_x = X.new_zeros((*X.shape[:-1], 0))  # (bs, n, 0)
            empty_e = E.new_zeros((*E.shape[:-1], 0))  # (bs, n, n, 0)
            empty_y = y.new_zeros((y.shape[0], 0))  # (bs, 0)
            return utils.PlaceHolder(
                X=empty_x, E=empty_e, y=empty_y
            ), 0., 0.
        else:
            # 稀疏格式，需要转换为密集格式
            # 使用 densify_noisy_data 来转换
            dense_noisy_data = utils.densify_noisy_data(noisy_data)
            X = dense_noisy_data["X_t"]  # (bs, n, dx)
            E = dense_noisy_data["E_t"]  # (bs, n, n, de)
            y = dense_noisy_data["y_t"]  # (bs, dy)
            
            # 确保E是4维的 (bs, n, n, de)
            if E.dim() == 3:
                # 如果E是 (bs, n, n)，需要添加de维度
                E = E.unsqueeze(-1)  # (bs, n, n, 1)
            
            empty_x = X.new_zeros((*X.shape[:-1], 0))  # (bs, n, 0)
            empty_e = E.new_zeros((*E.shape[:-1], 0))  # (bs, n, n, 0)
            empty_y = y.new_zeros((y.shape[0], 0))  # (bs, 0)
            return utils.PlaceHolder(
                X=empty_x, E=empty_e, y=empty_y
            ), 0., 0.


class ExtraFeatures:
    def __init__(self, eigenfeatures: bool, edge_features_type, dataset_info, num_eigenvectors,
                 num_eigenvalues, num_degree, dist_feat, use_positional: bool):
        self.eigenfeatures = eigenfeatures
        self.max_n_nodes = dataset_info.max_n_nodes
        self.edge_features = edge_features_type
        self.use_positional = use_positional
        
        # 检查是否为异质图
        self.heterogeneous = getattr(dataset_info, "heterogeneous", False)
        
        if use_positional:
            self.positional_encoding = PositionalEncoding(dataset_info.max_n_nodes)
        self.adj_features = AdjacencyFeatures(edge_features_type, num_degree=num_degree, dist_feat=dist_feat)
        
        # 对于异质图，使用专门设计的 HeterogeneousGraphFeatures
        # 对于同质图，使用 EigenFeatures
        if eigenfeatures:
            if self.heterogeneous:
                # 异质图使用专门的特征计算类
                self.eigenfeatures = HeterogeneousGraphFeatures(
                    dataset_info=dataset_info,
                    num_eigenvectors=num_eigenvectors,
                    num_eigenvalues=num_eigenvalues
                )
            else:
                # 同质图使用原始的特征值特征
                self.eigenfeatures = EigenFeatures(num_eigenvectors=num_eigenvectors, num_eigenvalues=num_eigenvalues)
        else:
            self.eigenfeatures = False

    def __call__(self, sparse_noisy_data):
        # make data dense in the beginning to avoid doing this twice for both cycles and eigenvalues
        noisy_data = utils.densify_noisy_data(sparse_noisy_data)
        n = noisy_data["node_mask"].sum(dim=1).unsqueeze(1) / self.max_n_nodes
        start_time = time.time()
        x_feat, y_feat, edge_feat = self.adj_features(noisy_data)  # (bs, n_cycles)
        y_feat = torch.hstack((y_feat, n))
        cycle_time = round(time.time() - start_time, 2)
        eigen_time = 0.

        if self.use_positional:
            node_feat = self.positional_encoding(noisy_data)
            x_feat = torch.cat([x_feat, node_feat], dim=-1)

        if self.eigenfeatures:
            start_time = time.time()
            try:
                eval_feat, evec_feat = self.eigenfeatures.compute_features(noisy_data)
                eigen_time = round(time.time() - start_time, 2)
                x_feat = torch.cat((x_feat, evec_feat), dim=-1)
                y_feat = torch.hstack((y_feat, eval_feat))
            except Exception as e:
                # 对于异质图，特征值计算可能失败（数值稳定性问题）
                # 如果失败，使用零特征
                if self.heterogeneous:
                    print(f"Warning: EigenFeatures computation failed for heterogeneous graph: {e}")
                    print("Using zero features instead. Consider setting eigenfeatures=False for heterogeneous graphs.")
                else:
                    raise e
                eigen_time = round(time.time() - start_time, 2)
                bs = noisy_data["node_mask"].shape[0]
                n_nodes = noisy_data["node_mask"].shape[1]
                # 获取正确的特征维度
                num_eigenvectors = self.eigenfeatures.num_eigenvectors
                num_eigenvalues = self.eigenfeatures.num_eigenvalues
                
                # 创建零特征，维度需要匹配 eigenvector_features 的输出
                # eigenvector_features 返回 (bs, n, num_eigenvectors + 1)
                # 其中 +1 是 not_lcc_indicator
                evec_feat = torch.zeros(bs, n_nodes, num_eigenvectors + 1, device=noisy_data["node_mask"].device)
                # eigenvalues_features 返回 (n_connected_comp, batch_eigenvalues)
                # n_connected_comp 是 (bs, 1)，batch_eigenvalues 是 (bs, num_eigenvalues)
                eval_feat = torch.zeros(bs, 1 + num_eigenvalues, device=noisy_data["node_mask"].device)
                x_feat = torch.cat((x_feat, evec_feat), dim=-1)
                y_feat = torch.hstack((y_feat, eval_feat))

        return utils.PlaceHolder(X=x_feat, E=edge_feat, y=y_feat), cycle_time, eigen_time


class PositionalEncoding:
    def __init__(self, n_max_dataset, D=30):
        self.n_max = n_max_dataset
        self.d = math.floor(D / 2)

    def __call__(self, dense_noisy_data):
        device = dense_noisy_data['X_t'].device
        n_max_batch = dense_noisy_data['X_t'].shape[1]

        arange_n = torch.arange(n_max_batch, device=device)                                    # n_max
        arange_d = torch.arange(self.d, device=device)                                         # d
        frequencies = math.pi / torch.pow(self.n_max, 2 * arange_d / self.d)    # d

        sines = torch.sin(arange_n.unsqueeze(1) * frequencies.unsqueeze(0))     # N, d
        cosines = torch.cos(arange_n.unsqueeze(1) * frequencies.unsqueeze(0))   # N, d
        encoding = torch.hstack((sines, cosines))                               # N, D
        extra_x = encoding.unsqueeze(0)                                         # 1, N, D
        extra_x = extra_x * dense_noisy_data['node_mask'].unsqueeze(-1)             # B, N, D
        return extra_x

class HeterogeneousGraphFeatures:
    """
    为异质图设计的图特征，替代 EigenFeatures。
    保留异质信息，为每种关系类型分别计算特征。
    """
    def __init__(self, dataset_info, num_eigenvectors, num_eigenvalues):
        self.num_eigenvectors = num_eigenvectors
        self.num_eigenvalues = num_eigenvalues
        self.dataset_info = dataset_info
        
        # 获取异质图信息
        self.edge_family_offsets = getattr(dataset_info, "edge_family_offsets", {})
        self.edge_family2id = getattr(dataset_info, "edge_family2id", {})
        self.id2edge_family = {v: k for k, v in self.edge_family2id.items()} if self.edge_family2id else {}
        self.num_node_types = getattr(dataset_info, "num_node_types", 1)
        
    def compute_features(self, noisy_data):
        """
        为异质图计算特征。
        
        Returns:
            evalue_feat: (bs, num_eigenvalues + 1) - 图级别的特征值特征（使用关系类型特定的统计）
            evector_feat: (bs, n, num_eigenvectors) - 节点级别的特征向量特征（使用关系类型特定的度特征）
        """
        E_t = noisy_data["E_t"]  # (bs, n, n, de)
        mask = noisy_data["node_mask"]  # (bs, n)
        bs, n = mask.shape
        
        # 计算关系类型特定的特征
        # 1. 为每种关系类型计算度特征（替代特征向量）
        evector_feat = self._compute_relation_type_degree_features(E_t, mask)  # (bs, n, num_eigenvectors)
        
        # 2. 计算关系类型分布特征（替代特征值）
        evalue_feat = self._compute_relation_type_statistics(E_t, mask)  # (bs, num_eigenvalues + 1)
        
        return evalue_feat, evector_feat
    
    def _compute_relation_type_degree_features(self, E_t, mask):
        """
        为每种关系类型计算节点的度特征。
        
        Args:
            E_t: (bs, n, n, de) - 边属性（one-hot编码）
            mask: (bs, n) - 节点掩码
            
        Returns:
            features: (bs, n, num_eigenvectors) - 节点级别的特征
        """
        bs, n, _, de = E_t.shape
        device = E_t.device
        
        # 获取所有关系族的offset
        edge_family_offsets = self.edge_family_offsets
        if not edge_family_offsets:
            # 如果没有关系族信息，使用全局度特征
            A = E_t[..., 1:].sum(dim=-1).float()  # (bs, n, n)
            degree = A.sum(dim=-1)  # (bs, n)
            # 归一化并扩展到 num_eigenvectors 维
            degree = degree / (degree.sum(dim=-1, keepdim=True) + 1e-8)
            # 使用不同的幂次来创建多个特征
            features = []
            for i in range(self.num_eigenvectors):
                features.append(degree.unsqueeze(-1))
            return torch.cat(features, dim=-1)  # (bs, n, num_eigenvectors)
        
        # 为每种关系类型计算度特征
        features_list = []
        
        # 计算每种关系类型的度
        for fam_name, offset in edge_family_offsets.items():
            # 找到该关系族的范围
            next_offset = de
            for other_fam_name, other_offset in edge_family_offsets.items():
                if other_offset > offset and other_offset < next_offset:
                    next_offset = other_offset
            
            # 确保索引在有效范围内
            offset = max(0, min(offset, de))
            next_offset = max(offset + 1, min(next_offset, de))
            
            # 提取该关系族的边
            fam_E_t = E_t[..., offset:next_offset]  # (bs, n, n, num_fam_states)
            fam_A = fam_E_t.sum(dim=-1).float()  # (bs, n, n) - 该关系族的邻接矩阵
            
            # 计算入度和出度
            in_degree = fam_A.sum(dim=1)  # (bs, n) - 入度
            out_degree = fam_A.sum(dim=2)  # (bs, n) - 出度
            total_degree = in_degree + out_degree  # (bs, n)
            
            # 归一化
            total_degree = total_degree / (total_degree.sum(dim=-1, keepdim=True) + 1e-8)
            in_degree = in_degree / (in_degree.sum(dim=-1, keepdim=True) + 1e-8)
            out_degree = out_degree / (out_degree.sum(dim=-1, keepdim=True) + 1e-8)
            
            features_list.extend([total_degree.unsqueeze(-1), in_degree.unsqueeze(-1), out_degree.unsqueeze(-1)])
        
        # 如果特征数量不够，用全局度特征填充
        all_features = torch.cat(features_list, dim=-1)  # (bs, n, num_fam_features)
        if all_features.shape[-1] < self.num_eigenvectors:
            # 使用全局度特征填充
            A_global = E_t[..., 1:].sum(dim=-1).float()
            degree_global = A_global.sum(dim=-1)
            degree_global = degree_global / (degree_global.sum(dim=-1, keepdim=True) + 1e-8)
            padding = self.num_eigenvectors - all_features.shape[-1]
            for _ in range(padding):
                all_features = torch.cat([all_features, degree_global.unsqueeze(-1)], dim=-1)
        elif all_features.shape[-1] > self.num_eigenvectors:
            # 截断到 num_eigenvectors
            all_features = all_features[..., :self.num_eigenvectors]
        
        # 应用节点掩码
        all_features = all_features * mask.unsqueeze(-1)
        
        return all_features  # (bs, n, num_eigenvectors)
    
    def _compute_relation_type_statistics(self, E_t, mask):
        """
        计算关系类型分布统计特征（替代特征值）。
        
        Args:
            E_t: (bs, n, n, de) - 边属性
            mask: (bs, n) - 节点掩码
            
        Returns:
            features: (bs, num_eigenvalues + 1) - 图级别的特征
        """
        bs, n, _, de = E_t.shape
        device = E_t.device
        
        # 计算图的连通分量数量（使用全局邻接矩阵）
        A_global = E_t[..., 1:].sum(dim=-1).float()  # (bs, n, n)
        A_global = A_global * mask.unsqueeze(1) * mask.unsqueeze(2)
        
        # 简单的连通分量估计：使用度分布
        degree = A_global.sum(dim=-1)  # (bs, n)
        num_connected_components = (degree == 0).sum(dim=-1).float()  # 孤立节点数量作为连通分量数量的下界
        num_connected_components = num_connected_components.unsqueeze(-1)  # (bs, 1)
        
        # 计算每种关系类型的统计特征
        edge_family_offsets = self.edge_family_offsets
        if not edge_family_offsets:
            # 如果没有关系族信息，使用全局统计
            edge_dist = E_t[..., 1:].sum(dim=[1, 2])  # (bs, de-1)
            edge_dist = edge_dist / (edge_dist.sum(dim=-1, keepdim=True) + 1e-8)
            # 取前 num_eigenvalues 个值
            if edge_dist.shape[-1] >= self.num_eigenvalues:
                edge_dist = edge_dist[..., :self.num_eigenvalues]
            else:
                padding = self.num_eigenvalues - edge_dist.shape[-1]
                edge_dist = torch.cat([edge_dist, torch.zeros(bs, padding, device=device)], dim=-1)
            return torch.cat([num_connected_components, edge_dist], dim=-1)  # (bs, num_eigenvalues + 1)
        
        # 为每种关系类型计算统计特征
        stats_list = []
        
        for fam_name, offset in edge_family_offsets.items():
            # 找到该关系族的范围
            next_offset = de
            for other_fam_name, other_offset in edge_family_offsets.items():
                if other_offset > offset and other_offset < next_offset:
                    next_offset = other_offset
            
            # 确保索引在有效范围内
            offset = max(0, min(offset, de))
            next_offset = max(offset + 1, min(next_offset, de))
            
            # 提取该关系族的边
            fam_E_t = E_t[..., offset:next_offset]  # (bs, n, n, num_fam_states)
            fam_A = fam_E_t.sum(dim=-1).float()  # (bs, n, n)
            
            # 计算该关系类型的统计特征
            num_edges = fam_A.sum(dim=[1, 2])  # (bs) - 该关系类型的边数量
            avg_degree = (fam_A.sum(dim=-1).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8))  # (bs) - 平均度
            
            stats_list.extend([num_edges.unsqueeze(-1), avg_degree.unsqueeze(-1)])
        
        # 组合所有统计特征
        all_stats = torch.cat(stats_list, dim=-1)  # (bs, num_fam_stats)
        
        # 如果统计特征数量不够，用零填充
        if all_stats.shape[-1] < self.num_eigenvalues:
            padding = self.num_eigenvalues - all_stats.shape[-1]
            all_stats = torch.cat([all_stats, torch.zeros(bs, padding, device=device)], dim=-1)
        elif all_stats.shape[-1] > self.num_eigenvalues:
            # 截断到 num_eigenvalues
            all_stats = all_stats[..., :self.num_eigenvalues]
        
        return torch.cat([num_connected_components, all_stats], dim=-1)  # (bs, num_eigenvalues + 1)


class EigenFeatures:
    """  Some code is taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py. """
    def __init__(self, num_eigenvectors, num_eigenvalues):
        self.num_eigenvectors = num_eigenvectors
        self.num_eigenvalues = num_eigenvalues

    def compute_features(self, noisy_data):
        E_t = noisy_data["E_t"]
        mask = noisy_data["node_mask"]
        A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = self.compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        # debug for protein dataset
        if L.isnan().any():
            import pdb; pdb.set_trace()

        try:
            eigvals, eigvectors = torch.linalg.eigh(L)
        except:
            eigvals, eigvectors = torch.linalg.eigh(L.cpu())  # debug for point cloud dataset
            eigvals = eigvals.to(L.device)
            eigvectors = eigvectors.to(L.device)

        eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
        eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
        # Retrieve eigenvalues features
        n_connected_comp, batch_eigenvalues = self.eigenvalues_features(
            eigenvalues=eigvals, num_eigenvalues=self.num_eigenvalues
        )
        # Retrieve eigenvectors features
        evector_feat = self.eigenvector_features(
            vectors=eigvectors,
            node_mask=noisy_data["node_mask"],
            n_connected=n_connected_comp,
            num_eigenvectors=self.num_eigenvectors,
        )

        evalue_feat = torch.hstack((n_connected_comp, batch_eigenvalues))
        return evalue_feat, evector_feat

    def compute_laplacian(self, adjacency, normalize: bool):
        """
        adjacency : batched adjacency matrix (bs, n, n)
        normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
        Return:
            L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
        """
        diag = torch.sum(adjacency, dim=-1)  # (bs, n)
        n = diag.shape[-1]
        D = torch.diag_embed(diag)  # Degree matrix      # (bs, n, n)
        combinatorial = D - adjacency  # (bs, n, n)

        if not normalize:
            return (combinatorial + combinatorial.transpose(1, 2)) / 2

        diag0 = diag.clone()
        diag[diag == 0] = 1e-12

        diag_norm = 1 / torch.sqrt(diag)  # (bs, n)
        D_norm = torch.diag_embed(diag_norm)  # (bs, n, n)
        L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
        L[diag0 == 0] = 0
        return (L + L.transpose(1, 2)) / 2

    def eigenvalues_features(self, eigenvalues, num_eigenvalues):
        """
        values : eigenvalues -- (bs, n)
        node_mask: (bs, n)
        k: num of non zero eigenvalues to keep
        """
        ev = eigenvalues
        bs, n = ev.shape
        n_connected_components = (ev < 1e-4).sum(dim=-1)

        # if (n_connected_components <= 0).any():
        #     import pdb; pdb.set_trace()
        # assert (n_connected_components > 0).all(), (n_connected_components, ev)

        try:
            to_extend = max(n_connected_components) + num_eigenvalues - n
            if to_extend > 0:
                ev = torch.hstack((ev, 2 * torch.ones(bs, to_extend, device=ev.device)))
            indices = torch.arange(num_eigenvalues, device=ev.device).unsqueeze(0) + n_connected_components.unsqueeze(1)
            first_k_ev = torch.gather(ev, dim=1, index=indices)
        except:
            import pdb; pdb.set_trace()

        return n_connected_components.unsqueeze(-1), first_k_ev

    def eigenvector_features(self, vectors, node_mask, n_connected, num_eigenvectors):
        """
        vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
        returns:
            not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
            k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
        """
        bs, n = vectors.size(0), vectors.size(1)

        # Create an indicator for the nodes outside the largest connected components
        first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask  # bs, n
        # Add random value to the mask to prevent 0 from becoming the mode
        random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)  # bs, n
        first_ev = first_ev + random
        most_common = torch.mode(first_ev, dim=1).values  # values: bs -- indices: bs
        mask = ~(first_ev == most_common.unsqueeze(1))
        not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

        # Get the eigenvectors corresponding to the first nonzero eigenvalues
        to_extend = max(n_connected) + num_eigenvectors - n
        if to_extend > 0:
            vectors = torch.cat(
                (vectors, torch.zeros(bs, n, to_extend, device=vectors.device)), dim=2
            )  # bs, n , n + to_extend

        indices = torch.arange(num_eigenvectors, device=vectors.device).long().unsqueeze(0).unsqueeze(0)
        indices = indices + n_connected.unsqueeze(2)  # bs, 1, k
        indices = indices.expand(-1, n, -1)  # bs, n, k
        first_k_ev = torch.gather(vectors, dim=2, index=indices)  # bs, n, k
        first_k_ev = first_k_ev * node_mask.unsqueeze(2)

        return torch.cat((not_lcc_indicator, first_k_ev), dim=-1)


class AdjacencyFeatures:
    """Builds cycle counts for each node in a graph."""
    def __init__(self, edge_features_type, num_degree, max_degree=10, dist_feat=True):
        self.edge_features_type = edge_features_type
        self.max_degree = max_degree
        self.num_degree = num_degree
        self.dist_feat = dist_feat

    def __call__(self, noisy_data):
        adj_matrix = noisy_data["E_t"][..., 1:].int().sum(dim=-1)  # (bs, n, n)
        num_nodes = noisy_data["node_mask"].sum(dim=1)
        self.calculate_kpowers(adj_matrix)

        k3x, k3y = self.k3_cycle()
        k4x, k4y = self.k4_cycle()
        k5x, k5y = self.k5_cycle()
        _, k6y = self.k6_cycle()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)

        if self.edge_features_type == "dist":
            edge_feats = self.path_features()
        elif self.edge_features_type == "localngbs":
            edge_feats = self.local_neighbors(num_nodes)
        elif self.edge_features_type == "all":
            dist = self.path_features()
            local_ngbs = self.local_neighbors(num_nodes)
            edge_feats = torch.cat([dist, local_ngbs], dim=-1)
        else:
            edge_feats = torch.zeros((*adj_matrix.shape, 0), device=adj_matrix.device)

        kcyclesx = torch.clamp(kcyclesx, 0, 5) / 5 * noisy_data["node_mask"].unsqueeze(-1)
        y_feat = [torch.clamp(kcyclesy, 0, 5) / 5]
        edge_feats = torch.clamp(edge_feats, 0, 5) / 5

        if self.dist_feat:
        # get degree distribution
            bs, n = noisy_data["node_mask"].shape
            degree = adj_matrix.sum(dim=-1).long()  # (bs, n)
            degree[degree > self.num_degree] = self.num_degree + 1    # bs, n
            one_hot_degree = F.one_hot(degree, num_classes=self.num_degree + 2).float()  # bs, n, num_degree + 2
            one_hot_degree[~noisy_data["node_mask"]] = 0
            degree_dist = one_hot_degree.sum(dim=1)  # bs, num_degree + 2
            s = degree_dist.sum(dim=-1, keepdim=True)
            s[s == 0] = 1
            degree_dist = degree_dist / s
            y_feat.append(degree_dist)

            # get node distribution
            X = noisy_data["X_t"]       # bs, n, dx
            node_dist = X.sum(dim=1)    # bs, dx
            s = node_dist.sum(-1)     # bs
            s[s == 0] = 1
            node_dist = node_dist / s.unsqueeze(-1)     # bs, dx
            y_feat.append(node_dist)

            # get edge distribution
            E = noisy_data["E_t"]
            edge_dist = E.sum(dim=[1, 2])    # bs, de
            s = edge_dist.sum(-1)     # bs
            s[s == 0] = 1
            edge_dist = edge_dist / s.unsqueeze(-1)     # bs, de
            y_feat.append(edge_dist)

        y_feat = torch.cat(y_feat, dim=-1)

        return kcyclesx, y_feat, edge_feats

    def calculate_kpowers(self, adj):
        """ adj: bs, n, n"""
        shape = (self.max_degree, *adj.shape)
        adj = adj.float()
        self.k = torch.zeros(shape, device=adj.device, dtype=torch.float)
        self.d = adj.sum(dim=-1)
        self.k[0] = adj
        for i in range(1, self.max_degree):
            self.k[i] = self.k[i-1] @ adj

        # Warning: index changes by 1 (count from 1 and not 0)
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6 = [self.k[i] for i in range(6)]

    def k3_cycle(self):
        c3 = batch_diagonal(self.k3)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(
            -1
        ).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4)
        c4 = (
            diag_a4
            - self.d * (self.d - 1)
            - (self.k1 @ self.d.unsqueeze(-1)).sum(dim=-1)
        )
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(
            -1
        ).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5)
        triangles = batch_diagonal(self.k3)

        c5 = (
            diag_a5
            - 2 * triangles * self.d
            - (self.k1 @ triangles.unsqueeze(-1)).sum(dim=-1)
            + triangles
        )
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(
            -1
        ).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6)
        term_2_t = batch_trace(self.k3 ** 2)
        term3_t = torch.sum(self.k1 * self.k2.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2)
        a_4_t = batch_diagonal(self.k4)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4)
        term_6_t = batch_trace(self.k3)
        term_7_t = batch_diagonal(self.k2).pow(3).sum(-1)
        term8_t = torch.sum(self.k3, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2).pow(2).sum(-1)
        term10_t = batch_trace(self.k2)

        c6_t = (
            term_1_t
            - 3 * term_2_t
            + 9 * term3_t
            - 6 * term_4_t
            + 6 * term_5_t
            - 4 * term_6_t
            + 4 * term_7_t
            + 3 * term8_t
            - 12 * term9_t
            + 4 * term10_t
        )
        return None, (c6_t / 12).unsqueeze(-1).float()

    def path_features(self):
        path_features = self.k.bool().float()        # max power, bs, n, n
        path_features = path_features.permute(1, 2, 3, 0)    # bs, n, n, max power
        return path_features

    def local_neighbors(self, num_nodes):
        """ Adamic-Adar index for each pair of nodes.
            this function captures the local neighborhood information, commonly used in social network analysis
            [i, j], sum of 1/log(degree(u)), u is a common neighbor of i and j.
        """
        normed_adj = self.k1 / self.k1.sum(-1).unsqueeze(1)        # divide each column by its degree

        normed_adj = torch.sqrt(torch.log(normed_adj).abs())
        normed_adj = torch.nan_to_num(1 / normed_adj, posinf=0)
        normed_adj = torch.matmul(normed_adj, normed_adj.transpose(-2, -1))

        # mask self-loops to 0
        mask = torch.eye(normed_adj.shape[-1]).repeat(normed_adj.shape[0], 1, 1).bool()
        normed_adj[mask] = 0

        # normalization
        normed_adj = (
            normed_adj * num_nodes.log()[:, None, None] / num_nodes[:, None, None]
        )
        return normed_adj.unsqueeze(-1)
