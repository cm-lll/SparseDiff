"""
异质图 Transformer 层：支持异质的 MP

1. 节点：子类别调制类别嵌入
   Q = (1 + γ_Q(S)) ⊙ W^Q_t X + β_Q(S)
   - X 来自类别嵌入 E^T，S 来自子类别嵌入 E_{sub}^T；无子类时 S 可退化到型内单一 id。

2. 边的融合（含子类与无子类）
   - 关系类别 (src_type, dst_type) 的 relation_emb，与 edge_attr（有子类如一作/二作/三作，
     无子类如隶属等该族状态）融合后做 FiLM，调制注意力。

3. 分类别 (per-relation-family) softmax
   - 族内对边做 softmax，避免不同关系族之间竞争。

4. 外层归一化
   - 对每个节点在其所有入边上做归一化（除以和），使入边权重和为 1，避免关系多的节点权重过大；
     使用归一化而非第二次 softmax，以保留族内 softmax 给出的相对重要性。
"""
import math
import warnings
from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.utils import softmax, scatter
from sparse_diffusion.models.layers import SparseXtoy, SparseEtoy


class HeterogeneousTransformerConv(MessagePassing):
    r"""异质图Transformer层
    
    实现图片中描述的异质图架构：
    - 类别和子类别的融合机制
    - Relation-aware的边嵌入
    - Per-relation-family softmax + outer softmax
    """
    _alpha: OptTensor

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        last_layer: bool = True,
        # 异质图相关参数
        heterogeneous: bool = False,
        num_node_types: int = 0,  # 类别数量
        num_node_subtypes: int = 0,  # 子类别数量
        num_relation_types: int = 0,  # 关系类型数量（src_type, dst_type的组合）
        type_embed_dim: int = 64,  # 类别嵌入维度
        subtype_embed_dim: int = 64,  # 子类别嵌入维度
        relation_embed_dim: int = 64,  # 关系嵌入维度
        edge_family_offsets: Optional[Dict[str, int]] = None,  # 关系族offset映射
        use_type_modulation: bool = True,  # 是否使用类别调制子类别（True: Q=(1+γ(T))⊙W^Q_t X+β(T), False: Q=W^Q_t X）
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / heads)
        self.heads = heads
        self.concat = concat
        self.last_layer = last_layer
        self.dropout = dropout
        self.heterogeneous = heterogeneous
        self.edge_family_offsets = edge_family_offsets or {}
        self.use_type_modulation = use_type_modulation  # 是否使用类别调制
        self.type_embed_dim = type_embed_dim  # 保存类别嵌入维度
        self.subtype_embed_dim = subtype_embed_dim  # 保存子类别嵌入维度
        self._warned_missing_heterogeneous_metadata = False

        # 基础线性层（用于非异质图模式或作为fallback）
        self.lin_key = Linear(dx, heads * self.df)
        self.lin_query = Linear(dx, heads * self.df)
        self.lin_value = Linear(dx, heads * self.df)

        if concat:
            self.lin_skip = Linear(dx, heads * self.df, bias=bias)
        else:
            self.lin_skip = Linear(dx, self.df, bias=bias)

        # 异质图相关嵌入和层
        if self.heterogeneous:
            # 1. 类别嵌入表 E^T (用于X)
            self.node_type_embed = nn.Embedding(num_node_types, type_embed_dim) if num_node_types > 0 else None
            
            # 2. 关系类型嵌入表 E^{s_i, s_j} (用于边)
            self.relation_type_embed = nn.Embedding(num_relation_types, relation_embed_dim) if num_relation_types > 0 else None
            
            # 3. 为每个类别类型创建不同的权重矩阵 W^Q_t, W^K_t, W^V_t
            if num_node_types > 0:
                self.lin_query_by_type = nn.ModuleList([
                    Linear(dx, heads * self.df) for _ in range(num_node_types)
                ])
                self.lin_key_by_type = nn.ModuleList([
                    Linear(dx, heads * self.df) for _ in range(num_node_types)
                ])
                self.lin_value_by_type = nn.ModuleList([
                    Linear(dx, heads * self.df) for _ in range(num_node_types)
                ])
            else:
                self.lin_query_by_type = None
                self.lin_key_by_type = None
                self.lin_value_by_type = None
            
            # 类别嵌入到dx维度的映射层
            if num_node_types > 0 and type_embed_dim > 0:
                self.type_emb_to_dx = Linear(type_embed_dim, dx)
            else:
                self.type_emb_to_dx = None
            
            # 关系嵌入到de维度的映射层
            if num_relation_types > 0 and relation_embed_dim > 0:
                self.relation_emb_to_de = Linear(relation_embed_dim, de)
            else:
                self.relation_emb_to_de = None
            
            # 边的融合：将 relation 嵌入与 edge_attr 融合，以同时利用
            # - 关系类别 (src_type, dst_type)，以及
            # - 边子类别（如一二三作）或仅类别（如隶属，无子类时 edge_attr 即该族状态）
            # 输出 de 维，供 FiLM 使用
            if num_relation_types > 0 and relation_embed_dim > 0:
                self.edge_fusion = Linear(relation_embed_dim + de, de)
            else:
                self.edge_fusion = None
            
            # 5. 子类别调制层（用于生成γ和β）
            if subtype_embed_dim > 0:
                self.subtype_gamma_Q = nn.Sequential(
                    Linear(subtype_embed_dim, dx),
                    nn.ReLU(),
                    Linear(dx, heads * self.df)
                )
                self.subtype_beta_Q = nn.Sequential(
                    Linear(subtype_embed_dim, dx),
                    nn.ReLU(),
                    Linear(dx, heads * self.df)
                )
                self.subtype_gamma_K = nn.Sequential(
                    Linear(subtype_embed_dim, dx),
                    nn.ReLU(),
                    Linear(dx, heads * self.df)
                )
                self.subtype_beta_K = nn.Sequential(
                    Linear(subtype_embed_dim, dx),
                    nn.ReLU(),
                    Linear(dx, heads * self.df)
                )
            else:
                self.subtype_gamma_Q = None
                self.subtype_beta_Q = None
                self.subtype_gamma_K = None
                self.subtype_beta_K = None

        # FiLM E to X: de = dx here as defined in lin_edge
        self.e_add = Linear(de, heads)
        self.e_mul = Linear(de, heads)

        # FiLM y to E
        self.y_e_mul = Linear(dy, de)
        self.y_e_add = Linear(dy, de)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        if self.last_layer:
            self.y_y = Linear(dy, dy)
            self.x_y = SparseXtoy(dx, dy)
            self.e_y = SparseEtoy(de, dy)
            self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        y: Tensor = None,
        batch: Tensor = None,
        # 异质图元数据
        node_type_ids: OptTensor = None,  # (N,) 类别ID
        node_subtype_ids: OptTensor = None,  # (N,) 子类别ID
        relation_type_ids: OptTensor = None,  # (E,) 关系类型ID (src_type, dst_type的组合)
        edge_family_ids: OptTensor = None,  # (E,) 关系族ID
        type_offsets: Optional[Dict[str, int]] = None,  # 节点类型offset映射
    ):
        r"""Runs the forward pass of the module.

        Args:
            x: 节点特征 (N, dx)
            edge_index: 边索引 (2, E)
            edge_attr: 边特征 (E, de)
            y: 图级特征 (bs, dy)
            batch: 批次信息 (N,)
            node_type_ids: 节点类别ID (N,)
            node_subtype_ids: 节点子类别ID (N,)
            relation_type_ids: 关系类型ID (E,)
            edge_family_ids: 关系族ID (E,)
            type_offsets: 节点类型offset映射
        """
        H, C = self.heads, self.df

        # 计算Q/K/V
        if self.heterogeneous and node_type_ids is not None and node_subtype_ids is not None:
            # 异质图模式：使用类别和子类别融合机制
            query, key, value = self._compute_qkv_heterogeneous(
                x, node_type_ids, node_subtype_ids
            )
        else:
            if self.heterogeneous and not self._warned_missing_heterogeneous_metadata:
                warnings.warn(
                    "HeterogeneousTransformerConv: missing node_type_ids/node_subtype_ids; "
                    "falling back to homogeneous QKV path.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_missing_heterogeneous_metadata = True
            # 同质图模式：使用原始方式
            query = self.lin_query(x).view(-1, H, C)
            key = self.lin_key(x).view(-1, H, C)
            value = self.lin_value(x).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor)
        out, new_edge_attr = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            index=edge_index[1],
            size=None,
            relation_type_ids=relation_type_ids,
            edge_family_ids=edge_family_ids,
        )

        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.df)
        else:
            out = out.mean(dim=1)

        x_r = self.lin_skip(x)
        out = out + x_r

        # Incorporate y to edge_attr
        batch = batch.long()
        batchE = batch[edge_index[0]]
        # Output _edge_attr
        new_edge_attr = new_edge_attr.flatten(start_dim=1)  # M, h * df (dx)
        new_edge_attr = self.e_out(new_edge_attr)  # M, de
        ye1 = self.y_e_add(y)  # M, de
        ye2 = self.y_e_mul(y)  # M, de
        new_edge_attr = ye1[batchE] + (ye2[batchE] + 1) * new_edge_attr  # M, de

        # Incorporate y to X
        yx1 = self.y_x_add(y)
        yx2 = self.y_x_mul(y)
        new_x = yx1[batch] + (yx2[batch] + 1) * out
        # Output X
        new_x = self.x_out(new_x)

        if self.last_layer:
            new_y = self.predict_graph(y, x, edge_index, edge_attr, batch)
        else:
            new_y = y

        return (new_x, new_edge_attr, new_y)

    def _compute_qkv_heterogeneous(
        self,
        x: Tensor,  # (N, dx) 子类别嵌入（lin_in_X 从 one-hot 得到）
        node_type_ids: Tensor,  # (N,) 类别ID
        node_subtype_ids: Tensor,  # (N,) 子类别ID
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """计算异质图的Q/K/V
        
        边关系预测本质上是子类别之间的关系，因此以子类别嵌入为基底，类别作为约束/调制：
        - X: 子类别嵌入（输入的 x，即 lin_in_X(one-hot(subtype))）
        - T: 类别嵌入，用于 (1) 选择 per-type 权重 W^Q_t (2) 可选：生成 γ、β 调制子类别表示
        实现: Q = (1 + γ_Q(T)) ⊙ W^Q_t X + β_Q(T)，即子类别为基底、类别调制
        """
        N, dx = x.shape
        H, C = self.heads, self.df

        # 1. 基底：子类别嵌入（边关系只和子类别有关）
        X = x  # (N, dx) 子类别嵌入，作为 Q/K/V 的基底

        # 2. 类别嵌入（用于选择 per-type 权重矩阵；若 type_embed_dim==subtype_embed_dim 则复用调制层做类别调制）
        type_emb = None
        if self.node_type_embed is not None:
            type_emb = self.node_type_embed(node_type_ids)  # (N, type_embed_dim)

        # 3. 按类别分组批量计算 Q/K/V（避免逐节点 Python 循环）
        qkv_dim = H * C
        query_flat = X.new_empty((N, qkv_dim))
        key_flat = X.new_empty((N, qkv_dim))
        value_flat = X.new_empty((N, qkv_dim))

        unique_types = torch.unique(node_type_ids)
        for t in unique_types.tolist():
            idx = (node_type_ids == t).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue

            X_t = X[idx]  # (N_t, dx)
            if (
                self.lin_query_by_type is not None
                and isinstance(t, int)
                and 0 <= t < len(self.lin_query_by_type)
            ):
                WQ_t = self.lin_query_by_type[t]
                WK_t = self.lin_key_by_type[t]
                WV_t = self.lin_value_by_type[t]
            else:
                WQ_t = self.lin_query
                WK_t = self.lin_key
                WV_t = self.lin_value

            WX_Q = WQ_t(X_t)  # (N_t, H*C)
            WX_K = WK_t(X_t)  # (N_t, H*C)
            WX_V = WV_t(X_t)  # (N_t, H*C)

            if (
                self.use_type_modulation
                and type_emb is not None
                and self.subtype_gamma_Q is not None
                and self.type_embed_dim == self.subtype_embed_dim
            ):
                type_emb_t = type_emb[idx]  # (N_t, type_embed_dim)
                gamma_Q = self.subtype_gamma_Q(type_emb_t)  # (N_t, H*C)
                beta_Q = self.subtype_beta_Q(type_emb_t)
                gamma_K = self.subtype_gamma_K(type_emb_t)
                beta_K = self.subtype_beta_K(type_emb_t)
                Q_t = (1 + gamma_Q) * WX_Q + beta_Q
                K_t = (1 + gamma_K) * WX_K + beta_K
            else:
                Q_t = WX_Q
                K_t = WX_K

            query_flat[idx] = Q_t
            key_flat[idx] = K_t
            value_flat[idx] = WX_V

        query = query_flat.view(N, H, C)
        key = key_flat.view(N, H, C)
        value = value_flat.view(N, H, C)

        return query, key, value

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages."""
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)

        msg_kwargs = self.inspector.distribute("message", coll_dict)
        out, edge_attr = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.distribute("aggregate", coll_dict)

        out = self.aggregate(out, **aggr_kwargs)

        update_kwargs = self.inspector.distribute("update", coll_dict)
        out = self.update(out, **update_kwargs)

        return (out, edge_attr)

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
        relation_type_ids: OptTensor = None,
        edge_family_ids: OptTensor = None,
    ) -> Tensor:
        """计算消息，实现relation-aware边嵌入和per-relation-family softmax"""
        M = query_i.shape[0]  # 边数
        H, C = self.heads, self.df

        # 1. 计算基础注意力分数 d(<Q_u, K_v>)
        Y = (query_i * key_j) / math.sqrt(self.df)  # (M, H, C)

        # 2. 边的融合 + FiLM 调制
        #    融合：关系类别 (src_type, dst_type) + 边信息（有子类时如一作/二作/三作，无子类时如隶属等该族状态）
        #    edge_attr 来自上一层的 projected 特征，已包含 one-hot 边类型/子类信息
        if self.heterogeneous and relation_type_ids is not None and self.relation_type_embed is not None:
            relation_emb = self.relation_type_embed(relation_type_ids)  # (M, relation_embed_dim)
            if self.edge_fusion is not None:
                # 边的融合：relation（类别级）+ edge_attr（子类或类别级边状态）
                edge_attr_enhanced = self.edge_fusion(
                    torch.cat([relation_emb, edge_attr], dim=-1)
                )  # (M, de)
            elif self.relation_emb_to_de is not None:
                edge_attr_enhanced = self.relation_emb_to_de(relation_emb)
            else:
                edge_attr_enhanced = edge_attr
            edge_attr_mul = self.e_mul(edge_attr_enhanced)
            edge_attr_add = self.e_add(edge_attr_enhanced)
        else:
            edge_attr_mul = self.e_mul(edge_attr)
            edge_attr_add = self.e_add(edge_attr)

        # luv = (1 + γ_α(e)) ⊙ d(<Q_u, K_v>) + β_α(e)
        Y = Y * (edge_attr_mul.unsqueeze(-1) + 1) + edge_attr_add.unsqueeze(-1)  # (M, H, C)

        # 3. Per-relation-family softmax + 外层归一化
        if self.heterogeneous and edge_family_ids is not None:
            alpha = self._softmax_per_relation_family(Y, index, ptr, size_i, edge_family_ids)
        else:
            # 非异质图模式：全局softmax
            alpha = softmax(Y.sum(-1), index, ptr, size_i)  # (M, H)

        self._alpha = alpha

        out = value_j  # (M, H, C)
        out = out * alpha.view(-1, H, 1)  # (M, H, C), out = weighted_V

        return (out, Y)

    def _softmax_per_relation_family(
        self,
        Y: Tensor,  # (M, H, C) 注意力分数
        index: Tensor,  # (M,) 目标节点索引
        ptr: OptTensor,
        size_i: Optional[int],
        edge_family_ids: Tensor,  # (M,) 关系族ID
    ) -> Tensor:
        """实现 per-relation-family softmax + 外层归一化（按目标节点入边和归一化）"""
        M, H, C = Y.shape
        Y_sum = Y.sum(-1)  # (M, H) 对特征维度求和

        # 获取唯一的关系族（若存在 -1 会在下方报错）
        unique_families = torch.unique(edge_family_ids)
        
        # 为每个关系族分别计算softmax
        alpha_per_family = torch.zeros_like(Y_sum)  # (M, H)
        
        for fam_id in unique_families:
            if fam_id.item() < 0:
                continue
            fam_mask = (edge_family_ids == fam_id)
            if not fam_mask.any():
                continue
            
            # 对该关系族的边进行softmax
            Y_fam = Y_sum[fam_mask]  # (M_fam, H)
            index_fam = index[fam_mask]  # (M_fam,)
            
            # 计算该关系族内的softmax
            alpha_fam = softmax(Y_fam, index_fam, None, size_i)  # (M_fam, H)
            alpha_per_family[fam_mask] = alpha_fam

        # 正常流程下不应出现未知 family（查询边合法、反向边已映射到正向族）
        unknown_mask = edge_family_ids < 0
        if unknown_mask.any():
            n_unknown = unknown_mask.sum().item()
            raise ValueError(
                f"Found {n_unknown} edge(s) with unknown family (edge_family_ids < 0). "
                "Check that all edges have (src_type, dst_type) or (dst_type, src_type) in fam_endpoints "
                "(e.g. reverse edges from to_undirected should be mapped in extract_edge_family_ids)."
            )

        # 外层归一化：对每个目标节点、每个 head，在其所有入边（跨关系族）上做归一化（除以和），
        # 使入边权重和为 1，避免关系多的节点权重过大。使用归一化而非第二次 softmax，
        # 因为 alpha_per_family 已是族内 softmax 得到的概率，再 softmax 会扭曲族间相对重要性。
        dim_size = size_i if size_i is not None else int(index.max().item()) + 1
        sum_per_node = scatter(
            alpha_per_family, index, dim=0, dim_size=dim_size, reduce="add"
        )  # (dim_size, H)
        alpha = alpha_per_family / (sum_per_node[index] + 1e-10)  # (M, H)

        return alpha

    def predict_graph(self, y, x, edge_index, edge_attr, batch):
        y = self.y_y(y)
        e_y = self.e_y(edge_index, edge_attr, batch, top_triu=True)
        x_y = self.x_y(x, batch)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)  # bs, dy

        return new_y

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.dx}, {self.df}, heads={self.heads}, "
            f"heterogeneous={self.heterogeneous})"
        )
