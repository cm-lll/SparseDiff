import time
import os
import os.path as osp
import math
import pickle
import json
import traceback

import torch
import wandb
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.conv_transformer_model import GraphTransformerConv
from diffusion.noise_schedule import (
    PredefinedNoiseScheduleDiscrete,
    MarginalUniformTransition,
)
from diffusion.heterogeneous_transition import HeterogeneousMarginalUniformTransition

from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from analysis.visualization import Visualizer
from sparse_diffusion import utils
from sparse_diffusion.diffusion import diffusion_utils
from sparse_diffusion.diffusion.sample_edges_utils import (
    get_computational_graph,
    mask_query_graph_from_comp_graph,
    sample_non_existing_edge_attr,
    condensed_to_matrix_index_batch,
    matrix_to_condensed_index_batch,
)
from sparse_diffusion.diffusion.sample_edges import (
    sample_query_edges,
    sample_non_existing_edges_batched,
    sample_non_existing_edges_batched_heterogeneous,
    sampled_condensed_indices_uniformly,
)
from sparse_diffusion.models.sign_pos_encoder import SignNetNodeEncoder


class DiscreteDenoisingDiffusion(pl.LightningModule):
    model_dtype = torch.float32
    best_val_nll = 1e8
    val_counter = 0
    start_epoch_time = None
    val_iterations = None

    def __init__(
        self,
        cfg,
        dataset_infos,
        train_metrics,
        extra_features,
        domain_features,
        val_sampling_metrics,
        test_sampling_metrics,
    ):
        super().__init__()

        self.in_dims = dataset_infos.input_dims
        self.out_dims = dataset_infos.output_dims
        self.use_charge = cfg.model.use_charge and self.out_dims.charge > 1
        self.node_dist = dataset_infos.nodes_dist
        self.extra_features = extra_features
        self.domain_features = domain_features
        self.sign_net = cfg.model.sign_net
        if not self.sign_net:
            cfg.model.sn_hidden_dim = 0

        # sparse settings
        self.edge_fraction = cfg.model.edge_fraction
        self.autoregressive = cfg.model.autoregressive

        self.cfg = cfg
        self.test_variance = cfg.general.test_variance
        self.dataset_info = dataset_infos
        self.visualization_tools = Visualizer(dataset_infos)
        self.name = cfg.general.name
        self.T = cfg.model.diffusion_steps

        self.train_loss = TrainLossDiscrete(cfg.model.lambda_train, self.edge_fraction, self.dataset_info)
        self.train_metrics = train_metrics
        self.val_sampling_metrics = val_sampling_metrics
        self.test_sampling_metrics = test_sampling_metrics

        # TODO: transform to torchmetrics.MetricCollection
        self.val_nll = NLL()
        # self.val_metrics = torchmetrics.MetricCollection([])
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.best_nll = 1e8
        self.best_epoch = 0

        # TODO: transform to torchmetrics.MetricCollection
        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        if self.use_charge:
            self.val_charge_kl = SumExceptBatchKL()
            self.val_charge_logp = SumExceptBatchMetric()
            self.test_charge_kl = SumExceptBatchKL()
            self.test_charge_logp = SumExceptBatchMetric()

        # 获取异质图相关参数
        heterogeneous = getattr(self.dataset_info, "heterogeneous", False)
        num_node_types = 0
        num_node_subtypes = 0
        num_relation_types = 0
        type_offsets = None
        edge_family_offsets = None
        
        if heterogeneous:
            node_type_names = getattr(self.dataset_info, "node_type_names", [])
            num_node_types = len(node_type_names) if node_type_names else 0
            num_node_subtypes = self.out_dims.X  # 所有子类别的总数
            num_relation_types = num_node_types * num_node_types if num_node_types > 0 else 0
            type_offsets = getattr(self.dataset_info, "type_offsets", None)
            edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", None)
            
            if hasattr(self, 'local_rank') and self.local_rank == 0:
                print(f"Initializing heterogeneous model:")
                print(f"  - num_node_types: {num_node_types}")
                print(f"  - num_node_subtypes: {num_node_subtypes}")
                print(f"  - num_relation_types: {num_relation_types}")
        
        self.model = GraphTransformerConv(
            n_layers=cfg.model.n_layers,
            input_dims=self.in_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.out_dims,
            sn_hidden_dim=cfg.model.sn_hidden_dim,
            output_y=cfg.model.output_y,
            dropout=cfg.model.dropout,
            heterogeneous=heterogeneous,
            num_node_types=num_node_types,
            num_node_subtypes=num_node_subtypes,
            num_relation_types=num_relation_types,
            type_embed_dim=getattr(cfg.model, "type_embed_dim", 64),
            subtype_embed_dim=getattr(cfg.model, "subtype_embed_dim", 64),
            relation_embed_dim=getattr(cfg.model, "relation_embed_dim", 64),
            edge_family_offsets=edge_family_offsets,
            type_offsets=type_offsets,
            use_type_modulation=getattr(cfg.model, "use_type_modulation", True),  # 是否使用类别调制子类别
        )

        # whether to use sign net
        if self.sign_net and cfg.model.extra_features == "all":
            self.sign_net = SignNetNodeEncoder(
                dataset_infos, cfg.model.sn_hidden_dim, cfg.model.num_eigenvectors
            )

        # whether to use scale layers
        self.scaling_layer = cfg.model.scaling_layer
        (
            self.node_scaling_layer,
            self.edge_scaling_layer,
            self.graph_scaling_layer,
        ) = self.get_scaling_layers()

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule,
            timesteps=cfg.model.diffusion_steps,
            skip=self.cfg.general.skip,
        )

        # Marginal transition
        node_types = self.dataset_info.node_types.float()
        x_marginals = node_types / torch.sum(node_types)

        edge_types = self.dataset_info.edge_types.float()
        e_marginals = edge_types / torch.sum(edge_types)

        if not self.use_charge:
            charge_marginals = node_types.new_zeros(0)
        else:
            charge_marginals = (
                self.dataset_info.charge_types * node_types[:, None]
            ).sum(dim=0)

        # 只在主进程打印（在on_fit_start中会再次打印）
        if hasattr(self, 'local_rank') and self.local_rank == 0:
            print(
                f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges"
            )
        
        # 检查是否为异质图模式
        self.heterogeneous = getattr(self.dataset_info, "heterogeneous", False)
        # 边采样：True=先采样存在性再采样子类型（与分层损失一致）；False=对所有类做一次 multinomial
        self.hierarchical_edge_sampling = getattr(cfg.model, "hierarchical_edge_sampling", self.heterogeneous)
        if self.heterogeneous and hasattr(self.dataset_info, "edge_family_marginals") and len(self.dataset_info.edge_family_marginals) > 0:
            # 异质图模式：使用关系族隔离的转移矩阵
            if hasattr(self, 'local_rank') and self.local_rank == 0:
                print("Using heterogeneous transition model with edge family isolation")
            self.transition_model = HeterogeneousMarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.out_dims.y,
                charge_marginals=charge_marginals,
                edge_family_marginals=getattr(self.dataset_info, "edge_family_marginals", None),
                edge_family_offsets=getattr(self.dataset_info, "edge_family_offsets", None),
            )
        else:
            # 同质图模式：使用原始转移矩阵
            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.out_dims.y,
                charge_marginals=charge_marginals,
            )

        self.limit_dist = utils.PlaceHolder(
            X=x_marginals,
            E=e_marginals,
            y=torch.ones(self.out_dims.y) / self.out_dims.y,
            charge=charge_marginals,
        )

        self.save_hyperparameters(ignore=["train_metrics", "sampling_metrics"])
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps

    def training_step(self, data, i):
        # The above code is using the Python debugger module `pdb` to set a breakpoint at a specific
        # line of code. When the code is executed, it will pause at that line and allow you to
        # interactively debug the program.
        if data.edge_index.numel() == 0:
            if hasattr(self, 'local_rank') and self.local_rank == 0:
                print("Found a batch with no edges. Skipping.")
            return
        # Map discrete classes to one hot encoding
        data = self.dataset_info.to_one_hot(data)

        sparse_noisy_data = self.apply_sparse_noise(data)
        # Sample the query edges and build the computational graph = union(noisy graph, query edges)
        # 检查是否为异质图模式，如果是，需要按关系族分别进行均匀采样
        if self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets") and len(self.dataset_info.edge_family_offsets) > 0:
            # 异质图模式：按关系族分别进行均匀采样
            # 参考原项目实现，但需要区分关系族
            # 获取关系族信息
            edge_family2id = getattr(self.dataset_info, "edge_family2id", {})
            id2edge_family = {v: k for k, v in edge_family2id.items()}
            fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
            edge_family_avg_edge_counts = getattr(self.dataset_info, "edge_family_avg_edge_counts", {})
            
            # 获取节点类型信息
            type_offsets = getattr(self.dataset_info, "type_offsets", {})
            node_type_names = getattr(self.dataset_info, "node_type_names", [])

            # True-edge mixing schedule (start high, decay to closer-to-real distribution)
            mix_start = float(getattr(self.cfg.train, "true_edge_mix_start", 1.0))
            mix_end = float(getattr(self.cfg.train, "true_edge_mix_end", 0.4))  # 从 0.2 提高到 0.4，保持更多真实边
            mix_decay_epochs = float(getattr(self.cfg.train, "true_edge_mix_decay_epochs", 100))  # 从 50 延长到 100，衰减更慢
            if mix_decay_epochs <= 0:
                mix_ratio = mix_end
            else:
                progress = min(max(self.current_epoch / mix_decay_epochs, 0.0), 1.0)
                mix_ratio = mix_start + (mix_end - mix_start) * progress
                mix_ratio = max(min(mix_ratio, max(mix_start, mix_end)), min(mix_start, mix_end))

            # Reduce pos-weight / focal strength when positives are heavily mixed in
            # 调整动态权重：当真实边减少时，更激进地增加负样本惩罚
            if hasattr(self, "train_loss"):
                self.train_loss.pos_weight_scale = max(0.2, 1.0 - 0.8 * mix_ratio)
                self.train_loss.focal_gamma_scale = max(0.5, 1.0 - 0.5 * mix_ratio)
                # 调整公式：初期保持较高惩罚，后期进一步增加
                self.train_loss.focal_alpha_neg_scale = max(0.7, 1.0 - 0.3 * mix_ratio)
            
            # 如果 type_offsets 不存在，尝试从 meta.json 推断（不加载 vocab.json）
            if not type_offsets and node_type_names:
                import os.path as osp
                import json
                vocab_path = osp.join(getattr(self.dataset_info, "vocab_path", ""), "vocab.json")
                if not vocab_path or not osp.exists(vocab_path):
                    if hasattr(self.dataset_info, "datamodule") and hasattr(self.dataset_info.datamodule, "inner"):
                        vocab_path = osp.join(self.dataset_info.datamodule.inner.processed_dir, "vocab.json")
                
                if osp.exists(vocab_path) and hasattr(self.dataset_info, "datamodule") and hasattr(self.dataset_info.datamodule, "inner"):
                    subgraph_dirs = [d for d in os.listdir(self.dataset_info.datamodule.inner.root) 
                                    if osp.isdir(osp.join(self.dataset_info.datamodule.inner.root, d)) and d.startswith("subgraph_")]
                    if len(subgraph_dirs) > 0:
                        meta0 = json.load(open(osp.join(self.dataset_info.datamodule.inner.root, subgraph_dirs[0], "meta.json"), "r", encoding="utf-8"))
                        schema_by_type = meta0.get("schema_by_type", {})
                        type_sizes = [len(schema_by_type.get(t, [])) for t in node_type_names]
                        type_offsets = {}
                        cur = 0
                        for t, size in zip(node_type_names, type_sizes):
                            type_offsets[t] = cur
                            cur += size
            
            # 为每个关系族分别生成查询边
            # 根据图片要求：|Eq| = km，其中 m 是该关系族的平均真实边数，k 是倍数（通过 edge_fraction 控制）
            all_query_edge_index_list = []
            all_query_edge_batch_list = []
            
            bs = int(data.batch.max() + 1)
            num_nodes_per_graph = data.ptr.diff()  # (bs,)
            node_t = data.x.argmax(dim=-1) if data.x.dim() > 1 else data.x  # (N,) - 全局子类别ID
            
            # 统计每个批次中每个关系族的真实边数（从原始数据中统计）
            edge_attr_discrete = data.edge_attr.argmax(dim=-1) if data.edge_attr.dim() > 1 else data.edge_attr  # (E,)
            edge_family_offsets = self.dataset_info.edge_family_offsets
            
            for fam_id, fam_name in id2edge_family.items():
                if fam_name not in fam_endpoints:
                    continue
                
                src_type = fam_endpoints[fam_name]["src_type"]
                dst_type = fam_endpoints[fam_name]["dst_type"]
                
                # 计算该关系族的offset范围
                offset = edge_family_offsets.get(fam_name, 0)
                next_offset = self.out_dims.E
                for _, other_offset in edge_family_offsets.items():
                    if other_offset > offset and other_offset < next_offset:
                        next_offset = other_offset
                
                # 统计每个批次中该关系族的真实边数 m_fam（从原始数据中统计）
                fam_edge_mask = (edge_attr_discrete >= offset) & (edge_attr_discrete < next_offset)  # (E,)
                fam_edge_index = data.edge_index[:, fam_edge_mask]  # (2, E_fam)
                if fam_edge_index.shape[1] > 0:
                    fam_edge_batch = data.batch[fam_edge_index[0]]  # (E_fam,)
                    unique_fam_batch, counts_fam = torch.unique(fam_edge_batch, sorted=True, return_counts=True)
                    num_fam_edges_per_batch = torch.zeros(bs, dtype=torch.long, device=self.device)
                    num_fam_edges_per_batch[unique_fam_batch] = counts_fam.long()
                else:
                    num_fam_edges_per_batch = torch.zeros(bs, dtype=torch.long, device=self.device)
                
                # 如果没有统计到真实边数，使用保存的平均值
                if fam_name in edge_family_avg_edge_counts:
                    avg_m_fam = edge_family_avg_edge_counts[fam_name]
                else:
                    # 如果没有保存的平均值，使用当前批次统计的值（作为近似）
                    avg_m_fam = num_fam_edges_per_batch.float().mean().item() if num_fam_edges_per_batch.sum() > 0 else 10.0
                
                # 计算每个批次中该关系族的 src_type 和 dst_type 节点数
                if type_offsets and src_type in type_offsets and dst_type in type_offsets:
                    src_offset = type_offsets[src_type]
                    dst_offset = type_offsets[dst_type]
                    
                    # 计算每个节点类型的 size
                    type_sizes = {}
                    sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                    for i, (t, off) in enumerate(sorted_types):
                        if i + 1 < len(sorted_types):
                            type_sizes[t] = sorted_types[i + 1][1] - off
                        else:
                            type_sizes[t] = self.out_dims.X - off
                    
                    src_size = type_sizes.get(src_type, 0)
                    dst_size = type_sizes.get(dst_type, 0)
                    
                    # 计算每个批次中 src_type 和 dst_type 的节点数
                    src_mask = (node_t >= src_offset) & (node_t < src_offset + src_size)  # (N,)
                    dst_mask = (node_t >= dst_offset) & (node_t < dst_offset + dst_size)  # (N,)
                    
                    # 为每个批次生成该关系族的查询边
                    for b in range(bs):
                        batch_mask = (data.batch == b)
                        batch_src_nodes = torch.where(src_mask & batch_mask)[0]  # 全局节点索引
                        batch_dst_nodes = torch.where(dst_mask & batch_mask)[0]  # 全局节点索引
                        
                        if len(batch_src_nodes) == 0 or len(batch_dst_nodes) == 0:
                            continue
                        
                        # 使用保存的平均真实边数 m_fam（各关系族保持一致）
                        m_fam = avg_m_fam
                        
                        # 计算该批次该关系族的可能边数（有向图：src_nodes * dst_nodes，排除自环）
                        num_src = len(batch_src_nodes)
                        num_dst = len(batch_dst_nodes)
                        num_fam_possible_edges = num_src * num_dst
                        if src_type == dst_type:
                            num_fam_possible_edges = num_src * num_dst - num_src
                        
                        # 根据图片要求：|Eq| = km，其中 k 是倍数（通过 edge_fraction 控制，各关系族保持一致）
                        k = self.edge_fraction  # 倍数
                        num_query_edges_fam = int(math.ceil(k * m_fam)) if m_fam > 0 else 0
                        num_query_edges_fam = min(num_query_edges_fam, num_fam_possible_edges)
                        
                        if num_query_edges_fam == 0:
                            continue
                        
                        # 使用 condensed_index 方式采样（与原项目一致，与采样时保持一致）
                        if src_type == dst_type:
                            # 同类型：使用上三角矩阵的 condensed_index（排除自环）
                            num_fam_nodes = num_src
                            max_condensed_value_fam = num_fam_nodes * (num_fam_nodes - 1) // 2
                            
                            if max_condensed_value_fam > 0 and num_query_edges_fam > 0:
                                num_query_edges_fam_tensor = torch.tensor([num_query_edges_fam], device=self.device, dtype=torch.long)
                                max_condensed_value_fam_tensor = torch.tensor([max_condensed_value_fam], device=self.device, dtype=torch.long)
                                
                                sampled_condensed_fam, _ = sampled_condensed_indices_uniformly(
                                    max_condensed_value=max_condensed_value_fam_tensor,
                                    num_edges_to_sample=num_query_edges_fam_tensor,
                                    return_mask=False
                                )
                                
                                # 将 condensed_index 转换为 matrix_index
                                fam_query_edge_index_local = condensed_to_matrix_index_batch(
                                    condensed_index=sampled_condensed_fam,
                                    num_nodes=torch.tensor([num_fam_nodes], device=self.device, dtype=torch.long),
                                    edge_batch=torch.zeros(len(sampled_condensed_fam), device=self.device, dtype=torch.long),
                                    ptr=torch.tensor([0, num_fam_nodes], device=self.device, dtype=torch.long),
                                ).long()
                                
                                # 边界检查：确保索引在有效范围内
                                # 注意：fam_query_edge_index_local 是相对于 num_fam_nodes 的局部索引
                                # 需要确保它不超过 batch_src_nodes 的长度
                                valid_mask = (fam_query_edge_index_local[0] >= 0) & (fam_query_edge_index_local[0] < num_fam_nodes) & \
                                            (fam_query_edge_index_local[1] >= 0) & (fam_query_edge_index_local[1] < num_fam_nodes) & \
                                            (fam_query_edge_index_local[0] < len(batch_src_nodes)) & \
                                            (fam_query_edge_index_local[1] < len(batch_src_nodes))
                                if not valid_mask.all():
                                    # 过滤无效索引
                                    fam_query_edge_index_local = fam_query_edge_index_local[:, valid_mask]
                                    if fam_query_edge_index_local.shape[1] == 0:
                                        continue
                                
                                # 将局部索引转换回全局节点索引
                                # 再次检查索引范围
                                if (fam_query_edge_index_local[0].max() >= len(batch_src_nodes)) or \
                                   (fam_query_edge_index_local[1].max() >= len(batch_src_nodes)):
                                    # 如果索引超出范围，跳过
                                    continue
                                
                                fam_query_edge_index = torch.stack([
                                    batch_src_nodes[fam_query_edge_index_local[0]],
                                    batch_src_nodes[fam_query_edge_index_local[1]]
                                ], dim=0)
                            else:
                                continue
                        else:
                            # 不同类型：直接使用 src*dst 的矩阵进行抽样
                            max_condensed_value_fam = num_src * num_dst
                            
                            if max_condensed_value_fam > 0 and num_query_edges_fam > 0:
                                num_query_edges_fam_tensor = torch.tensor([num_query_edges_fam], device=self.device, dtype=torch.long)
                                max_condensed_value_fam_tensor = torch.tensor([max_condensed_value_fam], device=self.device, dtype=torch.long)
                                
                                sampled_flat_indices, _ = sampled_condensed_indices_uniformly(
                                    max_condensed_value=max_condensed_value_fam_tensor,
                                    num_edges_to_sample=num_query_edges_fam_tensor,
                                    return_mask=False
                                )
                                
                                # 将展平的索引转换为 (src_idx, dst_idx) 的矩阵坐标
                                src_indices_local = sampled_flat_indices // num_dst
                                dst_indices_local = sampled_flat_indices % num_dst
                                
                                # 将局部索引转换回全局节点索引
                                fam_query_edge_index = torch.stack([
                                    batch_src_nodes[src_indices_local],
                                    batch_dst_nodes[dst_indices_local]
                                ], dim=0)
                            else:
                                continue
                        
                        if fam_query_edge_index.shape[1] > 0:
                            # Merge a proportion of true edges for this family into query edges
                            if fam_edge_index.shape[1] > 0:
                                fam_edge_batch = data.batch[fam_edge_index[0]]
                                batch_true_edges = fam_edge_index[:, fam_edge_batch == b]
                                if batch_true_edges.shape[1] > 0 and mix_ratio > 0:
                                    num_true = batch_true_edges.shape[1]
                                    if mix_ratio < 1.0:
                                        num_add = max(int(math.ceil(mix_ratio * num_true)), 1)
                                        perm = torch.randperm(num_true, device=self.device)[:num_add]
                                        batch_true_edges = batch_true_edges[:, perm]
                                    fam_query_edge_index = torch.cat(
                                        [fam_query_edge_index, batch_true_edges], dim=1
                                    )
                                    fam_query_edge_index = torch.unique(
                                        fam_query_edge_index, dim=1
                                    )

                            fam_query_edge_batch = torch.full(
                                (fam_query_edge_index.shape[1],),
                                b,
                                dtype=torch.long,
                                device=self.device,
                            )
                            all_query_edge_index_list.append(fam_query_edge_index)
                            all_query_edge_batch_list.append(fam_query_edge_batch)
            
            # 合并所有关系族的查询边
            if len(all_query_edge_index_list) > 0:
                triu_query_edge_index = torch.cat(all_query_edge_index_list, dim=1)  # (2, E_query)
                query_edge_batch = torch.cat(all_query_edge_batch_list)  # (E_query,)
            else:
                # 如果没有关系族信息，回退到全局采样
                triu_query_edge_index, query_edge_batch = sample_query_edges(
                    num_nodes_per_graph=num_nodes_per_graph, edge_proportion=self.edge_fraction
                )
        else:
            # 同质图模式：使用全局采样
            triu_query_edge_index, query_edge_batch = sample_query_edges(
                num_nodes_per_graph=data.ptr.diff(), edge_proportion=self.edge_fraction
            )

        query_mask, comp_edge_index, comp_edge_attr = get_computational_graph(
            triu_query_edge_index=triu_query_edge_index,
            clean_edge_index=sparse_noisy_data["edge_index_t"],
            clean_edge_attr=sparse_noisy_data["edge_attr_t"],
            heterogeneous=self.heterogeneous,
            for_message_passing=True,  # 训练时用于消息传递，需要双向信息流通
        )

        # pass sparse comp_graph to dense comp_graph for ease calculation
        sparse_noisy_data["comp_edge_index_t"] = comp_edge_index
        sparse_noisy_data["comp_edge_attr_t"] = comp_edge_attr
        sparse_pred = self.forward(sparse_noisy_data)
        
        # 异质图：限制节点预测只能在其所属类型的子类别范围内（训练时也要限制）
        if self.heterogeneous and hasattr(self.dataset_info, "type_offsets"):
            type_offsets = self.dataset_info.type_offsets
            if type_offsets and sparse_pred.node.numel() > 0:
                # 获取当前节点的子类别ID（从噪声图中）
                current_node_subtype = sparse_noisy_data["node_t"]
                if current_node_subtype.dim() > 1:
                    current_node_subtype = current_node_subtype.argmax(dim=-1)  # (N,)
                else:
                    current_node_subtype = current_node_subtype.long()  # (N,)
                
                num_nodes = current_node_subtype.shape[0]
                num_subtypes = self.out_dims.X
                node_type_mask = torch.zeros((num_nodes, num_subtypes), device=self.device)
                
                # 计算每个类型的size
                sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                type_sizes = {}
                for i, (t_name, off) in enumerate(sorted_types):
                    if i + 1 < len(sorted_types):
                        type_sizes[t_name] = sorted_types[i + 1][1] - off
                    else:
                        type_sizes[t_name] = num_subtypes - off
                
                # 为每个节点生成mask
                for t_name, offset in sorted_types:
                    type_size = type_sizes.get(t_name, 0)
                    if type_size <= 0:
                        continue
                    # 找到属于该类型的节点
                    if t_name == sorted_types[-1][0]:
                        # 最后一个类型
                        type_mask = current_node_subtype >= offset
                    else:
                        next_offset = sorted_types[sorted_types.index((t_name, offset)) + 1][1]
                        type_mask = (current_node_subtype >= offset) & (current_node_subtype < next_offset)
                    
                    if type_mask.any():
                        # 允许该类型范围内的所有子类别
                        node_type_mask[type_mask, offset:offset + type_size] = 1.0
                
                # 应用mask：将不属于当前节点类别的子类别的logits设为-inf
                # 这样softmax后这些类别的概率为0，不会影响损失计算
                node_type_mask_inv = 1.0 - node_type_mask  # (N, dx)
                sparse_pred.node = sparse_pred.node - node_type_mask_inv * 1e10  # 将不允许的类别设为-inf

        # Compute the loss on the query edges only
        sparse_pred.edge_attr = sparse_pred.edge_attr[query_mask]
        sparse_pred.edge_index = comp_edge_index[:, query_mask]

        # 异质图：限制边预测只能在其所属关系族的子类型范围内（训练时也要限制，与节点预测一致）
        if self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets") and len(getattr(self.dataset_info, "edge_family_offsets", {})) > 0:
            edge_family_offsets = self.dataset_info.edge_family_offsets
            edge_family2id = getattr(self.dataset_info, "edge_family2id", {})
            fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
            type_offsets = getattr(self.dataset_info, "type_offsets", {})
            
            if edge_family_offsets and fam_endpoints and type_offsets and sparse_pred.edge_attr.numel() > 0:
                # 获取查询边的节点类型（从 comp_edge_index 和节点子类别推断）
                query_edge_index = sparse_pred.edge_index  # (2, E_query)
                num_query_edges = query_edge_index.shape[1]
                
                # 获取当前节点的子类别ID（从噪声图中）
                current_node_subtype = sparse_noisy_data["node_t"]
                if current_node_subtype.dim() > 1:
                    current_node_subtype = current_node_subtype.argmax(dim=-1)  # (N,)
                else:
                    current_node_subtype = current_node_subtype.long()  # (N,)
                
                # 推断每个节点的类型
                node_type_ids = torch.zeros_like(current_node_subtype) - 1  # -1 表示未知
                sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                for i, (t_name, off) in enumerate(sorted_types):
                    if i + 1 < len(sorted_types):
                        next_offset = sorted_types[i + 1][1]
                        type_mask = (current_node_subtype >= off) & (current_node_subtype < next_offset)
                    else:
                        type_mask = current_node_subtype >= off
                    node_type_ids[type_mask] = i
                
                # 为每条查询边生成关系族 mask
                edge_family_mask = torch.zeros((num_query_edges, self.out_dims.E), device=self.device)
                
                for fam_name, endpoints in fam_endpoints.items():
                    if fam_name not in edge_family_offsets:
                        continue
                    
                    # 获取端点类型（fam_endpoints 的结构是 {fam_name: {"src_type": ..., "dst_type": ...}}）
                    src_type = endpoints.get("src_type", None)
                    dst_type = endpoints.get("dst_type", None)
                    if src_type is None or dst_type is None:
                        continue
                    
                    offset = edge_family_offsets[fam_name]
                    next_offset = self.out_dims.E
                    for other_fam_name, other_offset in edge_family_offsets.items():
                        if other_offset > offset and other_offset < next_offset:
                            next_offset = other_offset
                    
                    # 找到该关系族对应的节点类型索引
                    src_type_idx = None
                    dst_type_idx = None
                    for idx, (t_name, _) in enumerate(sorted_types):
                        if t_name == src_type:
                            src_type_idx = idx
                        if t_name == dst_type:
                            dst_type_idx = idx
                    
                    if src_type_idx is None or dst_type_idx is None:
                        continue
                    
                    # 找到属于该关系族的查询边
                    src_nodes = query_edge_index[0]  # (E_query,)
                    dst_nodes = query_edge_index[1]  # (E_query,)
                    src_types = node_type_ids[src_nodes]  # (E_query,)
                    dst_types = node_type_ids[dst_nodes]  # (E_query,)
                    fam_edge_mask = (src_types == src_type_idx) & (dst_types == dst_type_idx)  # (E_query,)
                    
                    if fam_edge_mask.any():
                        # 允许该关系族范围内的所有子类型（包括 no-edge）
                        edge_family_mask[fam_edge_mask, 0] = 1.0  # no-edge 始终允许
                        for gid in range(offset, next_offset):
                            edge_family_mask[fam_edge_mask, gid] = 1.0
                
                # 应用mask：将不属于当前边关系族的子类型的logits设为-inf
                # 这样softmax后这些类别的概率为0，不会影响损失计算
                edge_family_mask_inv = 1.0 - edge_family_mask  # (E_query, de)
                sparse_pred.edge_attr = sparse_pred.edge_attr - edge_family_mask_inv * 1e10  # 将不允许的类别设为-inf

        # mask true label for query edges
        # We have the true edge index at time 0, and the query edge index at time t. This function
        # merge the query edges and edge index at time 0, delete repeated one, and retune the mask
        # for the true attr of query edges
        (
            query_mask2,
            true_comp_edge_attr,
            true_comp_edge_index,
        ) = mask_query_graph_from_comp_graph(
            triu_query_edge_index=triu_query_edge_index,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            num_classes=self.out_dims.E,
            heterogeneous=self.heterogeneous,
            for_message_passing=True,  # 训练时用于消息传递，需要双向信息流通
        )

        query_true_edge_attr = true_comp_edge_attr[query_mask2]
        assert (
            true_comp_edge_index[:, query_mask2] - sparse_pred.edge_index == 0
        ).all()

        true_data = utils.SparsePlaceHolder(
            node=data.x,
            charge=data.charge,
            edge_attr=query_true_edge_attr,
            edge_index=sparse_pred.edge_index,
            y=data.y,
            batch=data.batch,
        )
        true_data.collapse()  # Map one-hot to discrete class
        # Loss calculation
        loss = self.train_loss.forward(
            pred=sparse_pred,
            true_data=true_data,
            log=i % self.log_every_steps == 0,
        )
        self.train_metrics(
            pred=sparse_pred, true_data=true_data, log=i % self.log_every_steps == 0
        )

        if i % 20 == 0:
            self.print(f"  [heartbeat] epoch {self.current_epoch} batch {i}")

        return {"loss": loss}

    def on_fit_start(self) -> None:
        if hasattr(self, 'local_rank') and self.local_rank == 0:
            print(
                f"Size of the input features:"
                f" X {self.in_dims.X}, E {self.in_dims.E}, charge {self.in_dims.charge}, y {self.in_dims.y}"
            )
        if self.local_rank == 0:
            utils.setup_wandb(
                self.cfg
            )  # Initialize wandb only on one process to log metrics only once
            # 异质图：将各关系族的正边比例 u1 记录到 wandb，便于监控
            if (
                self.heterogeneous
                and hasattr(self, "dataset_info")
                and hasattr(self.dataset_info, "edge_family_marginals")
                and self.dataset_info.edge_family_marginals
            ):
                u1_dict = {}
                for fam_name, marginals in self.dataset_info.edge_family_marginals.items():
                    if isinstance(marginals, torch.Tensor) and marginals.numel() > 0:
                        u0 = float(marginals[0].item())
                        u1 = max(0.0, min(1.0, 1.0 - u0))
                        u1_dict[f"init/正边比例_u1/{fam_name}"] = u1
                if u1_dict and wandb.run:
                    wandb.log(u1_dict, commit=False)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        epoch_loss = self.train_loss.log_epoch_metrics()
        # self.log("train_epoch/x_CE", epoch_loss["train_epoch/x_CE"], sync_dist=False)
        x_ce = epoch_loss.get("train_epoch/x_CE", -1)
        e_ce = epoch_loss.get("train_epoch/E_CE", -1)
        charge_ce = epoch_loss.get("train_epoch/charge_CE", -1)
        y_ce = epoch_loss.get("train_epoch/y_CE", -1)
        self.print(
            f"Epoch {self.current_epoch} finished: X: {x_ce :.2f} -- "
            f"E: {e_ce :.2f} --"
            f"charge: {charge_ce :.2f} --"
            f"y: {y_ce :.2f}"
        )
        epoch_node_metrics, epoch_edge_metrics = self.train_metrics.log_epoch_metrics(log_step=self.current_epoch)

        if wandb.run:
            wandb.log({"epoch": self.current_epoch}, commit=True)

        # Optionally export a sampled edge list (denoised graph) at a specific epoch
        export_epoch = int(getattr(self.cfg.train, "export_edge_list_epoch", -1))
        enable_val_sampling = getattr(self.cfg.general, "enable_val_sampling", False)
        if export_epoch >= 0 and enable_val_sampling:
            if hasattr(self, "local_rank") and self.local_rank != 0:
                return
            if self.current_epoch == export_epoch:
                try:
                    sample = self.sample_batch(
                        batch_id=0,
                        batch_size=1,
                        keep_chain=0,
                        number_chain_steps=self.number_chain_steps,
                        save_final=1,
                    )
                    self._export_edge_list_doc(sample, epoch=self.current_epoch)
                except Exception as exc:
                    print(f"[WARN] export_edge_list_doc failed: {exc}")

    def _export_edge_list_doc(self, generated_graphs, epoch: int) -> None:
        """Export a human-readable edge list for a sampled graph."""
        import os
        import json
        import os.path as osp

        # Build id -> label for edge types
        edge_label2id = getattr(self.dataset_info, "edge_label2id", {})
        id2edge_label = {v: k for k, v in edge_label2id.items()}

        # Build node subtype name map if available
        schema_by_type = getattr(self.dataset_info, "schema_by_type", None)
        if not schema_by_type and hasattr(self.dataset_info, "datamodule"):
            try:
                root = self.dataset_info.datamodule.inner.root
                subgraph_dirs = [d for d in os.listdir(root) if d.startswith("subgraph_")]
                if subgraph_dirs:
                    meta_path = osp.join(root, subgraph_dirs[0], "meta.json")
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    schema_by_type = meta.get("schema_by_type", None)
            except Exception:
                schema_by_type = None

        type_offsets = getattr(self.dataset_info, "type_offsets", {})
        node_type_names = getattr(self.dataset_info, "node_type_names", [])
        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])

        def _type_for_subtype(subtype_id: int):
            for i, (t_name, off) in enumerate(sorted_types):
                next_off = None
                for _, off2 in sorted_types:
                    if off2 > off and (next_off is None or off2 < next_off):
                        next_off = off2
                if next_off is None:
                    next_off = 10**9
                if off <= subtype_id < next_off:
                    return t_name, subtype_id - off
            return "Unknown", subtype_id

        # Use first graph in batch
        if hasattr(generated_graphs, "batch"):
            batch = generated_graphs.batch
            graph_id = int(batch.max().item()) if batch.numel() > 0 else 0
            mask = batch == graph_id
        else:
            graph_id = 0
            mask = None

        node = generated_graphs.node
        edge_index = generated_graphs.edge_index
        edge_attr = generated_graphs.edge_attr

        if mask is not None and node.numel() > 0:
            node = node[mask]
            # remap node indices to local 0..n-1
            local_map = {}
            idxs = mask.nonzero(as_tuple=False).view(-1).tolist()
            for new_i, old_i in enumerate(idxs):
                local_map[old_i] = new_i
            keep_edges = []
            for e in range(edge_index.shape[1]):
                u = int(edge_index[0, e].item())
                v = int(edge_index[1, e].item())
                if u in local_map and v in local_map:
                    keep_edges.append(e)
            if keep_edges:
                edge_index = edge_index[:, keep_edges]
                edge_attr = edge_attr[keep_edges]
                edge_index = torch.stack(
                    [
                        torch.tensor([local_map[int(u.item())] for u in edge_index[0]]),
                        torch.tensor([local_map[int(v.item())] for v in edge_index[1]]),
                    ],
                    dim=0,
                )

        # Build per-node labels without external IDs
        node_labels = []
        node_subtypes = []
        for i in range(node.shape[0]):
            subtype_id = int(node[i].item()) if node.numel() > 0 else 0
            t_name, local_sub = _type_for_subtype(subtype_id)
            subtype_name = None
            if schema_by_type and t_name in schema_by_type:
                if 0 <= local_sub < len(schema_by_type[t_name]):
                    subtype_name = schema_by_type[t_name][local_sub]
            label = f"{t_name}[{i}]"
            node_labels.append(label)
            node_subtypes.append((local_sub, subtype_name))

        # Write edge list
        out_dir = os.path.join(os.getcwd(), "edge_lists", f"epoch{epoch}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"graph_{graph_id}_edge_list.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"epoch={epoch} graph={graph_id} num_nodes={node.shape[0]} num_edges={edge_index.shape[1]}\n")
            f.write("Nodes:\n")
            for i, label in enumerate(node_labels):
                local_sub, subtype_name = node_subtypes[i]
                if subtype_name:
                    f.write(f"  {i}: {label} subtype={local_sub}({subtype_name})\n")
                else:
                    f.write(f"  {i}: {label} subtype={local_sub}\n")
            f.write("Edges:\n")
            for e in range(edge_index.shape[1]):
                et = int(edge_attr[e].item())
                if et == 0:
                    continue
                u = int(edge_index[0, e].item())
                v = int(edge_index[1, e].item())
                label = id2edge_label.get(et, f"edge_type_{et}")
                f.write(f"  {node_labels[u]} --{label}--> {node_labels[v]}\n")

    def on_validation_epoch_start(self) -> None:
        self._val_predicted_graphs_list = []
        # 用于边分布图「generate」：在「真实边属于该族」的条件下，统计模型预测的子类型分布，与训练指标一致
        if getattr(self, "dataset_info", None) and getattr(self.dataset_info, "edge_family_marginals", None):
            self._val_edge_pred_counts_by_family = {
                fam: torch.zeros(len(self.dataset_info.edge_family_marginals[fam]), device=self.device)
                for fam in self.dataset_info.edge_family_marginals
            }
        else:
            self._val_edge_pred_counts_by_family = {}
        val_metrics = [
            self.val_nll,
            self.val_X_kl,
            self.val_E_kl,
            self.val_X_logp,
            self.val_E_logp,
            self.val_sampling_metrics,
        ]
        if self.use_charge:
            val_metrics.extend([self.val_charge_kl, self.val_charge_logp])
        for metric in val_metrics:
            metric.reset()

    def validation_step(self, data, i):
        """
        The evaluation is made for the whole graph, not only the query edges.
        It neccecitates an iteration as in the sampling step
        """

        data = self.dataset_info.to_one_hot(data)
        sparse_noisy_data = self.apply_sparse_noise(data)
        ptr = data.ptr
        batch = data.batch

        # prepare sparse information
        num_nodes = ptr.diff().long()
        num_edges = (num_nodes * (num_nodes - 1) / 2).long()
        num_edges_ptr = torch.hstack(
            [torch.tensor([0]).to(self.device), num_edges.cumsum(-1)]
        ).long()

        # permute all edges
        num_edges_per_loop = torch.ceil(self.edge_fraction * num_edges)  # (bs, )
        len_loop = math.ceil(1.0 / self.edge_fraction)

        # 初始化变量（用于异质图模式和同质图模式）
        all_query_edge_index = None
        all_query_edge_batch = None
        all_condensed_index = None
        all_edge_batch = None
        all_edge_mask = None

        # 检查是否为异质图模式，如果是，需要按关系族分别进行均匀采样
        if self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets") and len(self.dataset_info.edge_family_offsets) > 0:
            # 异质图模式：为每个关系族分别进行均匀采样
            # 获取关系族信息
            edge_family2id = getattr(self.dataset_info, "edge_family2id", {})
            id2edge_family = {v: k for k, v in edge_family2id.items()}
            fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
            
            # 获取节点类型信息
            node_type2id = getattr(self.dataset_info, "node_type2id", {})
            type_offsets = getattr(self.dataset_info, "type_offsets", {})
            node_type_names = getattr(self.dataset_info, "node_type_names", [])
            
            # 获取边信息
            edge_attr_discrete = data.edge_attr.argmax(dim=-1) if data.edge_attr.dim() > 1 else data.edge_attr
            edge_family_offsets = self.dataset_info.edge_family_offsets
            edge_family_avg_edge_counts = getattr(self.dataset_info, "edge_family_avg_edge_counts", {})
            
            all_query_edge_index_list = []
            all_query_edge_batch_list = []
            
            bs = len(num_nodes)
            node_t = data.x.argmax(dim=-1) if data.x.dim() > 1 else data.x
            
            for fam_id, fam_name in id2edge_family.items():
                if fam_name not in fam_endpoints:
                    continue
                
                src_type = fam_endpoints[fam_name]["src_type"]
                dst_type = fam_endpoints[fam_name]["dst_type"]
                
                # 使用保存的平均真实边数 m_fam
                if fam_name in edge_family_avg_edge_counts:
                    avg_m_fam = edge_family_avg_edge_counts[fam_name]
                else:
                    avg_m_fam = 10.0
                
                # 计算每个批次中该关系族的 src_type 和 dst_type 节点数
                if type_offsets and src_type in type_offsets and dst_type in type_offsets:
                    src_offset = type_offsets[src_type]
                    dst_offset = type_offsets[dst_type]
                    
                    # 计算每个节点类型的 size
                    type_sizes = {}
                    sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                    for i, (t, off) in enumerate(sorted_types):
                        if i + 1 < len(sorted_types):
                            type_sizes[t] = sorted_types[i + 1][1] - off
                        else:
                            type_sizes[t] = self.out_dims.X - off
                    
                    src_size = type_sizes.get(src_type, 0)
                    dst_size = type_sizes.get(dst_type, 0)
                    
                    src_mask = (node_t >= src_offset) & (node_t < src_offset + src_size)
                    dst_mask = (node_t >= dst_offset) & (node_t < dst_offset + dst_size)
                    
                    # 为每个批次生成该关系族的查询边
                    for b in range(bs):
                        batch_mask = (batch == b)
                        batch_src_nodes = torch.where(src_mask & batch_mask)[0]
                        batch_dst_nodes = torch.where(dst_mask & batch_mask)[0]
                        
                        if len(batch_src_nodes) == 0 or len(batch_dst_nodes) == 0:
                            continue
                        
                        m_fam = avg_m_fam
                        num_src = len(batch_src_nodes)
                        num_dst = len(batch_dst_nodes)
                        
                        # 验证时使用全量可能边，多批预测后合并得到完整预测图（与完整去噪一致）；训练时仅用 k*m_fam 子集
                        num_fam_possible_edges = num_src * num_dst
                        if src_type == dst_type:
                            num_fam_possible_edges = num_src * num_dst - num_src
                        num_query_edges_fam = num_fam_possible_edges  # 验证：覆盖该关系族下所有可能 (u,v)，分 len_loop 批预测
                        
                        if num_query_edges_fam == 0:
                            continue
                        
                        # 使用 condensed_index 方式采样（验证时 num_query_edges_fam = num_fam_possible_edges，等价于取遍所有边）
                        if src_type == dst_type:
                            num_fam_nodes = num_src
                            max_condensed_value_fam = num_fam_nodes * (num_fam_nodes - 1) // 2
                            
                            if max_condensed_value_fam > 0 and num_query_edges_fam > 0:
                                num_query_edges_fam_tensor = torch.tensor([num_query_edges_fam], device=self.device, dtype=torch.long)
                                max_condensed_value_fam_tensor = torch.tensor([max_condensed_value_fam], device=self.device, dtype=torch.long)
                                
                                sampled_condensed_fam, _ = sampled_condensed_indices_uniformly(
                                    max_condensed_value=max_condensed_value_fam_tensor,
                                    num_edges_to_sample=num_query_edges_fam_tensor,
                                    return_mask=False
                                )
                                
                                fam_query_edge_index_local = condensed_to_matrix_index_batch(
                                    condensed_index=sampled_condensed_fam,
                                    num_nodes=torch.tensor([num_fam_nodes], device=self.device, dtype=torch.long),
                                    edge_batch=torch.zeros(len(sampled_condensed_fam), device=self.device, dtype=torch.long),
                                    ptr=torch.tensor([0, num_fam_nodes], device=self.device, dtype=torch.long),
                                ).long()
                                
                                fam_query_edge_index = torch.stack([
                                    batch_src_nodes[fam_query_edge_index_local[0]],
                                    batch_src_nodes[fam_query_edge_index_local[1]]
                                ], dim=0)
                            else:
                                continue
                        else:
                            max_condensed_value_fam = num_src * num_dst
                            
                            if max_condensed_value_fam > 0 and num_query_edges_fam > 0:
                                num_query_edges_fam_tensor = torch.tensor([num_query_edges_fam], device=self.device, dtype=torch.long)
                                max_condensed_value_fam_tensor = torch.tensor([max_condensed_value_fam], device=self.device, dtype=torch.long)
                                
                                sampled_flat_indices, _ = sampled_condensed_indices_uniformly(
                                    max_condensed_value=max_condensed_value_fam_tensor,
                                    num_edges_to_sample=num_query_edges_fam_tensor,
                                    return_mask=False
                                )
                                
                                src_indices_local = sampled_flat_indices // num_dst
                                dst_indices_local = sampled_flat_indices % num_dst
                                
                                fam_query_edge_index = torch.stack([
                                    batch_src_nodes[src_indices_local],
                                    batch_dst_nodes[dst_indices_local]
                                ], dim=0)
                            else:
                                continue
                        
                        if fam_query_edge_index.shape[1] > 0:
                            fam_query_edge_batch = torch.full((fam_query_edge_index.shape[1],), b, dtype=torch.long, device=self.device)
                            all_query_edge_index_list.append(fam_query_edge_index)
                            all_query_edge_batch_list.append(fam_query_edge_batch)
            
            # 合并所有关系族的查询边
            if len(all_query_edge_index_list) > 0:
                all_query_edge_index = torch.cat(all_query_edge_index_list, dim=1)
                all_query_edge_batch = torch.cat(all_query_edge_batch_list)
                
                # 计算每个批次的查询边数
                num_edges_per_batch = torch.zeros(bs, dtype=torch.long, device=self.device)
                for b in range(bs):
                    num_edges_per_batch[b] = (all_query_edge_batch == b).sum()
                
                num_edges_total = all_query_edge_index.shape[1]
                num_edges = num_edges_per_batch.max().unsqueeze(0) if num_edges_per_batch.max() > 0 else num_edges
            else:
                all_query_edge_index = None
                all_query_edge_batch = None
                num_edges = (num_nodes * (num_nodes - 1) / 2).long()
        
        # 如果不是异质图或没有关系族信息，使用全局采样
        if not (self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets") and len(self.dataset_info.edge_family_offsets) > 0) or (all_query_edge_index is None):
            (
                all_condensed_index,
                all_edge_batch,
                all_edge_mask,
            ) = sampled_condensed_indices_uniformly(
                max_condensed_value=num_edges,
                num_edges_to_sample=num_edges,
                return_mask=True,
            )
        all_edge_index, all_edge_attr, all_charge, all_nodes = (
            torch.zeros((2, 0), device=self.device, dtype=torch.long),
            torch.zeros((0, data.edge_attr.shape[-1]), device=self.device),
            torch.zeros(0, device=self.device, dtype=torch.long),
            torch.zeros(0, device=self.device, dtype=torch.long),
        )

        # 初始化 edges_to_keep_mask_sorted（用于最后一个循环的过滤）
        edges_to_keep_mask_sorted = None

        # make a loop for all query edges
        for i in range(len_loop):
            # 检查是否为异质图模式，使用不同的采样方式
            if self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets") and len(self.dataset_info.edge_family_offsets) > 0 and all_query_edge_index is not None:
                # 异质图模式：使用按关系族采样的查询边
                num_query_edges_total = all_query_edge_index.shape[1]
                
                # 计算每个循环要处理的边数（按edge_fraction比例）
                if i == 0:
                    # 第一次循环，打乱所有查询边（实现均匀采样）
                    perm = torch.randperm(num_query_edges_total, device=self.device)
                    all_query_edge_index = all_query_edge_index[:, perm]
                    all_query_edge_batch = all_query_edge_batch[perm]
                
                # 计算当前循环要采样的边索引范围
                num_query_edges_per_loop = int(math.ceil(num_query_edges_total * self.edge_fraction))
                start_idx = i * num_query_edges_per_loop
                end_idx = min((i + 1) * num_query_edges_per_loop, num_query_edges_total)
                
                if start_idx < num_query_edges_total:
                    triu_query_edge_index = all_query_edge_index[:, start_idx:end_idx]
                    query_edge_batch = all_query_edge_batch[start_idx:end_idx]
                else:
                    triu_query_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                    query_edge_batch = torch.empty((0,), dtype=torch.long, device=self.device)
            else:
                # 同质图模式：使用condensed索引采样
                # the last loop might have less edges, we need to make sure that each loop has the same number of edges
                if i == len_loop - 1:
                    edges_to_consider_mask = all_edge_mask >= (
                        num_edges[all_edge_batch] - num_edges_per_loop[all_edge_batch]
                    )
                    edges_to_keep_mask = torch.logical_and(
                        all_edge_mask >= num_edges_per_loop[all_edge_batch] * i,
                        all_edge_mask < num_edges_per_loop[all_edge_batch] * (i + 1),
                    )

                    triu_query_edge_index_condensed = all_condensed_index[edges_to_consider_mask]
                    query_edge_batch = all_edge_batch[edges_to_consider_mask]
                    condensed_query_edge_index = (
                        triu_query_edge_index_condensed + num_edges_ptr[query_edge_batch]
                    )
                    condensed_query_edge_index, condensed_query_edge_index_argsort = (
                        condensed_query_edge_index.sort()
                    )
                    edges_to_keep_mask_sorted = edges_to_keep_mask[edges_to_consider_mask][
                        condensed_query_edge_index_argsort
                    ]
                else:
                    edges_to_consider_mask = torch.logical_and(
                        all_edge_mask >= num_edges_per_loop[all_edge_batch] * i,
                        all_edge_mask < num_edges_per_loop[all_edge_batch] * (i + 1),
                    )

                # get query edges and pass to matrix index
                triu_query_edge_index_condensed = all_condensed_index[edges_to_consider_mask]
                query_edge_batch = all_edge_batch[edges_to_consider_mask]
                # the order of edges does not change
                triu_query_edge_index = condensed_to_matrix_index_batch(
                    condensed_index=triu_query_edge_index_condensed,
                    num_nodes=num_nodes,
                    edge_batch=query_edge_batch,
                    ptr=ptr,
                ).long()
            query_mask, comp_edge_index, comp_edge_attr = get_computational_graph(
                triu_query_edge_index=triu_query_edge_index,
                clean_edge_index=sparse_noisy_data["edge_index_t"],
                clean_edge_attr=sparse_noisy_data["edge_attr_t"],
                heterogeneous=self.heterogeneous,
                for_message_passing=True,  # 验证时也用于消息传递，需要双向信息流通
            )

            # pass sparse comp_graph to dense comp_graph for ease calculation
            sparse_noisy_data["comp_edge_index_t"] = comp_edge_index
            sparse_noisy_data["comp_edge_attr_t"] = comp_edge_attr
            sparse_pred = self.forward(sparse_noisy_data)
            all_node = sparse_pred.node
            all_charge = sparse_pred.charge
            new_edge_attr = sparse_pred.edge_attr[query_mask]
            new_edge_index = comp_edge_index[:, query_mask]

            # Heterogeneous graphs are directed: keep edge direction.
            # Homogeneous graphs keep legacy upper-triangle behavior.
            if not self.heterogeneous:
                new_edge_index, new_edge_attr = utils.undirected_to_directed(
                    new_edge_index, new_edge_attr
                )

            if i == len_loop - 1:
                new_edge_batch = batch[new_edge_index[0]]
                if self.heterogeneous:
                    # Directed heterogeneous edges: sort by (batch, u, v) directly.
                    new_edge_index_no_batch = new_edge_index - ptr[new_edge_batch]
                    max_nodes = int(num_nodes.max().item()) if num_nodes.numel() > 0 else 1
                    sort_key = (
                        new_edge_batch.to(torch.int64) * (max_nodes * max_nodes)
                        + new_edge_index_no_batch[0].to(torch.int64) * max_nodes
                        + new_edge_index_no_batch[1].to(torch.int64)
                    )
                    _, new_condensed_edge_index_argsort = sort_key.sort()
                else:
                    new_edge_index_no_batch = new_edge_index - ptr[new_edge_batch]
                    new_condensed_edge_index = matrix_to_condensed_index_batch(
                        matrix_index=new_edge_index_no_batch,
                        num_nodes=num_nodes,
                        edge_batch=new_edge_batch,
                    )
                    new_condensed_edge_index = (
                        new_condensed_edge_index + num_edges_ptr[new_edge_batch]
                    )
                    new_condensed_edge_index, new_condensed_edge_index_argsort = (
                        new_condensed_edge_index.sort()
                    )
                new_edge_attr = new_edge_attr[new_condensed_edge_index_argsort]
                new_edge_index = new_edge_index[:, new_condensed_edge_index_argsort]

                # 只在同质图模式下使用 edges_to_keep_mask_sorted（异质图模式下不需要过滤）
                if edges_to_keep_mask_sorted is not None:
                    new_edge_attr = new_edge_attr[edges_to_keep_mask_sorted]
                    new_edge_index = new_edge_index[:, edges_to_keep_mask_sorted]

            all_edge_index = torch.hstack([all_edge_index, new_edge_index])
            all_edge_attr = torch.vstack([all_edge_attr, new_edge_attr])

        # to dense
        dense_pred, node_mask = utils.to_dense(
            x=all_node,
            edge_index=all_edge_index,
            edge_attr=all_edge_attr,
            batch=sparse_pred.batch,
            charge=all_charge,
        )
        dense_original, _ = utils.to_dense(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
            charge=data.charge,
        )
        noisy_data = utils.densify_noisy_data(sparse_noisy_data)

        # Validation diagnostics: compare predicted vs true distributions
        if i == 0 and getattr(self, "local_rank", 0) == 0:
            try:
                print("[VAL-STAT] diagnostics start", flush=True)
                with torch.no_grad():
                    node_mask_bool = node_mask.bool()
                    num_nodes = node_mask_bool.size(1)
                    eye = torch.eye(num_nodes, device=node_mask.device).bool().unsqueeze(0)
                    edge_mask = node_mask_bool.unsqueeze(2) & node_mask_bool.unsqueeze(1) & (~eye)

                    # True node labels
                    true_node = dense_original.X
                    true_node_labels = true_node.argmax(dim=-1) if true_node.dim() == 3 else true_node
                    true_node_flat = true_node_labels[node_mask_bool]

                    # Pred node labels/probs
                    pred_node_logits = dense_pred.X
                    pred_node_labels = (
                        pred_node_logits.argmax(dim=-1)
                        if pred_node_logits.dim() == 3
                        else pred_node_logits
                    )
                    pred_node_flat = pred_node_labels[node_mask_bool]

                    num_node_classes = (
                        pred_node_logits.size(-1)
                        if pred_node_logits.dim() == 3
                        else int(self.out_dims.X)
                    )
                    true_node_counts = torch.bincount(
                        true_node_flat.long(), minlength=num_node_classes
                    ).float()
                    pred_node_counts = torch.bincount(
                        pred_node_flat.long(), minlength=num_node_classes
                    ).float()
                    true_node_frac = true_node_counts / true_node_counts.sum().clamp(min=1.0)
                    pred_node_frac = pred_node_counts / pred_node_counts.sum().clamp(min=1.0)

                    # True edge labels
                    true_edge = dense_original.E
                    true_edge_labels = true_edge.argmax(dim=-1) if true_edge.dim() == 4 else true_edge
                    true_edge_flat = true_edge_labels[edge_mask]

                    # Pred edge labels/probs
                    pred_edge_logits = dense_pred.E
                    pred_edge_labels = (
                        pred_edge_logits.argmax(dim=-1)
                        if pred_edge_logits.dim() == 4
                        else pred_edge_logits
                    )
                    pred_edge_flat = pred_edge_labels[edge_mask]
                    num_edge_classes = (
                        pred_edge_logits.size(-1)
                        if pred_edge_logits.dim() == 4
                        else int(self.out_dims.E)
                    )
                    true_edge_counts = torch.bincount(
                        true_edge_flat.long(), minlength=num_edge_classes
                    ).float()
                    pred_edge_counts = torch.bincount(
                        pred_edge_flat.long(), minlength=num_edge_classes
                    ).float()
                    true_edge_frac = true_edge_counts / true_edge_counts.sum().clamp(min=1.0)
                    pred_edge_frac = pred_edge_counts / pred_edge_counts.sum().clamp(min=1.0)

                    pred_edge_noedge_prob = None
                    if pred_edge_logits.dim() == 4:
                        pred_edge_probs = torch.softmax(pred_edge_logits, dim=-1)
                        pred_edge_noedge_prob = pred_edge_probs[..., 0][edge_mask].mean().item()

                    true_noedge_ratio = true_edge_frac[0].item() if true_edge_frac.numel() > 0 else -1.0
                    pred_noedge_ratio = pred_edge_frac[0].item() if pred_edge_frac.numel() > 0 else -1.0

                    print(
                        f"[VAL-STAT] node_dist_true={true_node_frac.tolist()} "
                        f"node_dist_pred={pred_node_frac.tolist()}",
                        flush=True,
                    )
                    print(
                        f"[VAL-STAT] edge_dist_true={true_edge_frac.tolist()} "
                        f"edge_dist_pred={pred_edge_frac.tolist()} "
                        f"no_edge_true={true_noedge_ratio:.4f} "
                        f"no_edge_pred={pred_noedge_ratio:.4f} "
                        f"no_edge_prob={pred_edge_noedge_prob if pred_edge_noedge_prob is not None else -1:.4f}",
                        flush=True,
                    )

                    if wandb.run:
                        log_dict = {
                            "val/no_edge_ratio_true": true_noedge_ratio,
                            "val/no_edge_ratio_pred": pred_noedge_ratio,
                        }
                        if pred_edge_noedge_prob is not None:
                            log_dict["val/no_edge_prob_pred"] = pred_edge_noedge_prob
                        for idx, val in enumerate(true_node_frac.tolist()):
                            log_dict[f"val/node_type_frac_true/{idx}"] = val
                        for idx, val in enumerate(pred_node_frac.tolist()):
                            log_dict[f"val/node_type_frac_pred/{idx}"] = val
                        for idx, val in enumerate(true_edge_frac.tolist()):
                            log_dict[f"val/edge_type_frac_true/{idx}"] = val
                        for idx, val in enumerate(pred_edge_frac.tolist()):
                            log_dict[f"val/edge_type_frac_pred/{idx}"] = val

                        # Per-node-type subtype distributions (within each type)
                        type_offsets = getattr(self.dataset_info, "type_offsets", {})
                        node_type_names = getattr(self.dataset_info, "node_type_names", [])
                        if type_offsets and node_type_names:
                            sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                            for t_idx, (t_name, t_off) in enumerate(sorted_types):
                                next_off = num_node_classes
                                for _, off2 in sorted_types:
                                    if off2 > t_off and off2 < next_off:
                                        next_off = off2
                                # true/pred within this type
                                true_mask_t = (true_node_flat >= t_off) & (true_node_flat < next_off)
                                pred_mask_t = (pred_node_flat >= t_off) & (pred_node_flat < next_off)
                                if true_mask_t.any():
                                    true_counts_t = torch.bincount(
                                        (true_node_flat[true_mask_t] - t_off).long(),
                                        minlength=max(next_off - t_off, 1),
                                    ).float()
                                    true_frac_t = true_counts_t / true_counts_t.sum().clamp(min=1.0)
                                    for s_idx, val in enumerate(true_frac_t.tolist()):
                                        log_dict[f"val/node_subtype_frac_true/{t_name}/{s_idx}"] = val
                                if pred_mask_t.any():
                                    pred_counts_t = torch.bincount(
                                        (pred_node_flat[pred_mask_t] - t_off).long(),
                                        minlength=max(next_off - t_off, 1),
                                    ).float()
                                    pred_frac_t = pred_counts_t / pred_counts_t.sum().clamp(min=1.0)
                                    for s_idx, val in enumerate(pred_frac_t.tolist()):
                                        log_dict[f"val/node_subtype_frac_pred/{t_name}/{s_idx}"] = val

                        # Per-edge-family subtype distributions (explicit edges only)
                        edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", {})
                        edge_family2id = getattr(self.dataset_info, "edge_family2id", {})
                        if edge_family_offsets and edge_family2id:
                            fam_sorted = sorted(edge_family_offsets.items(), key=lambda x: x[1])
                            for fam_name, fam_off in fam_sorted:
                                next_off = num_edge_classes
                                for _, off2 in fam_sorted:
                                    if off2 > fam_off and off2 < next_off:
                                        next_off = off2
                                true_mask_f = (true_edge_flat >= fam_off) & (true_edge_flat < next_off)
                                pred_mask_f = (pred_edge_flat >= fam_off) & (pred_edge_flat < next_off)
                                if true_mask_f.any():
                                    true_counts_f = torch.bincount(
                                        (true_edge_flat[true_mask_f] - fam_off).long(),
                                        minlength=max(next_off - fam_off, 1),
                                    ).float()
                                    true_frac_f = true_counts_f / true_counts_f.sum().clamp(min=1.0)
                                    for s_idx, val in enumerate(true_frac_f.tolist()):
                                        log_dict[f"val/edge_subtype_frac_true/{fam_name}/{s_idx}"] = val
                                if pred_mask_f.any():
                                    pred_counts_f = torch.bincount(
                                        (pred_edge_flat[pred_mask_f] - fam_off).long(),
                                        minlength=max(next_off - fam_off, 1),
                                    ).float()
                                    pred_frac_f = pred_counts_f / pred_counts_f.sum().clamp(min=1.0)
                                    for s_idx, val in enumerate(pred_frac_f.tolist()):
                                        log_dict[f"val/edge_subtype_frac_pred/{fam_name}/{s_idx}"] = val

                                # Per-family positive accuracy (explicit edges only)
                                if true_mask_f.any():
                                    fam_acc = (
                                        pred_edge_flat[true_mask_f] == true_edge_flat[true_mask_f]
                                    ).float().mean()
                                    log_dict[f"val/E_pos_acc/{fam_name}"] = fam_acc
                                else:
                                    log_dict[f"val/E_pos_acc/{fam_name}"] = -1

                        # 节点/边分布柱状图统一在 on_validation_epoch_end 里由 val_sampling_metrics.compute_all_metrics 记录，此处只记录标量，避免 wandb 出现两个相同图表
                        wandb.log(log_dict, commit=False)
            except Exception as _e:
                print(f"[VAL-STAT] skipped due to error: {_e}", flush=True)
                traceback.print_exc()

        # 用当前 batch 的预测构造「预测图」：直接使用模型去噪预测的边集（all_edge_index 中预测非 no-edge 的边），与真实图同口径对比
        try:
            with torch.no_grad():
                node_mask_bool = node_mask.bool()
                pred_node_flat = (dense_pred.X[node_mask_bool].argmax(dim=-1) if dense_pred.X.dim() == 3 else dense_pred.X[node_mask_bool].long())
                ptr = getattr(data, "ptr", None)
                if ptr is None:
                    ptr = torch.cat([torch.tensor([0], device=data.batch.device), torch.unique(data.batch, return_counts=True)[1].cumsum(0)])
                # 从模型预测的边集中取出「预测为有边」的边（argmax != 0），作为预测图结构
                all_edge_discrete = all_edge_attr.argmax(dim=-1) if all_edge_attr.dim() > 1 else all_edge_attr.long()
                pred_edge_mask = (all_edge_discrete != 0)
                pred_edge_index = all_edge_index[:, pred_edge_mask].clone()
                pred_edge_attr_flat = all_edge_discrete[pred_edge_mask].clone()
                n_graphs = int(data.batch.max().item()) + 1
                y = getattr(data, "y", None)
                if y is not None and y.numel() > 0:
                    y = y.to(pred_node_flat.device)
                else:
                    y = torch.empty(n_graphs, 0, device=pred_node_flat.device)
                charge = getattr(data, "charge", None)
                if charge is not None and charge.numel() > 0:
                    charge = charge.to(pred_node_flat.device)
                else:
                    charge = torch.empty(pred_node_flat.shape[0], 0, device=pred_node_flat.device)
                pred_ph = utils.SparsePlaceHolder(
                    node=pred_node_flat,
                    edge_index=pred_edge_index,
                    edge_attr=pred_edge_attr_flat,
                    y=y,
                    batch=data.batch.clone(),
                    ptr=ptr.to(pred_node_flat.device) if ptr is not None else None,
                    charge=charge,
                )
                self._val_predicted_graphs_list.append(pred_ph)
                # 统计「真实边属于某族」时，模型预测的子类型分布，供边分布图 generate 列使用（仍用真实边位置在 dense_pred 上的预测）
                if getattr(self, "_val_edge_pred_counts_by_family", None) and getattr(self.dataset_info, "edge_family_offsets", None):
                    batch_idx = data.batch[data.edge_index[0]]
                    offset = ptr[batch_idx].to(data.edge_index.device)
                    i_local = data.edge_index[0] - offset
                    j_local = data.edge_index[1] - offset
                    pred_edge_attr_at_real = dense_pred.E[batch_idx, i_local, j_local, :].argmax(dim=-1)
                    true_edge_attr = dense_original.E[batch_idx, i_local, j_local, :].argmax(dim=-1)
                    fam_offsets = self.dataset_info.edge_family_offsets
                    for fam_name, counts in self._val_edge_pred_counts_by_family.items():
                        off = fam_offsets.get(fam_name, 0)
                        n_sub = counts.numel()
                        end = off + n_sub
                        true_in_fam = (true_edge_attr >= off) & (true_edge_attr < end)
                        pred_in_fam = (pred_edge_attr_at_real >= off) & (pred_edge_attr_at_real < end)
                        both = true_in_fam & pred_in_fam
                        if both.any():
                            pred_local = (pred_edge_attr_at_real[both] - off + 1).clamp(min=1, max=n_sub - 1)
                            for j in range(1, n_sub):
                                self._val_edge_pred_counts_by_family[fam_name][j] += (pred_local == j).sum().item()
        except Exception as _e:
            if hasattr(self, "local_rank") and self.local_rank == 0:
                print(f"[VAL-PRED-GRAPH] build failed: {_e}", flush=True)

        nll = self.compute_val_loss(
            dense_pred,
            noisy_data,
            dense_original.X,
            dense_original.E,
            dense_original.y,
            node_mask,
            charge=dense_original.charge,
            test=False,
        )

        return {"loss": nll}

    def on_validation_epoch_end(self) -> None:
        # 检查metrics是否已更新，避免在update之前调用compute
        metrics = []
        if self.val_nll.total_samples > 0:
            metrics.append(self.val_nll.compute())
        else:
            metrics.append(-1)
        
        if self.val_X_kl.total_samples > 0:
            metrics.append(self.val_X_kl.compute() * self.T)
        else:
            metrics.append(-1)
        
        if self.val_E_kl.total_samples > 0:
            metrics.append(self.val_E_kl.compute() * self.T)
        else:
            metrics.append(-1)
        
        if self.val_X_logp.total_samples > 0:
            metrics.append(self.val_X_logp.compute())
        else:
            metrics.append(-1)
        
        if self.val_E_logp.total_samples > 0:
            metrics.append(self.val_E_logp.compute())
        else:
            metrics.append(-1)

        if self.use_charge:
            if self.val_charge_kl.total_samples > 0:
                metrics.append(self.val_charge_kl.compute() * self.T)
            else:
                metrics.append(-1)
            if self.val_charge_logp.total_samples > 0:
                metrics.append(self.val_charge_logp.compute())
            else:
                metrics.append(-1)
        else:
            metrics += [-1, -1]

        val_nll_value = metrics[0] if metrics[0] != -1 else float('inf')
        if val_nll_value != -1 and val_nll_value < self.best_nll:
            self.best_epoch = self.current_epoch
            self.best_nll = val_nll_value
        metrics += [self.best_epoch, self.best_nll]

        if wandb.run:
            # 处理NaN值，避免wandb报错
            def is_valid(val):
                """检查值是否有效（非NaN且非Inf）"""
                if val is None:
                    return False
                try:
                    val_tensor = torch.tensor(float(val))
                    return not (torch.isnan(val_tensor).item() or torch.isinf(val_tensor).item())
                except (ValueError, TypeError):
                    return False
            
            log_dict = {}
            # 只记录有效的数值
            if is_valid(metrics[0]):
                log_dict["val/epoch_NLL"] = float(metrics[0])
            if is_valid(metrics[1]):
                log_dict["val/X_kl"] = float(metrics[1])
            if is_valid(metrics[2]):
                log_dict["val/E_kl"] = float(metrics[2])
            if len(metrics) > 3 and is_valid(metrics[3]):
                log_dict["val/X_logp"] = float(metrics[3])
            if len(metrics) > 4 and is_valid(metrics[4]):
                log_dict["val/E_logp"] = float(metrics[4])
            if len(metrics) > 5 and is_valid(metrics[5]):
                log_dict["val/charge_kl"] = float(metrics[5])
            if len(metrics) > 6 and is_valid(metrics[6]):
                log_dict["val/charge_logp"] = float(metrics[6])
            # best_nll_epoch 总是整数，不需要检查
            if len(metrics) > 7:
                log_dict["val/best_nll_epoch"] = int(metrics[7])
            if len(metrics) > 8 and is_valid(metrics[8]):
                log_dict["val/best_nll"] = float(metrics[8])
            
            wandb.log(log_dict, commit=False)

        self.print(
            f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
            f"Val Edge type KL: {metrics[2] :.2f}",
        )

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0] if metrics[0] != -1 else float('inf')
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print(
            "Val loss: %.4f \t Best val loss:  %.4f\n" % (val_nll, self.best_val_nll)
        )

        self.val_counter += 1
        # 用验证集上的预测图算与采样同款的指标（不跑采样也能看到 NumNodesW1, NodeTypesTV, EdgeTypesTV, Disconnected, MeanComponents, MaxComponents 及子类别分布图）
        # 所有 rank 都跑 compute_all_metrics，避免只有 rank 0 算导致 DDP 同步时卡住；wandb 只由 rank 0 写（在 compute_all_metrics 内部已按 local_rank 控制）
        if getattr(self, "_val_predicted_graphs_list", None) and len(self._val_predicted_graphs_list) > 0:
            try:
                self.val_sampling_metrics.reset()
                pred_concat = utils.concat_sparse_graphs(self._val_predicted_graphs_list)
                true_conditioned_pred_edge_counts = getattr(self, "_val_edge_pred_counts_by_family", None) or {}
                self.val_sampling_metrics.compute_all_metrics(
                    pred_concat,
                    self.current_epoch,
                    local_rank=getattr(self, "local_rank", 0),
                    key_suffix="_pred",
                    chart_title_suffix=" 预测",
                    true_conditioned_pred_edge_counts=true_conditioned_pred_edge_counts,
                )
            except Exception as _e:
                if hasattr(self, "local_rank") and self.local_rank == 0:
                    print(f"[VAL-PRED-METRICS] failed: {_e}", flush=True)
            self._val_predicted_graphs_list = []

        enable_val_sampling = getattr(self.cfg.general, "enable_val_sampling", False)
        if enable_val_sampling and self.val_counter % self.cfg.general.sample_every_val == 0:
            if hasattr(self, 'local_rank') and self.local_rank == 0:
                print("Starting to sample")
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save  # multi gpu operation
            samples_left_to_generate = math.ceil(
                samples_left_to_generate / max(self._trainer.num_devices, 1)
            )
            self.print(
                f"Samples to generate: {samples_left_to_generate} for each of the {max(self._trainer.num_devices, 1)} devices"
            )
            if hasattr(self, 'local_rank') and self.local_rank == 0:
                print(f"Sampling start on GR{self.global_rank}")
                print(
                    "multi-gpu metrics for uniqueness is not accurate in the validation step."
                )
            
            start = time.time()  # 记录采样开始时间
            generated_graphs = []
            ident = 0
            while samples_left_to_generate > 0:
                bs = self.cfg.train.batch_size * 2
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)

                sampled_batch = self.sample_batch(
                    batch_id=ident,
                    batch_size=to_generate,
                    save_final=to_save,
                    keep_chain=chains_save,
                    number_chain_steps=self.number_chain_steps,
                )
                generated_graphs.append(sampled_batch)
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save

            generated_graphs = utils.concat_sparse_graphs(generated_graphs)
            if hasattr(self, 'local_rank') and self.local_rank == 0:
                print(
                    f"Sampled {generated_graphs.batch.max().item()+1} batches on local rank {self.local_rank}. ",
                    f"Sampling took {time.time() - start:.2f} seconds\n",
                )

            if hasattr(self, 'local_rank') and self.local_rank == 0:
                print("Computing sampling metrics...")
            to_log, _ = self.val_sampling_metrics.compute_all_metrics(
                generated_graphs, self.current_epoch, local_rank=self.local_rank
            )

            filename = os.path.join(
                os.getcwd(), f"epoch{self.current_epoch}_res_Mean.txt"
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

    def on_test_epoch_start(self) -> None:
        print("Starting test...")
        if self.local_rank == 0 and wandb.run is None:
            utils.setup_wandb(
                self.cfg
            )  # Initialize wandb only when not already in a run (e.g. test_only); train+test reuses run from on_fit_start
        test_metrics = [
            self.test_nll,
            self.test_X_kl,
            self.test_E_kl,
            self.test_X_logp,
            self.test_E_logp,
            self.test_sampling_metrics,
        ]
        if self.use_charge:
            test_metrics.extend([self.test_charge_kl, self.test_charge_logp])
        for metric in test_metrics:
            metric.reset()

    def test_step(self, data, i):
        pass

    def on_test_epoch_end(self) -> None:
        """Measure likelihood on a test set and compute stability metrics."""
        enable_test_sampling = getattr(self.cfg.general, "enable_test_sampling", True)
        if not enable_test_sampling and not self.cfg.general.generated_path:
            if hasattr(self, "local_rank") and self.local_rank == 0:
                self.print("Test sampling disabled (enable_test_sampling=false), skipping.")
            return
        if self.cfg.general.generated_path:
            self.print("Loading generated samples...")
            # samples = np.load(self.cfg.general.generated_path)
            with open(self.cfg.general.generated_path, "rb") as f:
                samples = pickle.load(f)
        else:
            self.cfg.general.final_model_samples_to_generate = (
                self.cfg.general.test_variance
                * self.cfg.general.final_model_samples_to_generate
            )
            samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
            samples_left_to_save = self.cfg.general.final_model_samples_to_save
            chains_left_to_save = self.cfg.general.final_model_chains_to_save
            # multi gpu operation
            samples_left_to_generate = math.ceil(
                samples_left_to_generate / max(self._trainer.num_devices, 1)
            )
            self.print(
                f"Samples to generate: {samples_left_to_generate} for each of the {max(self._trainer.num_devices, 1)} devices"
            )
            print(f"Sampling start on GR{self.global_rank}")

            samples = []
            id = 0
            while samples_left_to_generate > 0:
                print(
                    f"Samples left to generate: {samples_left_to_generate}/"
                    f"{self.cfg.general.final_model_samples_to_generate}",
                    end="",
                    flush=True,
                )
                bs = self.cfg.train.batch_size * 2
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)

                # 支持通过配置指定采样时的节点数（用于大图训练、小图生成）
                # 默认使用20个节点（便于快速测试和可视化）
                sample_num_nodes = getattr(self.cfg.general, 'sample_num_nodes', None)
                if sample_num_nodes is not None:
                    if isinstance(sample_num_nodes, int):
                        # 固定节点数：所有图使用相同节点数
                        num_nodes = sample_num_nodes
                    elif isinstance(sample_num_nodes, list):
                        # 每个图指定不同节点数
                        num_nodes = torch.tensor(sample_num_nodes[:to_generate], device=self.device, dtype=torch.long)
                    else:
                        # 默认20个节点
                        num_nodes = 20
                else:
                    # 默认20个节点（便于快速测试和可视化）
                    num_nodes = 20
                
                sampled_batch = self.sample_batch(
                    batch_id=id,
                    batch_size=to_generate,
                    num_nodes=num_nodes,
                    save_final=to_save,
                    keep_chain=chains_save,
                    number_chain_steps=self.number_chain_steps,
                )
                samples.append(sampled_batch)

                id += to_generate
                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save

            if hasattr(self, 'local_rank') and self.local_rank == 0:
                print("Saving the generated graphs")

            samples = utils.concat_sparse_graphs(samples)
            filename = f"generated_samples1.txt"

            # Save the samples list as pickle to a file that depends on the local rank
            # This is needed to avoid overwriting the same file on different GPUs
            with open(f"generated_samples_rank{self.local_rank}.pkl", "wb") as f:
                pickle.dump(samples, f)

            # This line is used to sync between gpus
            self._trainer.strategy.barrier()
            for i in range(2, 10):
                if os.path.exists(filename):
                    filename = f"generated_samples{i}.txt"
                else:
                    break
            with open(filename, "w") as f:
                for i in range(samples.batch.max().item() + 1):
                    atoms = samples.node[samples.batch == i]
                    f.write(f"N={atoms.shape[0]}\n")
                    atoms = atoms.tolist()
                    f.write("X: \n")
                    for at in atoms:
                        f.write(f"{at} ")
                    f.write("\n")
                    f.write("E: \n")
                    bonds = samples.edge_attr[samples.batch[samples.edge_index[0]] == i]
                    for bond in bonds:
                        f.write(f"{bond} ")
                    f.write("\n")
            if hasattr(self, 'local_rank') and self.local_rank == 0:
                print("Saved.")
                print("Computing sampling metrics...")

            # Load the pickles of the other GPUs
            samples = []
            for i in range(self._trainer.num_devices):
                with open(f"generated_samples_rank{i}.pkl", "rb") as f:
                    samples.append(pickle.load(f))
            samples = utils.concat_sparse_graphs(samples)
            print("saving all samples")
            with open(f"generated_samples.pkl", "wb") as f:
                pickle.dump(samples, f)

        if hasattr(self, 'local_rank') and self.local_rank == 0:
            print("Computing sampling metrics...")
        if self.test_variance == 1:
            to_log, _ = self.test_sampling_metrics.compute_all_metrics(
                samples, self.current_epoch, local_rank=self.local_rank
            )
            # save results for testing
            if hasattr(self, 'local_rank') and self.local_rank == 0:
                print("saving results for testing")
            current_path = os.getcwd()
            res_path = os.path.join(
                current_path,
                f"test_epoch{self.current_epoch}.json",
            )
            with open(res_path, "w") as file:
                # Convert the dictionary to a JSON string and write it to the file
                json.dump(to_log, file)
        else:
            # import pdb; pdb.set_trace()
            samples = utils.split_samples(samples, self.test_variance)
            to_log = {}
            for i in range(self.test_variance):
                print(samples[i].batch.max().item() + 1)
                self.test_sampling_metrics.reset()  # reset the metrics for new evaluations
                cur_to_log, _ = self.test_sampling_metrics.compute_all_metrics(
                    samples[i], self.current_epoch, local_rank=self.local_rank
                )
                if i == 0:
                    to_log = {i: [cur_to_log[i]] for i in cur_to_log}
                else:
                    to_log = {i: to_log[i] + [cur_to_log[i]] for i in cur_to_log}

                print(f"For the {i} th sampling, we have: ")
                print(cur_to_log)
                filename = os.path.join(
                    os.getcwd(),
                    f"epoch{self.current_epoch}_res_part{i}_mean{self.test_variance}.txt",
                )
                with open(filename, "w") as file:
                    for key, value in cur_to_log.items():
                        file.write(f"{key}: {value}\n")

                with open(f"generated_samples_test{i}.pkl", "wb") as f:
                    pickle.dump(samples[i], f)

            to_log = {
                i: (np.array(to_log[i]).mean(), np.array(to_log[i]).std())
                for i in to_log
            }

        if hasattr(self, 'local_rank') and self.local_rank == 0:
            print(f"For overall {self.test_variance} samplings, we have: ")
            print(to_log)
        filename = os.path.join(
            os.getcwd(), f"epoch{self.current_epoch}_res_mean{self.test_variance}.txt"
        )
        with open(filename, "w") as file:
            for key, value in to_log.items():
                file.write(f"{key}: {value}\n")

        if hasattr(self, 'local_rank') and self.local_rank == 0:
            print("Test sampling metrics computed.")

    def apply_sparse_noise(self, data):
        """Sample noise and apply it to the data."""
        bs = int(data.batch.max() + 1)
        t_int = torch.randint(
            1, self.T + 1, size=(bs, 1), device=self.device
        ).float()  # (bs, 1)

        s_int = t_int - 1
        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(
            alpha_t_bar, device=self.device
        )  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.0) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.0) < 1e-4).all()

        # Compute transition probabilities
        # get charge distribution
        if self.use_charge:
            prob_charge = data.charge.unsqueeze(1) @ Qtb.charge[data.batch]
            charge_t = prob_charge.squeeze(1).multinomial(1).flatten()  # (N, )
            charge_t = F.one_hot(charge_t, num_classes=self.out_dims.charge)
        else:
            charge_t = data.charge

        # Diffuse sparse nodes and sample sparse node labels
        probN = data.x.unsqueeze(1) @ Qtb.X[data.batch]  # (N, 1, dx) or (N, dx)
        # 确保probN是2维: (N, dx)
        if probN.dim() == 3:
            probN = probN.squeeze(1)  # (N, dx)
        elif probN.dim() == 1:
            probN = probN.unsqueeze(0)  # (1, dx) -> 但这种情况不应该发生
        
        # 异质图：限制节点只能在其所属类型的子类别范围内采样（训练和采样保持一致）
        if self.heterogeneous and hasattr(self.dataset_info, "type_offsets"):
            type_offsets = self.dataset_info.type_offsets
            if type_offsets:
                # 获取当前节点的子类别ID
                current_node_subtype = data.x.argmax(dim=-1) if data.x.dim() > 1 else data.x.long()  # (N,)
                
                num_nodes = current_node_subtype.shape[0]
                num_subtypes = self.out_dims.X
                node_type_mask = torch.zeros((num_nodes, num_subtypes), device=self.device)
                
                # 计算每个类型的size
                sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                type_sizes = {}
                for i, (t_name, off) in enumerate(sorted_types):
                    if i + 1 < len(sorted_types):
                        type_sizes[t_name] = sorted_types[i + 1][1] - off
                    else:
                        type_sizes[t_name] = num_subtypes - off
                
                # 为每个节点生成mask
                for t_name, offset in sorted_types:
                    type_size = type_sizes.get(t_name, 0)
                    if type_size <= 0:
                        continue
                    # 找到属于该类型的节点
                    if t_name == sorted_types[-1][0]:
                        # 最后一个类型
                        type_mask = current_node_subtype >= offset
                    else:
                        next_offset = sorted_types[sorted_types.index((t_name, offset)) + 1][1]
                        type_mask = (current_node_subtype >= offset) & (current_node_subtype < next_offset)
                    
                    if type_mask.any():
                        # 允许该类型范围内的所有子类别
                        node_type_mask[type_mask, offset:offset + type_size] = 1.0
                
                # 应用mask
                probN_masked = probN * node_type_mask
                row_sum = probN_masked.sum(dim=-1, keepdim=True)
                all_zero = (row_sum.squeeze(-1) == 0)
                if all_zero.any():
                    # Fallback: use original probabilities for masked nodes (should not happen)
                    probN_masked[all_zero] = probN[all_zero]
                probN = probN_masked / probN_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # probN shape: (N, dx) - 确保是2维
        assert probN.dim() == 2, f"probN should be 2D, got {probN.dim()}D with shape {probN.shape}"
        node_t = probN.multinomial(1).flatten()  # (N, )
        # count node numbers and edge numbers for existing edges for each graph
        num_nodes = data.ptr.diff().long()
        batch_edge = data.batch[data.edge_index[0]]
        num_edges = torch.zeros(num_nodes.shape).to(self.device)
        unique, counts = torch.unique(batch_edge, sorted=True, return_counts=True)
        num_edges[unique] = counts.float()
        # count number of non-existing edges for each graph
        # heterogeneous graphs are directed: total possible edges = n * (n - 1)
        # homogeneous graphs keep legacy undirected upper-triangle formula.
        if self.heterogeneous:
            num_neg_edge = (num_nodes * (num_nodes - 1) - num_edges).clamp(min=0)  # (bs, )
        else:
            num_neg_edge = (((num_nodes - 1) * num_nodes - num_edges) / 2).clamp(min=0)  # (bs, )

        # Step1: diffuse on existing edges
        # Heterogeneous graphs are directed: keep original direction.
        # Homogeneous graphs keep legacy triu path to avoid duplicated undirected pairs.
        if self.heterogeneous:
            dir_edge_index, dir_edge_attr = data.edge_index, data.edge_attr
        else:
            # get edges defined in the top triangle of the adjacency matrix
            dir_edge_index, dir_edge_attr = utils.undirected_to_directed(
                data.edge_index, data.edge_attr
            )
        # debug: limit noise stats logging
        debug_noise = hasattr(self, "local_rank") and self.local_rank == 0
        if debug_noise:
            if not hasattr(self, "_noise_debug_count"):
                self._noise_debug_count = 0
            debug_noise = self._noise_debug_count < 1
        noise_stats = {} if debug_noise else None
        
        # 检查是否为异质图模式且有 edge_family 信息
        has_edge_family = hasattr(data, 'edge_family') and data.edge_family is not None
        if self.heterogeneous and has_edge_family:
            # 异质图模式：按关系族隔离处理
            # 异质图边本身是有向的，edge_family 与 edge_index 一一对应
            dir_edge_family = data.edge_family
            
            # 获取所有关系族的转移矩阵
            all_family_qt = self.transition_model.get_all_family_Qt_bar(alpha_t_bar, device=self.device)
            edge_family2id = getattr(self.dataset_info, "edge_family2id", {})
            id2edge_family = {v: k for k, v in edge_family2id.items()}
            
            # 确保 alpha_t_bar 是 1D 的 (bs,)
            if alpha_t_bar.dim() > 1:
                alpha_t_bar = alpha_t_bar.squeeze()
            
            # 按关系族分组处理已存在的边
            dir_edge_attr_list = []
            dir_edge_index_list = []
            neg_edge_attr_list = []
            neg_edge_index_list = []
            
            for fam_id, fam_name in id2edge_family.items():
                fam_mask = (dir_edge_family == fam_id)
                if not fam_mask.any():
                    continue
                
                # 获取该关系族的已存在边
                fam_dir_edge_index = dir_edge_index[:, fam_mask]
                fam_dir_edge_attr = dir_edge_attr[fam_mask]
                fam_batch_edge = data.batch[fam_dir_edge_index[0]]
                
                # one-hot -> discrete edge ids
                if fam_dir_edge_attr.dim() > 1:
                    fam_dir_edge_attr = fam_dir_edge_attr.argmax(dim=-1).long()
                else:
                    fam_dir_edge_attr = fam_dir_edge_attr.long()
                num_edges_fam = fam_dir_edge_index.shape[1]
                
                # 使用关系族特定的转移矩阵
                Qtb_fam = all_family_qt[fam_name]
                # Qtb_fam.E: (bs, num_states, num_states)
                # fam_batch_edge: (num_edges,) - 每条边所属的批次
                batch_Qtb_fam = Qtb_fam.E[fam_batch_edge]  # 应该是 (num_edges, num_states, num_states)
                
                # 调试：检查形状
                if batch_Qtb_fam.dim() != 3:
                    raise RuntimeError(
                        f"batch_Qtb_fam should be 3D but got {batch_Qtb_fam.dim()}D. "
                        f"Shape: {batch_Qtb_fam.shape}, Qtb_fam.E shape: {Qtb_fam.E.shape}, "
                        f"fam_batch_edge shape: {fam_batch_edge.shape}, "
                        f"fam_batch_edge unique: {torch.unique(fam_batch_edge)}"
                    )
                
                # 将全局边属性 ID 转换为关系族内的局部 ID
                # 局部 ID: 0 = no-edge, 1, 2, ... = 子类别
                # 全局 ID: 0 = no-edge, offset, offset+1, ... = 该关系族的子类别
                edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", {})
                offset = edge_family_offsets.get(fam_name, 0)
                fam_local_attr = fam_dir_edge_attr.clone()
                # no-edge (0) 保持为 0
                non_zero_mask = fam_local_attr != 0
                if non_zero_mask.any():
                    # 全局 ID -> 局部 ID: 全局 ID - offset + 1（因为局部 ID 从 1 开始）
                    fam_local_attr[non_zero_mask] = fam_local_attr[non_zero_mask] - offset + 1
                
                # 扩散已存在的边
                # 将局部 ID 转换为 one-hot 编码以进行矩阵乘法
                num_fam_states = batch_Qtb_fam.shape[-1]  # 关系族的状态数
                num_edges = len(fam_local_attr)
                
                if num_edges == 0:
                    # 如果没有边，跳过
                    dir_edge_attr_fam = torch.empty(0, dtype=torch.long, device=self.device)
                else:
                    # 确保 fam_local_attr 是 1D 的
                    fam_local_attr = fam_local_attr.flatten()
                    
                    # 确保 fam_local_attr 的长度与 num_edges 匹配
                    if len(fam_local_attr) != num_edges:
                        raise RuntimeError(
                            f"fam_local_attr length mismatch: {len(fam_local_attr)} != {num_edges}, "
                            f"fam_local_attr.shape={fam_local_attr.shape}"
                        )
                    
                    fam_local_attr_onehot = F.one_hot(fam_local_attr.long(), num_classes=num_fam_states).float()  # (num_edges, num_states)
                    
                    # 确保 batch_Qtb_fam 是 3D: (num_edges, num_states, num_states)
                    if batch_Qtb_fam.dim() == 2:
                        batch_Qtb_fam = batch_Qtb_fam.unsqueeze(0)  # (1, num_states, num_states)
                        fam_local_attr_onehot = fam_local_attr_onehot.unsqueeze(0)  # (1, num_states)
                    
                    # 确保 fam_local_attr_onehot 是 2D 的 (num_edges, num_states)
                    if fam_local_attr_onehot.dim() != 2:
                        # 如果是 3D 或更高维度，reshape 为 2D，但保持 num_edges 维度
                        if fam_local_attr_onehot.dim() == 3:
                            # 如果是 (num_edges, 1, num_states) 或类似，squeeze
                            fam_local_attr_onehot = fam_local_attr_onehot.squeeze()
                        else:
                            # 其他情况，reshape 为 (num_edges, num_states)
                            fam_local_attr_onehot = fam_local_attr_onehot.view(num_edges, num_fam_states)
                    
                    # 使用 bmm 进行批处理矩阵乘法
                    # fam_local_attr_onehot: (num_edges, num_states) -> (num_edges, 1, num_states)
                    # batch_Qtb_fam: (num_edges, num_states, num_states)
                    # 结果: (num_edges, 1, num_states) -> (num_edges, num_states)
                    fam_local_attr_onehot_3d = fam_local_attr_onehot.unsqueeze(1)  # (num_edges, 1, num_states)
                    
                    probE_fam = torch.bmm(fam_local_attr_onehot_3d, batch_Qtb_fam).squeeze(1)  # (num_edges, num_states)
                    dir_edge_attr_fam = probE_fam.multinomial(1).flatten()  # (num_edges,)
                # 转换回全局 ID
                # 局部 ID -> 全局 ID: 局部 ID 0 -> 全局 0, 局部 ID 1,2,... -> 全局 offset, offset+1,...
                global_attr_fam = dir_edge_attr_fam.clone()
                non_zero_local_mask = global_attr_fam != 0
                if non_zero_local_mask.any():
                    global_attr_fam[non_zero_local_mask] = global_attr_fam[non_zero_local_mask] - 1 + offset
                dir_edge_attr_fam = global_attr_fam
                
                dir_edge_attr_list.append(dir_edge_attr_fam)
                dir_edge_index_list.append(fam_dir_edge_index)
                if debug_noise:
                    noise_stats[fam_name] = {
                        "existing_dir": int(num_edges_fam),
                        "existing_to_noedge": int((dir_edge_attr_fam == 0).sum().item()),
                        "sampled_new": 0,
                    }
                
                # Step2: 计算该关系族的非存在边数并采样（按关系族隔离的伯努利采样）
                # 根据图片：m̄_fam = 该关系族可能的边数 - 该关系族已存在的边数
                # qt_fam = emerge_prob_fam = 1 - Q_Y|a,b[0,0]（从no-edge转移到有边的概率）
                # k_fam ~ B(m̄_fam, qt_fam)
                fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
                if fam_name in fam_endpoints:
                    src_type = fam_endpoints[fam_name]["src_type"]
                    dst_type = fam_endpoints[fam_name]["dst_type"]
                    
                    # 获取节点类型到全局ID的映射（type_offsets）
                    # 从 dataset_info 获取，如果没有则从 vocab.json 加载
                    node_type2id = getattr(self.dataset_info, "node_type2id", {})
                    type_offsets = getattr(self.dataset_info, "type_offsets", {})
                    node_type_names = getattr(self.dataset_info, "node_type_names", [])
                    
                    # 如果 type_offsets 不存在，需要从 vocab.json 加载
                    if not type_offsets and node_type_names:
                        # 从 vocab.json 加载 type_offsets
                        vocab_path = osp.join(getattr(self.dataset_info, "vocab_path", ""), "vocab.json")
                        if not vocab_path or not osp.exists(vocab_path):
                            # 尝试从 dataset 的 processed_dir 获取
                            if hasattr(self.dataset_info, "datamodule") and hasattr(self.dataset_info.datamodule, "inner"):
                                vocab_path = osp.join(self.dataset_info.datamodule.inner.processed_dir, "vocab.json")
                        
                        if osp.exists(vocab_path):
                            import json
                            with open(vocab_path, "r", encoding="utf-8") as f:
                                vocab = json.load(f)
                            # 从第一个子图的 meta.json 获取 type_sizes
                            if hasattr(self.dataset_info, "datamodule") and hasattr(self.dataset_info.datamodule, "inner"):
                                subgraph_dirs = [d for d in os.listdir(self.dataset_info.datamodule.inner.root) 
                                                if osp.isdir(osp.join(self.dataset_info.datamodule.inner.root, d)) and d.startswith("subgraph_")]
                                if len(subgraph_dirs) > 0:
                                    meta0 = json.load(open(osp.join(self.dataset_info.datamodule.inner.root, subgraph_dirs[0], "meta.json"), "r", encoding="utf-8"))
                                    schema_by_type = meta0.get("schema_by_type", {})
                                    type_sizes = [len(schema_by_type.get(t, [])) for t in node_type_names]
                                    type_offsets = {}
                                    cur = 0
                                    for t, size in zip(node_type_names, type_sizes):
                                        type_offsets[t] = cur
                                        cur += size
                    
                    # 计算每个批次中该关系族的 src_type 和 dst_type 节点数
                    # node_t 是全局子类别 ID，需要转换为节点类型
                    src_mask = None
                    dst_mask = None
                    if type_offsets and src_type in type_offsets and dst_type in type_offsets:
                        src_offset = type_offsets[src_type]
                        dst_offset = type_offsets[dst_type]
                        # 计算每个节点类型的 size（从 type_offsets 推断）
                        type_sizes = {}
                        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                        for i, (t, off) in enumerate(sorted_types):
                            if i + 1 < len(sorted_types):
                                type_sizes[t] = sorted_types[i + 1][1] - off
                            else:
                                # 最后一个类型
                                type_sizes[t] = self.out_dims.X - off
                        
                        src_size = type_sizes.get(src_type, 0)
                        dst_size = type_sizes.get(dst_type, 0)
                        
                        # 计算每个批次中 src_type 和 dst_type 的节点数
                        # node_t 是全局子类别 ID，需要判断是否在 src_type 或 dst_type 的范围内
                        src_mask = (node_t >= src_offset) & (node_t < src_offset + src_size)  # (N,)
                        dst_mask = (node_t >= dst_offset) & (node_t < dst_offset + dst_size)  # (N,)
                        
                        # 按批次统计节点数
                        bs = len(num_nodes)
                        num_src_nodes = torch.zeros(bs, dtype=torch.long, device=self.device)
                        num_dst_nodes = torch.zeros(bs, dtype=torch.long, device=self.device)
                        for b in range(bs):
                            batch_mask = (data.batch == b)
                            num_src_nodes[b] = (src_mask & batch_mask).sum()
                            num_dst_nodes[b] = (dst_mask & batch_mask).sum()
                        
                        # 计算该关系族可能的边数（有向图：src_nodes * dst_nodes）
                        num_fam_possible_edges = num_src_nodes * num_dst_nodes  # (bs,)
                        # 同类型关系需要排除自环 (u,u)
                        if src_type == dst_type:
                            num_fam_possible_edges = torch.clamp(
                                num_fam_possible_edges - num_src_nodes, min=0
                            )
                        
                        # 计算该关系族已存在的边数
                        fam_batch_edge = data.batch[fam_dir_edge_index[0]]
                        num_fam_existing_edges = torch.zeros(bs, dtype=torch.long, device=self.device)
                        unique_fam_batch, counts_fam = torch.unique(fam_batch_edge, sorted=True, return_counts=True)
                        num_fam_existing_edges[unique_fam_batch] = counts_fam.long()
                        
                        # 计算该关系族的非存在边数：m̄_fam = 可能的边数 - 已存在的边数
                        num_fam_neg_edge = (num_fam_possible_edges - num_fam_existing_edges).float()  # (bs,)
                        num_fam_neg_edge = torch.clamp(num_fam_neg_edge, min=0)  # 确保非负
                    else:
                        # 如果无法获取节点类型信息，回退到全局值
                        num_fam_neg_edge = num_neg_edge
                else:
                    # 如果没有端点类型信息，使用全局值
                    num_fam_neg_edge = num_neg_edge
                
                # 计算该关系族的 emerge_prob（qt_fam = 1 - Q_Y|a,b[0,0]）
                # Qtb_fam.E[:, 0, 1:] 是从 no-edge (0) 转移到有边状态 (1, 2, ...) 的概率
                emerge_prob_fam = Qtb_fam.E[:, 0, 1:].sum(-1)  # (bs, )
                
                # 使用二项分布采样：k_fam ~ B(m̄_fam, qt_fam)
                num_emerge_edges_fam = (
                    torch.distributions.binomial.Binomial(num_fam_neg_edge.long(), emerge_prob_fam)
                    .sample()
                    .int()
                )
                
                if num_emerge_edges_fam.max() > 0:
                    # 采样非存在的边（需要考虑端点类型约束）
                    # 对于异质图，需要确保只采样符合该关系族端点类型的边对
                    if src_mask is not None and dst_mask is not None:
                        # 使用支持端点类型约束的采样函数
                        neg_edge_index_fam = sample_non_existing_edges_batched_heterogeneous(
                            num_edges_to_sample=num_emerge_edges_fam,
                            existing_edge_index=dir_edge_index,
                            num_nodes=num_nodes,
                            batch=data.batch,
                            src_mask=src_mask,
                            dst_mask=dst_mask,
                        )
                    else:
                        # 回退到全局采样函数
                        neg_edge_index_fam = sample_non_existing_edges_batched(
                            num_edges_to_sample=num_emerge_edges_fam,
                            existing_edge_index=dir_edge_index,
                            num_nodes=num_nodes,
                            batch=data.batch,
                        )
                    neg_edge_attr_fam = sample_non_existing_edge_attr(
                        query_edges_dist_batch=Qtb_fam.E[:, 0, 1:],
                        num_edges_to_sample=num_emerge_edges_fam,
                    )
                    # 转换回全局 ID
                    # sample_non_existing_edge_attr 返回的局部 ID 从 1 开始（对应关系族的子类别）
                    # 需要转换为全局 ID: 局部 ID 1,2,... -> 全局 offset, offset+1,...
                    global_neg_attr_fam = neg_edge_attr_fam.clone()
                    non_zero_neg_mask = global_neg_attr_fam != 0
                    if non_zero_neg_mask.any():
                        global_neg_attr_fam[non_zero_neg_mask] = global_neg_attr_fam[non_zero_neg_mask] - 1 + offset
                    neg_edge_attr_fam = global_neg_attr_fam
                    
                    neg_edge_attr_list.append(neg_edge_attr_fam)
                    neg_edge_index_list.append(neg_edge_index_fam)
                    if debug_noise and fam_name in noise_stats:
                        noise_stats[fam_name]["sampled_new"] = int(neg_edge_attr_fam.numel())
            
            # 合并所有关系族的结果
            if len(dir_edge_attr_list) > 0:
                dir_edge_attr = torch.cat(dir_edge_attr_list, dim=0)
                dir_edge_index = torch.cat(dir_edge_index_list, dim=1)
            else:
                dir_edge_attr = torch.empty(0, dtype=torch.long, device=self.device)
                dir_edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)
            
            if len(neg_edge_attr_list) > 0:
                neg_edge_attr = torch.cat(neg_edge_attr_list, dim=0)
                neg_edge_index = torch.cat(neg_edge_index_list, dim=1)
                E_t_attr = torch.hstack([dir_edge_attr, neg_edge_attr])
                E_t_index = torch.hstack([dir_edge_index, neg_edge_index])
            else:
                E_t_attr = dir_edge_attr
                E_t_index = dir_edge_index
        else:
            # 同质图模式：使用原始逻辑
            batch_edge = data.batch[dir_edge_index[0]]
            batch_Qtb = Qtb.E[batch_edge]
            probE = dir_edge_attr.unsqueeze(1) @ batch_Qtb
            dir_edge_attr = probE.squeeze(1).multinomial(1).flatten()

            # Step2: diffuse on non-existing edges
            # get number of new edges according to Qtb
            emerge_prob = Qtb.E[:, 0, 1:].sum(-1)  # (bs, )
            num_emerge_edges = (
                torch.distributions.binomial.Binomial(num_neg_edge, emerge_prob)
                .sample()
                .int()
            )

            # combine existing and non-existing edges (both are directed, i.e. triu)
            if num_emerge_edges.max() > 0:
                # sample non-existing edges
                neg_edge_index = sample_non_existing_edges_batched(
                    num_edges_to_sample=num_emerge_edges,
                    existing_edge_index=dir_edge_index,
                    num_nodes=num_nodes,
                    batch=data.batch,
                )
                neg_edge_attr = sample_non_existing_edge_attr(
                    query_edges_dist_batch=Qtb.E[:, 0, 1:],
                    num_edges_to_sample=num_emerge_edges,
                )

                E_t_attr = torch.hstack([dir_edge_attr, neg_edge_attr])
                E_t_index = torch.hstack([dir_edge_index, neg_edge_index])
            else:
                E_t_attr = dir_edge_attr
                E_t_index = dir_edge_index

        # mask non-existing edges
        mask = E_t_attr != 0
        if debug_noise:
            total_dir_edges = int(dir_edge_index.shape[1])
            total_noisy_before = int(E_t_attr.numel())
            total_noisy_after = int(mask.sum().item())
            print(
                f"[NOISE] total_dir_edges={total_dir_edges}, "
                f"noisy_before_mask={total_noisy_before}, noisy_after_mask={total_noisy_after}"
            )
            # overlap ratio between original directed edges and noisy directed edges
            try:
                orig_edges = dir_edge_index.t().tolist()
                noisy_edges = E_t_index[:, mask].t().tolist()
                orig_set = set((u, v) for u, v in orig_edges)
                noisy_set = set((u, v) for u, v in noisy_edges)
                overlap = len(orig_set & noisy_set)
                overlap_ratio = overlap / len(orig_set) * 100 if orig_set else 0.0
                print(
                    f"[NOISE] overlap_original={overlap} "
                    f"({overlap_ratio:.1f}%)"
                )
            except Exception as _e:
                print(f"[NOISE] overlap calc failed: {_e}")
            if noise_stats:
                for fam_name, stats in noise_stats.items():
                    print(
                        f"[NOISE] fam={fam_name} existing_dir={stats['existing_dir']}, "
                        f"existing_to_noedge={stats['existing_to_noedge']}, "
                        f"sampled_new={stats['sampled_new']}"
                    )
            self._noise_debug_count += 1
        E_t_attr = E_t_attr[mask]
        E_t_index = E_t_index[:, mask]
        # Heterogeneous graphs keep directed edges; homogeneous keeps legacy symmetry.
        if not self.heterogeneous:
            E_t_index, E_t_attr = utils.to_undirected(E_t_index, E_t_attr)

        E_t_attr = F.one_hot(E_t_attr, num_classes=self.out_dims.E)
        node_t = F.one_hot(node_t, num_classes=self.out_dims.X)

        sparse_noisy_data = {
            "t_int": t_int,
            "t_float": t_float,
            "beta_t": beta_t,
            "alpha_s_bar": alpha_s_bar,
            "alpha_t_bar": alpha_t_bar,
            "node_t": node_t,
            "edge_index_t": E_t_index,
            "edge_attr_t": E_t_attr,
            "comp_edge_index_t": None,
            "comp_edge_attr_t": None,  # computational graph
            "y_t": data.y,
            "batch": data.batch,
            "ptr": data.ptr,
            "charge_t": charge_t,
        }

        return sparse_noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, charge, test):
        """Computes an estimator for the variational lower bound.
        pred: (batch_size, n, total_features)
        noisy_data: dict
        X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
        node_mask : (bs, n)
        Output: nll (size 1)
        """
        t = noisy_data["t_float"]

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask, charge=charge)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(
            X, E, y, charge, pred, noisy_data, node_mask, test=test
        )

        # Combine terms
        nlls = -log_pN + kl_prior + loss_all_t
        # nlls = loss_all_t
        assert (~nlls.isnan()).all(), f"NLLs contain NaNs: {nlls}"
        assert len(nlls.shape) == 1, f"{nlls.shape} has more than only batch dim."

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)  # Average over the batch

        if wandb.run:
            wandb.log(
                {
                    "kl prior": kl_prior.mean(),
                    "Estimator loss terms": loss_all_t.mean(),
                    "log_pn": log_pN.mean(),
                    "val_nll": nll,
                    "epoch": self.current_epoch,
                },
                commit=False,
            )

        return nll

    def kl_prior(self, X, E, node_mask, charge):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).
        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        # 对于异质图，KL 先验计算使用全局转移矩阵作为近似
        # 因为 KL 先验通常值很小，对训练影响不大
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        if (
            self.heterogeneous
            and getattr(self.dataset_info, "edge_family_marginals", None)
            and getattr(self.dataset_info, "edge_family_offsets", None)
            and getattr(self.dataset_info, "fam_endpoints", None)
            and getattr(self.dataset_info, "type_offsets", None)
        ):
            limit_E = self._build_limit_E_hetero(X, node_mask).type_as(probE)
        else:
            limit_E = (
                self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)
            )

        if self.use_charge:
            prob_charge = charge @ Qtb.charge  # (bs, n, de_out)
            limit_charge = (
                self.limit_dist.charge[None, None, :]
                .expand(bs, n, -1)
                .type_as(prob_charge)
            )
            limit_charge = limit_charge.clone()
        else:
            prob_charge = limit_charge = None

        # Make sure that masked rows do not contribute to the loss
        (
            limit_dist_X,
            limit_dist_E,
            probX,
            probE,
            limit_dist_charge,
            prob_charge,
        ) = diffusion_utils.mask_distributions(
            true_X=limit_X.clone(),
            true_E=limit_E.clone(),
            pred_X=probX,
            pred_E=probE,
            node_mask=node_mask,
            true_charge=limit_charge,
            pred_charge=prob_charge,
        )

        kl_distance_X = F.kl_div(
            input=probX.log(), target=limit_dist_X, reduction="none"
        )
        kl_distance_E = F.kl_div(
            input=probE.log(), target=limit_dist_E, reduction="none"
        )

        # not all edges are used for loss calculation
        E_mask = torch.logical_or(
            kl_distance_E.sum(-1).isnan(), kl_distance_E.sum(-1).isinf()
        )
        kl_distance_E[E_mask] = 0
        X_mask = torch.logical_or(
            kl_distance_X.sum(-1).isnan(), kl_distance_X.sum(-1).isinf()
        )
        kl_distance_X[X_mask] = 0

        loss = diffusion_utils.sum_except_batch(
            kl_distance_X
        ) + diffusion_utils.sum_except_batch(kl_distance_E)

        # The above code is using the Python debugger module `pdb` to set a breakpoint in the code.
        # When the code is executed, it will pause at this line and allow you to interactively debug
        # the program.

        if self.use_charge:
            kl_distance_charge = F.kl_div(
                input=prob_charge.log(), target=limit_dist_charge, reduction="none"
            )
            kl_distance_charge[X_mask] = 0
            loss = loss + diffusion_utils.sum_except_batch(kl_distance_charge)

        assert (~loss.isnan()).any()

        return loss

    def _build_limit_E_hetero(self, X, node_mask):
        """Build per-family limit distribution for edges based on node types."""
        device = X.device
        edge_family_marginals = getattr(self.dataset_info, "edge_family_marginals", {})
        edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", {})
        fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
        type_offsets = getattr(self.dataset_info, "type_offsets", {})

        if not edge_family_marginals or not edge_family_offsets or not fam_endpoints or not type_offsets:
            return self.limit_dist.E[None, None, None, :].expand(
                X.size(0), X.size(1), X.size(1), -1
            )

        # node subtype -> node type id
        subtype_ids = X.argmax(dim=-1)
        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
        type_names_ordered = [t for t, _ in sorted_types]
        type_sizes = {}
        for i, (t, off) in enumerate(sorted_types):
            if i + 1 < len(sorted_types):
                type_sizes[t] = sorted_types[i + 1][1] - off
            else:
                type_sizes[t] = max(1, self.out_dims.X - off)

        num_types = len(type_names_ordered)
        node_type_ids = subtype_ids.new_full(subtype_ids.shape, -1)
        for tidx, tname in enumerate(type_names_ordered):
            off = type_offsets[tname]
            size = type_sizes.get(tname, 0)
            mask = (subtype_ids >= off) & (subtype_ids < off + size)
            node_type_ids[mask] = tidx
        node_type_ids[~node_mask] = -1

        # map type pair -> family index
        fam_list = sorted(edge_family_marginals.keys())
        fam_index = {f: i for i, f in enumerate(fam_list)}
        fallback_idx = len(fam_list)
        pair_table = torch.full((num_types, num_types), fallback_idx, device=device, dtype=torch.long)
        for fam_name, endpoints in fam_endpoints.items():
            if fam_name not in fam_index:
                continue
            src_t = endpoints.get("src_type")
            dst_t = endpoints.get("dst_type")
            if src_t in type_names_ordered and dst_t in type_names_ordered:
                s_idx = type_names_ordered.index(src_t)
                d_idx = type_names_ordered.index(dst_t)
                pair_table[s_idx, d_idx] = fam_index[fam_name]

        # build global marginal vectors for each family
        fam_global = []
        for fam_name in fam_list:
            fam_marginals = edge_family_marginals[fam_name]
            if not isinstance(fam_marginals, torch.Tensor):
                fam_marginals = torch.tensor(fam_marginals, dtype=torch.float, device=device)
            else:
                fam_marginals = fam_marginals.to(device)
            global_vec = torch.zeros(self.out_dims.E, device=device, dtype=fam_marginals.dtype)
            global_vec[0] = fam_marginals[0] if fam_marginals.numel() > 0 else 0.0
            offset = edge_family_offsets.get(fam_name, 0)
            next_offset = self.out_dims.E
            for _, o in edge_family_offsets.items():
                if o > offset and o < next_offset:
                    next_offset = o
            num_subtypes = max(next_offset - offset, 0)
            for i in range(1, min(num_subtypes + 1, fam_marginals.numel())):
                gid = offset + (i - 1)
                if 0 <= gid < self.out_dims.E:
                    global_vec[gid] = fam_marginals[i]
            fam_global.append(global_vec)

        # add fallback global marginal
        fam_global.append(self.limit_dist.E.to(device))
        fam_global = torch.stack(fam_global, dim=0)  # (F+1, de)

        src = node_type_ids.unsqueeze(2).clamp(min=0)
        dst = node_type_ids.unsqueeze(1).clamp(min=0)
        fam_idx = pair_table[src, dst]
        fam_idx = torch.where(
            (node_type_ids.unsqueeze(2) < 0) | (node_type_ids.unsqueeze(1) < 0),
            torch.full_like(fam_idx, fallback_idx),
            fam_idx,
        )

        return fam_global[fam_idx]

    def compute_Lt(self, X, E, y, charge, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)

        if self.use_charge:
            pred_probs_charge = F.softmax(pred.charge, dim=-1)
        else:
            pred_probs_charge = None
            charge = None

        # 检查是否为异质图模式
        if self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets") and len(getattr(self.dataset_info, "edge_family_offsets", {})) > 0:
            # 异质图模式：为每个关系族使用独立的转移矩阵
            # 获取所有关系族的转移矩阵
            all_family_qt = self.transition_model.get_all_family_Qt(noisy_data["beta_t"], device=self.device)
            all_family_qsb = self.transition_model.get_all_family_Qt_bar(noisy_data["alpha_s_bar"], device=self.device)
            all_family_qtb = self.transition_model.get_all_family_Qt_bar(noisy_data["alpha_t_bar"], device=self.device)
            
            # 使用第一个关系族的转移矩阵计算节点和电荷（所有关系族共享）
            first_fam_name = list(all_family_qt.keys())[0]
            Qt_X = all_family_qt[first_fam_name].X
            Qsb_X = all_family_qsb[first_fam_name].X
            Qtb_X = all_family_qtb[first_fam_name].X
            Qt_charge = all_family_qt[first_fam_name].charge if self.use_charge else None
            Qsb_charge = all_family_qsb[first_fam_name].charge if self.use_charge else None
            Qtb_charge = all_family_qtb[first_fam_name].charge if self.use_charge else None
            
            # 计算节点和电荷的后验分布（使用全局转移矩阵）
            prob_true_X = diffusion_utils.compute_posterior_distribution(
                M=X, M_t=noisy_data["X_t"], Qt_M=Qt_X, Qsb_M=Qsb_X, Qtb_M=Qtb_X
            )
            prob_pred_X = diffusion_utils.compute_posterior_distribution(
                M=pred_probs_X, M_t=noisy_data["X_t"], Qt_M=Qt_X, Qsb_M=Qsb_X, Qtb_M=Qtb_X
            )
            
            prob_true_charge = None
            prob_pred_charge = None
            if self.use_charge and charge is not None:
                prob_true_charge = diffusion_utils.compute_posterior_distribution(
                    M=charge, M_t=noisy_data["charge_t"], Qt_M=Qt_charge, Qsb_M=Qsb_charge, Qtb_M=Qtb_charge
                )
                prob_pred_charge = diffusion_utils.compute_posterior_distribution(
                    M=pred_probs_charge, M_t=noisy_data["charge_t"], Qt_M=Qt_charge, Qsb_M=Qsb_charge, Qtb_M=Qtb_charge
                )
            
            # 为每个关系族独立计算边的后验分布
            bs, n, d = X.shape
            E_t = noisy_data["E_t"]  # (bs, n, n, de) - 噪声后的边
            # 使用真实的 E 来判断关系族，而不是噪声后的 E_t
            # 因为损失计算的目标是预测真实的 E，应该使用真实边所属关系族的转移矩阵
            E_discrete = E.argmax(dim=-1)  # (bs, n, n) - 全局ID（基于真实的 E）
            E_t_discrete = E_t.argmax(dim=-1)  # (bs, n, n) - 全局ID（基于噪声后的 E_t，用于后续计算）
            
            edge_family_offsets = self.dataset_info.edge_family_offsets
            fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
            type_offsets = getattr(self.dataset_info, "type_offsets", {})
            num_global_states = self.out_dims.E
            
            # 初始化全局状态空间的概率
            prob_true_E = torch.zeros_like(E)  # (bs, n, n, de)
            prob_pred_E = torch.zeros_like(pred_probs_E)  # (bs, n, n, de)
            
            # 获取每个节点的类型（用于确定每条边应该属于哪个关系族）
            X_discrete = X.argmax(dim=-1)  # (bs, n) - 节点子类别ID
            node_type_ids = torch.zeros((bs, n), dtype=torch.long, device=self.device) - 1  # -1 表示未知
            if type_offsets:
                sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                for b in range(bs):
                    for i, (t_name, off) in enumerate(sorted_types):
                        if i + 1 < len(sorted_types):
                            next_offset = sorted_types[i + 1][1]
                            type_mask = (X_discrete[b] >= off) & (X_discrete[b] < next_offset)
                        else:
                            type_mask = X_discrete[b] >= off
                        node_type_ids[b][type_mask] = i
            
            # 为每个关系族计算后验分布
            for fam_name in all_family_qt.keys():
                if fam_name not in fam_endpoints:
                    continue
                    
                offset = edge_family_offsets.get(fam_name, 0)
                
                # 找到下一个关系族的offset
                next_offset = num_global_states
                for _, other_offset in edge_family_offsets.items():
                    if other_offset > offset and other_offset < next_offset:
                        next_offset = other_offset
                
                # 获取该关系族的端点类型
                src_type = fam_endpoints[fam_name]["src_type"]
                dst_type = fam_endpoints[fam_name]["dst_type"]
                
                # 找到该关系族对应的节点类型索引
                src_type_idx = None
                dst_type_idx = None
                if type_offsets:
                    sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                    for idx, (t_name, _) in enumerate(sorted_types):
                        if t_name == src_type:
                            src_type_idx = idx
                        if t_name == dst_type:
                            dst_type_idx = idx
                
                if src_type_idx is None or dst_type_idx is None:
                    continue
                
                # 判断哪些边属于这个关系族
                # 对于有边的位置：使用真实的 E 判断（E_discrete >= offset & < next_offset）
                # 对于 no-edge 位置：根据 (src_type, dst_type) 判断是否属于该关系族
                # 这样每个 (i,j) 位置只属于一个关系族，避免多族覆盖
                fam_mask = torch.zeros((bs, n, n), dtype=torch.bool, device=self.device)
                for b in range(bs):
                    # 有边的位置：使用真实 E 判断
                    has_edge_mask = (E_discrete[b] >= offset) & (E_discrete[b] < next_offset)  # (n, n)
                    
                    # no-edge 位置：根据节点类型判断是否属于该关系族
                    src_types = node_type_ids[b].unsqueeze(1).expand(-1, n)  # (n, n)
                    dst_types = node_type_ids[b].unsqueeze(0).expand(n, -1)  # (n, n)
                    no_edge_mask = (E_discrete[b] == 0)  # (n, n)
                    type_match_mask = (src_types == src_type_idx) & (dst_types == dst_type_idx)  # (n, n)
                    no_edge_in_fam = no_edge_mask & type_match_mask  # (n, n)
                    
                    # 合并：有边且属于该族，或 no-edge 且节点类型匹配该族
                    fam_mask[b] = has_edge_mask | no_edge_in_fam
                
                if not fam_mask.any():
                    continue
                
                # 获取该关系族的转移矩阵
                Qt_fam = all_family_qt[fam_name].E  # (bs, num_fam_states, num_fam_states)
                Qsb_fam = all_family_qsb[fam_name].E  # (bs, num_fam_states, num_fam_states)
                Qtb_fam = all_family_qtb[fam_name].E  # (bs, num_fam_states, num_fam_states)
                num_fam_states = Qt_fam.shape[-1]
                
                # 按batch处理每个关系族的边
                for b in range(bs):
                    batch_fam_mask = fam_mask[b]  # (n, n)
                    if not batch_fam_mask.any():
                        continue
                    
                    # 获取该batch中属于该关系族的边
                    E_b_fam = E[b][batch_fam_mask]  # (num_edges_b_fam, de)
                    E_t_b_fam = E_t[b][batch_fam_mask]  # (num_edges_b_fam, de)
                    pred_E_b_fam = pred_probs_E[b][batch_fam_mask]  # (num_edges_b_fam, de)
                    
                    # 将全局状态转换为局部状态
                    E_t_b_fam_discrete = E_t_b_fam.argmax(dim=-1)  # (num_edges_b_fam,)
                    E_t_b_fam_local = E_t_b_fam_discrete.clone()
                    # 只处理属于该关系族的边（全局ID在 [offset, next_offset) 范围内）
                    valid_mask = (E_t_b_fam_discrete == 0) | ((E_t_b_fam_discrete >= offset) & (E_t_b_fam_discrete < next_offset))
                    non_zero_valid_mask = valid_mask & (E_t_b_fam_local != 0)
                    if non_zero_valid_mask.any():
                        E_t_b_fam_local[non_zero_valid_mask] = E_t_b_fam_local[non_zero_valid_mask] - offset + 1
                    # 对于不在范围内的边，设置为0（无边）
                    E_t_b_fam_local[~valid_mask] = 0
                    # 确保局部状态值在有效范围内 [0, num_fam_states)
                    E_t_b_fam_local = torch.clamp(E_t_b_fam_local, 0, num_fam_states - 1)
                    
                    E_b_fam_discrete = E_b_fam.argmax(dim=-1)  # (num_edges_b_fam,)
                    E_b_fam_local = E_b_fam_discrete.clone()
                    # 只处理属于该关系族的边
                    valid_mask_b = (E_b_fam_discrete == 0) | ((E_b_fam_discrete >= offset) & (E_b_fam_discrete < next_offset))
                    non_zero_valid_mask_b = valid_mask_b & (E_b_fam_local != 0)
                    if non_zero_valid_mask_b.any():
                        E_b_fam_local[non_zero_valid_mask_b] = E_b_fam_local[non_zero_valid_mask_b] - offset + 1
                    # 对于不在范围内的边，设置为0（无边）
                    E_b_fam_local[~valid_mask_b] = 0
                    # 确保局部状态值在有效范围内
                    E_b_fam_local = torch.clamp(E_b_fam_local, 0, num_fam_states - 1)
                    
                    # 转换为局部状态的one-hot编码
                    E_t_b_fam_local_onehot = F.one_hot(E_t_b_fam_local.long(), num_classes=num_fam_states).float()  # (num_edges_b_fam, num_fam_states)
                    E_b_fam_local_onehot = F.one_hot(E_b_fam_local.long(), num_classes=num_fam_states).float()  # (num_edges_b_fam, num_fam_states)
                    # 评估分支使用 soft 概率而非 argmax->onehot，避免丢失存在性/子类别不确定性
                    pred_E_b_fam_local_prob = torch.zeros(
                        (pred_E_b_fam.shape[0], num_fam_states),
                        device=pred_E_b_fam.device,
                        dtype=pred_E_b_fam.dtype,
                    )
                    # 局部状态 0 对应全局 no-edge(0)
                    pred_E_b_fam_local_prob[:, 0] = pred_E_b_fam[:, 0]
                    # 局部状态 1.. 对应该关系族的全局子类别 [offset, next_offset)
                    num_subtypes_local = max(next_offset - offset, 0)
                    copy_width = min(
                        num_subtypes_local,
                        max(num_fam_states - 1, 0),
                        max(pred_E_b_fam.shape[-1] - offset, 0),
                    )
                    if copy_width > 0:
                        pred_E_b_fam_local_prob[:, 1 : 1 + copy_width] = pred_E_b_fam[
                            :, offset : offset + copy_width
                        ]
                    # 归一化为局部分布；若出现全零行，退化为 no-edge
                    local_sum = pred_E_b_fam_local_prob.sum(dim=-1, keepdim=True)
                    zero_row = local_sum.squeeze(-1) <= 0
                    if zero_row.any():
                        pred_E_b_fam_local_prob[zero_row, 0] = 1.0
                        local_sum = pred_E_b_fam_local_prob.sum(dim=-1, keepdim=True)
                    pred_E_b_fam_local_prob = pred_E_b_fam_local_prob / local_sum.clamp(min=1e-8)
                    
                    # 获取该batch的转移矩阵
                    Qt_fam_b = Qt_fam[b:b+1]  # (1, num_fam_states, num_fam_states)
                    Qsb_fam_b = Qsb_fam[b:b+1]  # (1, num_fam_states, num_fam_states)
                    Qtb_fam_b = Qtb_fam[b:b+1]  # (1, num_fam_states, num_fam_states)
                    
                    # 计算局部状态空间的后验分布
                    # compute_posterior_distribution 期望: M (1, N, d), Qt_M (1, d, d)
                    prob_true_E_b_fam_local = diffusion_utils.compute_posterior_distribution(
                        M=E_b_fam_local_onehot.unsqueeze(0),  # (1, num_edges_b_fam, num_fam_states)
                        M_t=E_t_b_fam_local_onehot.unsqueeze(0),  # (1, num_edges_b_fam, num_fam_states)
                        Qt_M=Qt_fam_b,  # (1, num_fam_states, num_fam_states)
                        Qsb_M=Qsb_fam_b,  # (1, num_fam_states, num_fam_states)
                        Qtb_M=Qtb_fam_b,  # (1, num_fam_states, num_fam_states)
                    )  # (1, num_edges_b_fam, num_fam_states)
                    prob_true_E_b_fam_local = prob_true_E_b_fam_local.squeeze(0)  # (num_edges_b_fam, num_fam_states)
                    
                    prob_pred_E_b_fam_local = diffusion_utils.compute_posterior_distribution(
                        M=pred_E_b_fam_local_prob.unsqueeze(0),  # (1, num_edges_b_fam, num_fam_states)
                        M_t=E_t_b_fam_local_onehot.unsqueeze(0),  # (1, num_edges_b_fam, num_fam_states)
                        Qt_M=Qt_fam_b,  # (1, num_fam_states, num_fam_states)
                        Qsb_M=Qsb_fam_b,  # (1, num_fam_states, num_fam_states)
                        Qtb_M=Qtb_fam_b,  # (1, num_fam_states, num_fam_states)
                    )  # (1, num_edges_b_fam, num_fam_states)
                    prob_pred_E_b_fam_local = prob_pred_E_b_fam_local.squeeze(0)  # (num_edges_b_fam, num_fam_states)
                    
                    # 映射回全局状态空间（直接填充，避免创建中间张量）
                    # 直接填充到全局概率矩阵，减少内存占用
                    # batch_fam_mask 是 (n, n) 布尔mask，prob_true_E[b] 是 (n, n, de)
                    # 使用 mask 索引会得到 (num_edges_b_fam, de) 形状的张量
                    # NOTE: avoid chained boolean indexing assignment
                    # `prob_true_E[b][batch_fam_mask][:, k] = ...` writes to a temporary tensor.
                    # Use flattened indices to ensure in-place writeback to base tensor.
                    flat_edge_idx = torch.where(batch_fam_mask.reshape(-1))[0]
                    prob_true_b = prob_true_E[b].reshape(-1, num_global_states)
                    prob_pred_b = prob_pred_E[b].reshape(-1, num_global_states)
                    for local_state in range(num_fam_states):
                        if local_state == 0:
                            global_state = 0
                        else:
                            global_state = offset + local_state - 1
                        if global_state < num_global_states:
                            prob_true_b[flat_edge_idx, global_state] = prob_true_E_b_fam_local[:, local_state]
                            prob_pred_b[flat_edge_idx, global_state] = prob_pred_E_b_fam_local[:, local_state]
                    
                    # 释放中间张量以节省内存
                    del E_t_b_fam_local_onehot, E_b_fam_local_onehot, pred_E_b_fam_local_prob
                    del prob_true_E_b_fam_local, prob_pred_E_b_fam_local
                    del E_t_b_fam_local, E_b_fam_local
            
            # 创建 PlaceHolder 对象
            prob_true = utils.PlaceHolder(X=prob_true_X, E=prob_true_E.reshape(bs, n * n, -1), y=noisy_data["y_t"], charge=prob_true_charge)
            prob_pred = utils.PlaceHolder(X=prob_pred_X, E=prob_pred_E.reshape(bs, n * n, -1), y=noisy_data["y_t"], charge=prob_pred_charge)
            prob_true.E = prob_true.E.reshape((bs, n, n, -1))
            prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))
        else:
            # 同质图模式：使用全局转移矩阵
            Qtb = self.transition_model.get_Qt_bar(noisy_data["alpha_t_bar"], self.device)
            Qsb = self.transition_model.get_Qt_bar(noisy_data["alpha_s_bar"], self.device)
            Qt = self.transition_model.get_Qt(noisy_data["beta_t"], self.device)

            # Compute distributions to compare with KL
            bs, n, d = X.shape
            prob_true = diffusion_utils.posterior_distributions(
                X=X,
                E=E,
                X_t=noisy_data["X_t"],
                E_t=noisy_data["E_t"],
                charge=charge,
                charge_t=noisy_data["charge_t"],
                y_t=noisy_data["y_t"],
                Qt=Qt,
                Qsb=Qsb,
                Qtb=Qtb,
            )
            prob_true.E = prob_true.E.reshape((bs, n, n, -1))
            prob_pred = diffusion_utils.posterior_distributions(
                X=pred_probs_X,
                E=pred_probs_E,
                X_t=noisy_data["X_t"],
                E_t=noisy_data["E_t"],
                charge=pred_probs_charge,
                charge_t=noisy_data["charge_t"],
                y_t=noisy_data["y_t"],
                Qt=Qt,
                Qsb=Qsb,
                Qtb=Qtb,
            )
            prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        (
            prob_true_X,
            prob_true_E,
            prob_pred.X,
            prob_pred.E,
            prob_true.charge,
            prob_pred.charge,
        ) = diffusion_utils.mask_distributions(
            true_X=prob_true.X,
            true_E=prob_true.E,
            pred_X=prob_pred.X,
            pred_E=prob_pred.E,
            node_mask=node_mask,
            true_charge=prob_true.charge,
            pred_charge=prob_pred.charge,
        )
        kl_x = (self.test_X_kl if test else self.val_X_kl)(
            prob_true_X, torch.log(prob_pred.X)
        )
        kl_e = (self.test_E_kl if test else self.val_E_kl)(
            prob_true_E, torch.log(prob_pred.E)
        )

        assert (~(kl_x + kl_e).isnan()).any()
        loss = kl_x + kl_e

        if self.use_charge:
            kl_charge = (self.test_charge_kl if test else self.val_charge_kl)(
                prob_true.charge, torch.log(prob_pred.charge)
            )
            assert (~(kl_charge).isnan()).any()
            loss = loss + kl_charge

        return self.T * loss

    def reconstruction_logp(self, t, X, E, node_mask, charge):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        prob_charge0 = None
        if self.use_charge:
            prob_charge0 = charge @ Q0.charge

        sampled0 = diffusion_utils.sample_discrete_features(
            probX=probX0, probE=probE0, node_mask=node_mask, prob_charge=prob_charge0
        )

        X0 = F.one_hot(sampled0.X, num_classes=self.out_dims.X).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.out_dims.E).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        charge0 = X0.new_zeros((*X0.shape[:-1], 0))
        if self.use_charge:
            charge0 = F.one_hot(
                sampled0.charge, num_classes=self.out_dims.charge
            ).float()

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0, charge=charge0).mask(node_mask)

        # Predictions
        noisy_data = {
            "X_t": sampled_0.X,
            "E_t": sampled_0.E,
            "y_t": sampled_0.y,
            "node_mask": node_mask,
            "t_int": torch.zeros((X0.shape[0], 1), dtype=torch.long).to(self.device),
            "t_float": torch.zeros((X0.shape[0], 1), dtype=torch.float).to(self.device),
            "charge_t": sampled_0.charge,
        }
        sparse_noisy_data = utils.to_sparse(
            noisy_data["X_t"],
            noisy_data["E_t"],
            noisy_data["y_t"],
            node_mask,
            charge=noisy_data["charge_t"],
        )
        noisy_data.update(sparse_noisy_data)
        noisy_data["comp_edge_index_t"] = sparse_noisy_data["edge_index_t"]
        noisy_data["comp_edge_attr_t"] = sparse_noisy_data["edge_attr_t"]

        pred0 = self.forward(noisy_data)
        pred0, _ = utils.to_dense(
            pred0.node, pred0.edge_index, pred0.edge_attr, pred0.batch, pred0.charge
        )

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.out_dims.X).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(
            self.out_dims.E
        ).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.out_dims.E).type_as(probE0)

        assert (~probX0.isnan()).any()
        assert (~probE0.isnan()).any()

        prob_charge0 = charge
        if self.use_charge:
            prob_charge0 = F.softmax(pred0.charge, dim=-1)
            prob_charge0[~node_mask] = torch.ones(self.out_dims.charge).type_as(
                prob_charge0
            )
            assert (~prob_charge0.isnan()).any()

        return utils.PlaceHolder(X=probX0, E=probE0, y=None, charge=prob_charge0)

    def forward_sparse(self, sparse_noisy_data):
        node = sparse_noisy_data["node_t"]
        edge_attr = sparse_noisy_data["edge_attr_t"].float()
        edge_index = sparse_noisy_data["edge_index_t"].to(torch.int64)
        y = sparse_noisy_data["y_t"]
        batch = sparse_noisy_data["batch"].long()

        # Guard y dimension to match lin_in_y input (avoid shape mismatch)
        if (
            y is not None
            and hasattr(self.model, "lin_in_y")
            and self.model.lin_in_y is not None
            and y.dim() == 2
        ):
            expected_y = self.model.lin_in_y.in_features
            if y.size(-1) != expected_y:
                if y.size(-1) > expected_y:
                    y = y[:, :expected_y]
                else:
                    pad = y.new_zeros((y.size(0), expected_y - y.size(-1)))
                    y = torch.cat([y, pad], dim=-1)

        # 提取异质图元数据（如果启用）
        if self.heterogeneous and self.model.heterogeneous:
            from sparse_diffusion.utils_heterogeneous import extract_heterogeneous_metadata
            
            type_offsets = getattr(self.dataset_info, "type_offsets", None)
            node_type_names = getattr(self.dataset_info, "node_type_names", [])
            edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", None)
            fam_endpoints = getattr(self.dataset_info, "fam_endpoints", None)
            num_node_types = len(node_type_names) if node_type_names else 0
            # node 可能已拼接 [node_t, charge, extraX, signnet]，metadata 提取只能使用真实节点状态维度。
            node_for_metadata = node
            if node.dim() > 1 and self.out_dims.X > 0 and node.size(-1) > self.out_dims.X:
                node_for_metadata = node[:, : self.out_dims.X]
            
            metadata = extract_heterogeneous_metadata(
                node_t=node_for_metadata,
                edge_attr=edge_attr,
                edge_index=edge_index,
                type_offsets=type_offsets,
                node_type_names=node_type_names,
                edge_family_offsets=edge_family_offsets,
                fam_endpoints=fam_endpoints,
                num_node_types=num_node_types,
                num_edge_classes=self.out_dims.E,
            )
            
            return self.model(
                node, edge_attr, edge_index, y, batch,
                node_type_ids=metadata.get("node_type_ids"),
                node_subtype_ids=metadata.get("node_subtype_ids"),
                relation_type_ids=metadata.get("relation_type_ids"),
                edge_family_ids=metadata.get("edge_family_ids"),
            )
        else:
            return self.model(node, edge_attr, edge_index, y, batch)

    def forward(self, noisy_data):
        """
        noisy data contains: node_t, comp_edge_index_t, comp_edge_attr_t, batch
        """
        # build the sparse_noisy_data for the forward function of the sparse model
        sparse_noisy_data = self.compute_extra_data(sparse_noisy_data=noisy_data)

        if self.sign_net and self.cfg.model.extra_features == "all":
            x = self.sign_net(
                sparse_noisy_data["node_t"],
                sparse_noisy_data["edge_index_t"],
                sparse_noisy_data["batch"],
            )
            sparse_noisy_data["node_t"] = torch.hstack([sparse_noisy_data["node_t"], x])

        res = self.forward_sparse(sparse_noisy_data)

        return res

    @torch.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        num_nodes=None,
    ):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (node_types, charge, positions)
        """
        if num_nodes is None:
            num_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            num_nodes = num_nodes * torch.ones(
                batch_size, device=self.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            num_nodes = num_nodes
        num_max = torch.max(num_nodes)

        # Build the masks
        arange = (
            torch.arange(num_max, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        node_mask = arange < num_nodes.unsqueeze(1)

        # Sample noise (z_T): 异质图时用训练集每族平均边数初始化边，使 m 在首步就有合理起点
        if (self.heterogeneous
                and getattr(self.dataset_info, "edge_family_avg_edge_counts", None)
                and getattr(self.dataset_info, "fam_endpoints", None)
                and getattr(self.dataset_info, "type_offsets", None)):
            sparse_sampled_data = diffusion_utils.sample_sparse_discrete_feature_noise_heterogeneous(
                limit_dist=self.limit_dist, node_mask=node_mask, dataset_info=self.dataset_info,
                out_dims_E=self.out_dims.E, device=self.device
            )
        else:
            sparse_sampled_data = diffusion_utils.sample_sparse_discrete_feature_noise(
                limit_dist=self.limit_dist, node_mask=node_mask
            )

        # Anchor strategy for heterogeneous edge diffusion:
        # keep a fixed node-subtype snapshot from z_T to avoid recomputing per-step
        # relation buckets from changing node states.
        if self.heterogeneous:
            sparse_sampled_data.anchor_node_subtype = (
                sparse_sampled_data.node.argmax(dim=-1)
                if sparse_sampled_data.node.dim() > 1
                else sparse_sampled_data.node.long()
            )

        assert number_chain_steps < self.T
        chain = utils.SparseChainPlaceHolder(keep_chain=keep_chain)
        
        # 记录采样过程：初始化状态
        sampling_log = []
        if hasattr(self, 'local_rank') and self.local_rank == 0:
            # 记录初始状态
            init_edge_count = sparse_sampled_data.edge_index.shape[1] if sparse_sampled_data.edge_index.numel() > 0 else 0
            init_node_count = sparse_sampled_data.node.shape[0]
            edge_attr_discrete = sparse_sampled_data.edge_attr.argmax(dim=-1) if sparse_sampled_data.edge_attr.dim() > 1 else sparse_sampled_data.edge_attr
            non_zero_edges = (edge_attr_discrete != 0).sum().item() if edge_attr_discrete.numel() > 0 else 0
            sampling_log.append({
                'step': self.T,
                't_norm': 1.0,
                'num_nodes': init_node_count,
                'num_edges': init_edge_count,
                'num_nonzero_edges': non_zero_edges
            })
            print(f"[采样记录] 初始化: {init_node_count} 节点, {init_edge_count} 边 (其中 {non_zero_edges} 条非零边)")

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        time_range = torch.arange(0, self.T, self.cfg.general.skip)
        # 只在主进程（rank 0）显示进度条，避免多GPU时进度条跳动
        disable_tqdm = self.local_rank != 0 if hasattr(self, 'local_rank') else False
        for s_int in tqdm(reversed(time_range), total=self.T // self.cfg.general.skip, disable=disable_tqdm):
            s_array = (s_int * torch.ones((batch_size, 1))).to(self.device)
            t_array = s_array + int(self.cfg.general.skip)
            s_norm = s_array / self.T
            t_norm = t_array / self.T
            # print(s_norm, t_norm)

            # Sample z_s
            sparse_sampled_data = self.sample_p_zs_given_zt(
                s_norm, t_norm, sparse_sampled_data
            )
            
            # 记录采样过程：每一步的状态
            if hasattr(self, 'local_rank') and self.local_rank == 0:
                edge_count = sparse_sampled_data.edge_index.shape[1] if sparse_sampled_data.edge_index.numel() > 0 else 0
                node_count = sparse_sampled_data.node.shape[0]
                edge_attr_discrete = sparse_sampled_data.edge_attr.argmax(dim=-1) if sparse_sampled_data.edge_attr.dim() > 1 else sparse_sampled_data.edge_attr
                non_zero_edges = (edge_attr_discrete != 0).sum().item() if edge_attr_discrete.numel() > 0 else 0
                sampling_log.append({
                    'step': s_int,
                    't_norm': t_norm[0].item(),
                    'num_nodes': node_count,
                    'num_edges': edge_count,
                    'num_nonzero_edges': non_zero_edges
                })
                # 每10步或最后几步打印一次
                if s_int % (self.cfg.general.skip * 10) == 0 or s_int < self.cfg.general.skip * 3:
                    print(f"[采样记录] 步 {s_int} (t={t_norm[0].item():.3f}): {node_count} 节点, {edge_count} 边 (其中 {non_zero_edges} 条非零边)")

            # keep_chain can be very small, e.g., 1
            if ((s_int * number_chain_steps) % self.T == 0) and (keep_chain != 0):
                chain.append(sparse_sampled_data)

        # 记录最终状态
        if hasattr(self, 'local_rank') and self.local_rank == 0:
            final_edge_count = sparse_sampled_data.edge_index.shape[1] if sparse_sampled_data.edge_index.numel() > 0 else 0
            final_node_count = sparse_sampled_data.node.shape[0]
            edge_attr_discrete = sparse_sampled_data.edge_attr.argmax(dim=-1) if sparse_sampled_data.edge_attr.dim() > 1 else sparse_sampled_data.edge_attr
            final_non_zero_edges = (edge_attr_discrete != 0).sum().item() if edge_attr_discrete.numel() > 0 else 0
            sampling_log.append({
                'step': 0,
                't_norm': 0.0,
                'num_nodes': final_node_count,
                'num_edges': final_edge_count,
                'num_nonzero_edges': final_non_zero_edges
            })
            print(f"[采样记录] 最终: {final_node_count} 节点, {final_edge_count} 边 (其中 {final_non_zero_edges} 条非零边)")
            
            # 保存采样记录到文件
            try:
                import json
                current_path = os.getcwd()
                log_dir = os.path.join(current_path, f"sampling_logs/{self.cfg.general.name}/epoch{self.current_epoch}/")
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"batch_{batch_id}_sampling_log.json")
                # 将Tensor转换为Python原生类型
                sampling_log_serializable = []
                for entry in sampling_log:
                    serializable_entry = {
                        'step': int(entry['step']),
                        't_norm': float(entry['t_norm']),
                        'num_nodes': int(entry['num_nodes']),
                        'num_edges': int(entry['num_edges']),
                        'num_nonzero_edges': int(entry['num_nonzero_edges'])
                    }
                    sampling_log_serializable.append(serializable_entry)
                with open(log_file, 'w') as f:
                    json.dump(sampling_log_serializable, f, indent=2)
                print(f"[采样记录] 已保存到: {log_file}")
            except Exception as e:
                print(f"[采样记录] 保存失败: {e}")
        
        # get generated graphs
        generated_graphs = sparse_sampled_data.to_device("cpu")
        generated_graphs.edge_attr = sparse_sampled_data.edge_attr.argmax(-1)
        generated_graphs.node = sparse_sampled_data.node.argmax(-1)
        if self.use_charge:
            generated_graphs.charge = sparse_sampled_data.charge.argmax(-1) - 1
        if self.visualization_tools is not None:
            current_path = os.getcwd()

            # Visualize chains
            if keep_chain > 0:
                if hasattr(self, 'local_rank') and self.local_rank == 0:
                    print("Visualizing chains...")
                chain_path = os.path.join(
                    current_path,
                    f"chains/{self.cfg.general.name}/" f"epoch{self.current_epoch}/",
                )
                try:
                    _ = self.visualization_tools.visualize_chain(
                        chain_path, batch_id, chain, local_rank=self.local_rank
                    )
                except OSError:
                    if hasattr(self, 'local_rank') and self.local_rank == 0:
                        print("Warn: image chains failed to be visualized ")

            # Visualize the final molecules
            if hasattr(self, 'local_rank') and self.local_rank == 0:
                print("\nVisualizing molecules...")
            result_path = os.path.join(
                current_path,
                f"graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            try:
                self.visualization_tools.visualize(
                    result_path,
                    generated_graphs,
                    save_final,
                    local_rank=self.local_rank,
                )
            except OSError:
                if hasattr(self, 'local_rank') and self.local_rank == 0:
                    print("Warn: image failed to be visualized ")

            if hasattr(self, 'local_rank') and self.local_rank == 0:
                print("Done.")
        return generated_graphs

    def sample_node(self, pred_X, p_s_and_t_given_0_X, node_mask):
        # Normalize predictions
        pred_X = F.softmax(pred_X, dim=-1)  # bs, n, d0
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(
            unnormalized_prob_X, dim=-1, keepdim=True
        )  # bs, n, d_t

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()

        X_t = diffusion_utils.sample_discrete_node_features(prob_X, node_mask)

        return X_t, prob_X

    def sample_edge(self, pred_E, p_s_and_t_given_0_E, node_mask):
        # Normalize predictions
        bs, n, n, de = pred_E.shape
        pred_E = F.softmax(pred_E, dim=-1)  # bs, n, n, d0
        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E  # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(
            unnormalized_prob_E, dim=-1, keepdim=True
        )
        prob_E = prob_E.reshape(bs, n, n, de)

        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        E_t = diffusion_utils.sample_discrete_edge_features(prob_E, node_mask)

        return E_t, prob_E

    def sample_node_edge(
        self, pred, p_s_and_t_given_0_X, p_s_and_t_given_0_E, node_mask
    ):
        _, prob_X = self.sample_node(pred.X, p_s_and_t_given_0_X, node_mask)
        _, prob_E = self.sample_edge(pred.E, p_s_and_t_given_0_E, node_mask)

        sampled_s = diffusion_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask
        )

        return sampled_s

    def sample_sparse_node(self, pred_node, p_s_and_t_given_0_X, node_type_mask=None):
        """
        Sample node subtypes, with optional mask to restrict to same node type.
        
        Args:
            pred_node: (N, dx) predicted node logits
            p_s_and_t_given_0_X: (N, dx, dx) transition probabilities
            node_type_mask: (N, dx) mask where 1.0 allows sampling, 0.0 forbids.
                          If None, no restriction (backward compatibility).
        """
        # Normalize predictions
        pred_X = F.softmax(pred_node, dim=-1)  # N, dx
        # Dim of the second tensor: N, dx, dx
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # N, dx, dx
        unnormalized_prob_X = weighted_X.sum(dim=1)  # N, dx
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = (
            1e-5  # TODO: delete/masking?
        )
        prob_X = unnormalized_prob_X / torch.sum(
            unnormalized_prob_X, dim=-1, keepdim=True
        )  # N, dx
        
        # Apply node type mask: restrict each node to its own type's subtypes
        if node_type_mask is not None:
            prob_X = prob_X * node_type_mask
            row_sum = prob_X.sum(dim=-1, keepdim=True)
            all_zero = (row_sum.squeeze(-1) == 0)
            # If all subtypes are masked out, fallback to uniform within current type
            # (This should not happen if mask is correct, but safety check)
            if all_zero.any():
                # Fallback: use original probabilities for masked nodes
                prob_X[all_zero] = unnormalized_prob_X[all_zero] / unnormalized_prob_X[all_zero].sum(dim=-1, keepdim=True).clamp(min=1e-8)
            else:
                prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        X_t = prob_X.multinomial(1)[:, 0]

        return X_t

    def sample_sparse_edge(self, pred_edge, p_s_and_t_given_0_E, edge_type_mask=None):
        """
        边预测：可预测空间由关系族决定（edge_type_mask 已按 query_edge_family 限制）。

        - 机构-作者（隶属）：仅 no-edge / 隶属一种状态 → 等价于**存在性预测**；
        - 作者-论文（撰写）：no-edge / 一作 / 二作 / 通信等 → **先预测存在性，再在存在的边上预测子类别**。
        分层采样（hierarchical_edge_sampling）与训练时的分层损失一致：先「有无边」，再「有边时」采样子类型。
        """
        # Normalize predictions
        pred_E = F.softmax(pred_edge, dim=-1)  # N, d0
        # Dim of the second tensor: N, d0, dt-1
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E  # N, d0, dt-1
        unnormalized_prob_E = weighted_E.sum(dim=1)  # N, dt-1
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(
            unnormalized_prob_E, dim=-1, keepdim=True
        )
        
        # 异质图：仅允许 (src_type,dst_type) 在 fam_endpoints 中合法的边类型，避免 Paper->Org 的 author_of 等非法关系
        if edge_type_mask is not None:
            prob_E = prob_E * edge_type_mask
            row_sum = prob_E.sum(dim=-1, keepdim=True)
            all_zero = (row_sum.squeeze(-1) == 0)
            prob_E[all_zero, 0] = 1.0  # 无合法边类型时只允许 no-edge
            prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        # 分层采样（与训练时的分层损失一致）：先采样「有无边」，再在「有边」时采样子类型，避免 no-edge 单类概率 < 有边总和时仍被 argmax 选中的问题
        if getattr(self, "hierarchical_edge_sampling", False):
            no_edge_prob = prob_E[:, 0]  # (N,)
            exist_prob = prob_E[:, 1:].sum(dim=-1)  # (N,)
            u = torch.rand(prob_E.shape[0], device=prob_E.device, dtype=prob_E.dtype)
            has_edge = u < exist_prob
            E_t = torch.zeros(prob_E.shape[0], dtype=torch.long, device=prob_E.device)
            where_has = has_edge.nonzero(as_tuple=True)[0]
            if where_has.numel() > 0:
                sub = prob_E[where_has, 1:]
                sub_sum = sub.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                sub = sub / sub_sum
                idx = sub.multinomial(1)[:, 0]  # 0-based 子类型下标
                E_t[where_has] = idx + 1  # 全局 ID：1 ~ d0-1
            return E_t

        E_t = prob_E.multinomial(1)[:, 0]
        return E_t

    def sample_sparse_node_edge(
        self,
        pred_node,
        pred_edge,
        p_s_and_t_given_0_X,
        p_s_and_t_given_0_E,
        pred_charge,
        p_s_and_t_given_0_charge,
        edge_type_mask=None,
        node_type_mask=None,
    ):
        # 采样顺序：先节点后边
        # 节点主类型不变（只允许子类型变化），所以 edge_type_mask（基于主类型的关系族）不需要更新
        sampled_node = self.sample_sparse_node(pred_node, p_s_and_t_given_0_X, node_type_mask).long()
        sampled_edge = self.sample_sparse_edge(pred_edge, p_s_and_t_given_0_E, edge_type_mask).long()

        if pred_charge.size(-1) > 0:
            sampled_charge = self.sample_sparse_node(
                pred_charge, p_s_and_t_given_0_charge
            ).long()
        else:
            sampled_charge = pred_charge

        return sampled_node, sampled_edge, sampled_charge

    def sample_p_zs_given_zt(self, s_float, t_float, data):
        """
        Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well
        """
        node = data.node
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        y = data.y
        charge = data.charge
        ptr = data.ptr
        batch = data.batch

        # Heterogeneous sampling anchor: fixed node subtype ids from initial z_T.
        anchor_node_subtype = getattr(data, "anchor_node_subtype", None)
        if anchor_node_subtype is None:
            anchor_node_subtype = node.argmax(dim=-1) if node.dim() > 1 else node.long()
            data.anchor_node_subtype = anchor_node_subtype

        # 同一段连续步（例如 t, t+1, ..., t+skip-1）会共用同一个 β 和 ᾱ，时间被“粗粒度”成每 skip 步一块
        beta_t = self.noise_schedule(t_normalized=t_float, skip=True)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float, skip=True)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float, skip=True)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Prior distribution
        # 对于异质图，需要为每个节点类型分别计算后验分布，确保只考虑同一节点类型的子类别
        # out_dims.X是全局子类别数量，type_offsets 存的是：每种节点类型 对应的那一段子类别 ID 从几开始
        # node在流程里不同阶段用不同的张量形式
        # 在流程里不同阶段用哪种张量形式
        # 1.初始噪声：one-hot：形状 (N, dx)，每行一个 one-hot，表示该节点是哪一个全局子类别（dx = 全局子类别数）。
        # sample_sparse_discrete_feature_noise / sample_sparse_discrete_feature_noise_heterogeneous 里用 F.one_hot(..., num_classes=...)，所以 node 一开始是 (N, dx)。
        # 和转移矩阵 Q 做运算时，也按“每个节点一个 dx 维分布”来算，所以内部会保持 (N, dx) 这种形式。
        # 2. 采样：每一步采样：形状 (N,)，每个元素是全局子类别 id（0 到 out_dims.X - 1）。
        # sample_sparse_node 返回的是 multinomial(1)[:, 0]，即 离散类别下标 (N,)。写回时是 data.node = new_node，没有再转成 one-hot，所以 从第一步采样之后，data.node 就变成 (N,) 的离散 id。
        # 第一次进入 sample_p_zs_given_zt 时，data 是初始噪声，node 是 one-hot (N, dx)。之后每一步，data 是上一步采样结果，node 已经是离散 (N,)。
        if self.heterogeneous and hasattr(self.dataset_info, "type_offsets") and len(getattr(self.dataset_info, "type_offsets", {})) > 0:
            type_offsets = self.dataset_info.type_offsets
            # 获取当前节点的子类别ID
            current_node_subtype = node.argmax(dim=-1) if node.dim() > 1 else node.long()  # 当 node.dim() > 1 时在最后一维上取而当 node.dim() == 1 时 可能是 float 或别的类型 .long() 把它转成整型（int64），形状不变，仍是 (N,)
            # current_node_subtype 始终是 形状 (N,) 的全局子类别 id，后面用 type_offsets 和 type_sizes 再按类型切块，用 type_offsets + type_sizes 判断每个节点属于哪种类型

            # 计算每个节点类型的大小，
            sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
            type_sizes = {}
            for i, (t_name, off) in enumerate(sorted_types):
                if i + 1 < len(sorted_types):
                    type_sizes[t_name] = sorted_types[i + 1][1] - off
                else:
                    # 最后一个类型
                    type_sizes[t_name] = self.out_dims.X - off
            # type_offsets 按 offset 排序后，相邻两项的 offset 之差 = 前一种类型的子类别个数

            # 初始化全局后验分布
            num_nodes = node.shape[0]
            num_global_states = self.out_dims.X
            p_s_and_t_given_0_X = torch.zeros(
                (num_nodes, num_global_states, num_global_states),
                device=self.device
            )
            
            # 为每个节点类型分别计算后验分布
            for t_name, offset in sorted_types:
                type_size = type_sizes.get(t_name, 0)
                if type_size <= 0:
                    continue
                
                # 找到属于该类型的节点
                if t_name == sorted_types[-1][0]:
                    # 最后一个类型 type_mask = 形状 (N,) 的 bool，True 表示「该节点属于当前类型 t_name」
                    type_mask = current_node_subtype >= offset
                else:
                    next_offset = sorted_types[sorted_types.index((t_name, offset)) + 1][1]
                    type_mask = (current_node_subtype >= offset) & (current_node_subtype < next_offset)
                
                if not type_mask.any():
                    continue
                
                # 获取该类型的节点
                type_nodes = node[type_mask]  # (num_type_nodes, dx)
                type_batch = batch[type_mask]  # (num_type_nodes,)
                
                # 切片转移矩阵：只使用该节点类型范围内的转移矩阵
                # Qt.X: (bs, dx, dx) -> (bs, type_size, type_size)
                Qt_X_type = Qt.X[:, offset:offset+type_size, offset:offset+type_size]  # (bs, type_size, type_size)
                Qsb_X_type = Qsb.X[:, offset:offset+type_size, offset:offset+type_size]  # (bs, type_size, type_size)
                Qtb_X_type = Qtb.X[:, offset:offset+type_size, offset:offset+type_size]  # (bs, type_size, type_size)
                
                # 切片节点特征：只使用该节点类型范围内的特征
                # 将全局子类别ID转换为局部子类别ID
                type_nodes_local = type_nodes.clone()
                if type_nodes_local.dim() > 1:
                    # one-hot编码：只保留该类型范围内的维度
                    type_nodes_local = type_nodes_local[:, offset:offset+type_size]  # (num_type_nodes, type_size)
                else:
                    # 离散ID：转换为局部ID
                    type_nodes_local = type_nodes_local - offset
                    type_nodes_local = F.one_hot(type_nodes_local.long(), num_classes=type_size).float()
                
                # 计算该节点类型的后验分布 p(s和t相关量|给定 0)
                p_s_and_t_given_0_X_type = (
                    diffusion_utils.compute_sparse_batched_over0_posterior_distribution(
                        input_data=type_nodes_local, batch=type_batch, 
                        Qt=Qt_X_type, Qsb=Qsb_X_type, Qtb=Qtb_X_type
                    )
                )  # (num_type_nodes, type_size, type_size)
                
                # 将局部后验分布映射回全局状态空间
                # p_s_and_t_given_0_X_type: (num_type_nodes, type_size, type_size)
                # 需要映射到: (num_type_nodes, num_global_states, num_global_states)
                for local_from in range(type_size):
                    global_from = offset + local_from
                    for local_to in range(type_size):
                        global_to = offset + local_to
                        p_s_and_t_given_0_X[type_mask, global_from, global_to] = p_s_and_t_given_0_X_type[:, local_from, local_to]
        else:
            # 同质图模式：使用全局转移矩阵
            p_s_and_t_given_0_X = (
                diffusion_utils.compute_sparse_batched_over0_posterior_distribution(
                    input_data=node, batch=batch, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
                )
            )

        p_s_and_t_given_0_charge = None
        if self.use_charge:
            p_s_and_t_given_0_charge = (
                diffusion_utils.compute_sparse_batched_over0_posterior_distribution(
                    input_data=charge,
                    batch=batch,
                    Qt=Qt.charge,
                    Qsb=Qsb.charge,
                    Qtb=Qtb.charge,
                )
            )

        # 以上计算完了节点的转移矩阵（用于后续在提供解析后验的形式），接下来要构建查询边，结合已有的噪声边构建出MP的边

        # prepare sparse information
        num_nodes = ptr.diff().long()
        num_edges = (num_nodes * (num_nodes - 1) / 2).long()
        num_edges_ptr = torch.hstack(
            [torch.tensor([0]).to(self.device), num_edges.cumsum(-1)]
        ).long()

        # 初始化变量（用于异质图模式和同质图模式）
        all_query_edge_index = None
        all_query_edge_batch = None
        all_condensed_index = None
        all_edge_batch = None
        all_edge_mask = None

        # 下面是要构建查询边，结合已有的噪声边构建出MP的边

        # 检查是否为异质图模式，如果是，需要按关系族分别进行均匀采样
        # 根据图片要求：|Eq| = km，其中 m 是真实边数，k 是倍数（通过 edge_fraction 控制）
        if self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets") and len(self.dataset_info.edge_family_offsets) > 0:
            # 异质图模式：为每个关系族分别进行均匀采样
            # 获取关系族信息
            edge_family2id = getattr(self.dataset_info, "edge_family2id", {})
            id2edge_family = {v: k for k, v in edge_family2id.items()}
            fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
            
            # 获取节点类型信息
            type_offsets = getattr(self.dataset_info, "type_offsets", {})
            node_type_names = getattr(self.dataset_info, "node_type_names", [])
            
            # 如果 type_offsets 不存在，尝试从 meta.json 推断（不加载 vocab.json）
            if not type_offsets and node_type_names:
                import os.path as osp
                import json
                vocab_path = osp.join(getattr(self.dataset_info, "vocab_path", ""), "vocab.json")
                if not vocab_path or not osp.exists(vocab_path):
                    if hasattr(self.dataset_info, "datamodule") and hasattr(self.dataset_info.datamodule, "inner"):
                        vocab_path = osp.join(self.dataset_info.datamodule.inner.processed_dir, "vocab.json")
                
                if osp.exists(vocab_path) and hasattr(self.dataset_info, "datamodule") and hasattr(self.dataset_info.datamodule, "inner"):
                    subgraph_dirs = [d for d in os.listdir(self.dataset_info.datamodule.inner.root) 
                                    if osp.isdir(osp.join(self.dataset_info.datamodule.inner.root, d)) and d.startswith("subgraph_")]
                    if len(subgraph_dirs) > 0:
                        meta0 = json.load(open(osp.join(self.dataset_info.datamodule.inner.root, subgraph_dirs[0], "meta.json"), "r", encoding="utf-8"))
                        schema_by_type = meta0.get("schema_by_type", {})
                        type_sizes = [len(schema_by_type.get(t, [])) for t in node_type_names]
                        type_offsets = {}
                        cur = 0
                        for t, size in zip(node_type_names, type_sizes):
                            type_offsets[t] = cur
                            cur += size
            
            # 为每个关系族分别生成查询边
            # 根据图片要求：|Eq| = km，其中 m 使用 dataset 预存的各关系族平均边数（edge_family_avg_edge_counts）
            edge_family_offsets = self.dataset_info.edge_family_offsets
            
            all_query_edge_index_list = []
            all_query_edge_batch_list = []
            all_query_edge_family_list = []  # 记录每条查询边属于哪个关系族
            
            bs = len(num_nodes)
            node_t = anchor_node_subtype  # (N,) - 使用锚定子类别ID进行关系族查询边划分
            
            for fam_id, fam_name in id2edge_family.items():
                if fam_name not in fam_endpoints:
                    continue
                
                src_type = fam_endpoints[fam_name]["src_type"]
                dst_type = fam_endpoints[fam_name]["dst_type"]
                
                # 使用保存的平均真实边数 m_fam（各关系族保持一致）
                # 根据图片要求：|Eq| = km，其中 m 是各关系族在真实图中的数量，k 是各关系族都保持一致的倍数
                edge_family_avg_edge_counts = getattr(self.dataset_info, "edge_family_avg_edge_counts", {})
                if fam_name in edge_family_avg_edge_counts:
                    avg_m_fam = edge_family_avg_edge_counts[fam_name]
                else:
                    # 如果没有保存的平均值，使用默认值
                    avg_m_fam = 10.0
                
                # 计算每个批次中该关系族的 src_type 和 dst_type 节点数
                src_mask = None
                dst_mask = None
                if type_offsets and src_type in type_offsets and dst_type in type_offsets:
                    src_offset = type_offsets[src_type]
                    dst_offset = type_offsets[dst_type]
                    
                    # 计算每个节点类型的 size
                    type_sizes = {}
                    sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                    for i, (t, off) in enumerate(sorted_types):
                        if i + 1 < len(sorted_types):
                            type_sizes[t] = sorted_types[i + 1][1] - off
                        else:
                            type_sizes[t] = self.out_dims.X - off
                    
                    src_size = type_sizes.get(src_type, 0)
                    dst_size = type_sizes.get(dst_type, 0)
                    
                    # 计算每个批次中 src_type 和 dst_type 的节点数
                    src_mask = (node_t >= src_offset) & (node_t < src_offset + src_size)  # (N,)
                    dst_mask = (node_t >= dst_offset) & (node_t < dst_offset + dst_size)  # (N,)
                    
                    # 为每个批次生成该关系族的查询边
                    # 使用类似 sampled_condensed_indices_uniformly 的方式，但适用于有向图的关系族采样
                    for b in range(bs):
                        batch_mask = (batch == b)
                        batch_src_nodes = torch.where(src_mask & batch_mask)[0]  # 全局节点索引
                        batch_dst_nodes = torch.where(dst_mask & batch_mask)[0]  # 全局节点索引
                        
                        if len(batch_src_nodes) == 0 or len(batch_dst_nodes) == 0:
                            continue
                        
                        # 使用保存的平均真实边数 m_fam（各关系族保持一致）
                        m_fam = avg_m_fam
                        
                        # 计算该批次该关系族的可能边数（有向图：src_nodes * dst_nodes，排除自环）
                        num_src = len(batch_src_nodes)
                        num_dst = len(batch_dst_nodes)
                        num_fam_possible_edges = num_src * num_dst
                        if src_type == dst_type:
                            # 如果是同类型，需要排除自环
                            num_fam_possible_edges = num_src * num_dst - num_src
                        
                        # 根据图片要求：|Eq| = km，其中 k 是倍数（通过 edge_fraction 控制）
                        # 这里使用 edge_fraction 作为 k，即 |Eq_fam| = edge_fraction * m_fam
                        # 但需要确保不超过可能的边数
                        # 采样时：每族至少查询 min(20, num_fam_possible_edges) 条边，给模型更多机会
                        # 生成边，缓解 Disconnected 100%（训练仍用 k*m_fam，此处仅采样路径）
                        k = self.edge_fraction  # 倍数
                        num_query_edges_fam = int(math.ceil(k * m_fam)) if m_fam > 0 else 0
                        num_query_edges_fam = min(num_fam_possible_edges, max(num_query_edges_fam, 20))
                        
                        if num_query_edges_fam == 0:
                            continue
                        
                        # 使用 condensed_index / flat_index 方式采样（与原项目一致）
                        # 将每个关系族视为一个独立的子图，使用 condensed_index 采样
                        if src_type == dst_type:
                            # 同类型：使用上三角矩阵的 condensed_index（排除自环）
                            num_fam_nodes = num_src  # 该关系族的节点数
                            max_condensed_value_fam = num_fam_nodes * (num_fam_nodes - 1) // 2
                            
                            if max_condensed_value_fam > 0 and num_query_edges_fam > 0:
                                # 使用 sampled_condensed_indices_uniformly 采样
                                num_query_edges_fam_tensor = torch.tensor([num_query_edges_fam], device=self.device, dtype=torch.long)
                                max_condensed_value_fam_tensor = torch.tensor([max_condensed_value_fam], device=self.device, dtype=torch.long)
                                
                                sampled_condensed_fam, _ = sampled_condensed_indices_uniformly(
                                    max_condensed_value=max_condensed_value_fam_tensor,
                                    num_edges_to_sample=num_query_edges_fam_tensor,
                                    return_mask=False
                                )
                                
                                # 将 condensed_index 转换为 matrix_index（相对于 batch_src_nodes）
                                fam_query_edge_index_local = condensed_to_matrix_index_batch(
                                    condensed_index=sampled_condensed_fam,
                                    num_nodes=torch.tensor([num_fam_nodes], device=self.device, dtype=torch.long),
                                    edge_batch=torch.zeros(len(sampled_condensed_fam), device=self.device, dtype=torch.long),
                                    ptr=torch.tensor([0, num_fam_nodes], device=self.device, dtype=torch.long),
                                ).long()
                                
                                # 边界检查：确保索引在有效范围内
                                # 注意：fam_query_edge_index_local 是相对于 num_fam_nodes 的局部索引
                                # 需要确保它不超过 batch_src_nodes 的长度
                                valid_mask = (fam_query_edge_index_local[0] >= 0) & (fam_query_edge_index_local[0] < num_fam_nodes) & \
                                            (fam_query_edge_index_local[1] >= 0) & (fam_query_edge_index_local[1] < num_fam_nodes) & \
                                            (fam_query_edge_index_local[0] < len(batch_src_nodes)) & \
                                            (fam_query_edge_index_local[1] < len(batch_src_nodes))
                                if not valid_mask.all():
                                    # 过滤无效索引
                                    fam_query_edge_index_local = fam_query_edge_index_local[:, valid_mask]
                                    if fam_query_edge_index_local.shape[1] == 0:
                                        continue
                                
                                # 将局部索引转换回全局节点索引
                                # 再次检查索引范围
                                if (fam_query_edge_index_local[0].max() >= len(batch_src_nodes)) or \
                                   (fam_query_edge_index_local[1].max() >= len(batch_src_nodes)):
                                    # 如果索引超出范围，跳过
                                    continue
                                
                                fam_query_edge_index = torch.stack([
                                    batch_src_nodes[fam_query_edge_index_local[0]],
                                    batch_src_nodes[fam_query_edge_index_local[1]]
                                ], dim=0)
                            else:
                                continue
                        else:
                            # 不同类型：有向图，直接使用 src*dst 的矩阵进行抽样
                            # 将 src*dst 的矩阵展平为一维数组，索引范围是 [0, num_src * num_dst)
                            # 使用 sampled_condensed_indices_uniformly 从该范围中均匀采样
                            max_condensed_value_fam = num_src * num_dst
                            
                            if max_condensed_value_fam > 0 and num_query_edges_fam > 0:
                                # 使用 sampled_condensed_indices_uniformly 采样
                                num_query_edges_fam_tensor = torch.tensor([num_query_edges_fam], device=self.device, dtype=torch.long)
                                max_condensed_value_fam_tensor = torch.tensor([max_condensed_value_fam], device=self.device, dtype=torch.long)
                                
                                sampled_flat_indices, _ = sampled_condensed_indices_uniformly(
                                    max_condensed_value=max_condensed_value_fam_tensor,
                                    num_edges_to_sample=num_query_edges_fam_tensor,
                                    return_mask=False
                                )
                                
                                # 将展平的索引转换为 (src_idx, dst_idx) 的矩阵坐标
                                # flat_idx = src_idx * num_dst + dst_idx
                                src_indices_local = sampled_flat_indices // num_dst
                                dst_indices_local = sampled_flat_indices % num_dst
                                
                                # 边界检查：确保索引在有效范围内
                                valid_mask = (src_indices_local >= 0) & (src_indices_local < num_src) & \
                                            (dst_indices_local >= 0) & (dst_indices_local < num_dst) & \
                                            (src_indices_local < len(batch_src_nodes)) & \
                                            (dst_indices_local < len(batch_dst_nodes))
                                if not valid_mask.all():
                                    # 过滤无效索引
                                    src_indices_local = src_indices_local[valid_mask]
                                    dst_indices_local = dst_indices_local[valid_mask]
                                    if len(src_indices_local) == 0:
                                        continue
                                
                                # 再次检查索引范围
                                if (src_indices_local.max() >= len(batch_src_nodes)) or \
                                   (dst_indices_local.max() >= len(batch_dst_nodes)):
                                    # 如果索引超出范围，跳过
                                    continue
                                
                                # 将局部索引转换回全局节点索引
                                fam_query_edge_index = torch.stack([
                                    batch_src_nodes[src_indices_local],
                                    batch_dst_nodes[dst_indices_local]
                                ], dim=0)
                            else:
                                continue
                        
                        if fam_query_edge_index.shape[1] > 0:
                            fam_query_edge_batch = torch.full((fam_query_edge_index.shape[1],), b, dtype=torch.long, device=self.device)
                            fam_query_edge_family = torch.full((fam_query_edge_index.shape[1],), fam_id, dtype=torch.long, device=self.device)  # 记录关系族ID
                            all_query_edge_index_list.append(fam_query_edge_index)
                            all_query_edge_batch_list.append(fam_query_edge_batch)
                            all_query_edge_family_list.append(fam_query_edge_family)
            
            # 合并所有关系族的查询边
            if len(all_query_edge_index_list) > 0:
                all_query_edge_index = torch.cat(all_query_edge_index_list, dim=1)  # (2, E_query)
                all_query_edge_batch = torch.cat(all_query_edge_batch_list)  # (E_query,)
                if len(all_query_edge_family_list) > 0:
                    all_query_edge_family = torch.cat(all_query_edge_family_list)  # (E_query,) - 每条边的关系族ID
                else:
                    all_query_edge_family = None  # 如果没有关系族信息，设为None
                
                # 调试信息：记录查询边集大小（仅在第一步记录，通过t_norm判断）
                if hasattr(self, 'local_rank') and self.local_rank == 0 and t_float[0].item() > 0.95:
                    num_existing_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
                    num_query_edges = all_query_edge_index.shape[1]
                    print(f"[DEBUG] 初始化: 上一步图内边数={num_existing_edges}, 本步候选边位数={num_query_edges}")
                
                # 计算每个批次的查询边数（用于后续的循环处理）
                num_edges_per_batch = torch.zeros(bs, dtype=torch.long, device=self.device)
                for b in range(bs):
                    num_edges_per_batch[b] = (all_query_edge_batch == b).sum()
                
                # 使用所有关系族的查询边总数作为参考（保持与训练时一致）
                num_edges_total = all_query_edge_index.shape[1]
                # 为了保持与训练时一致，使用每个批次的最大查询边数
                num_edges = num_edges_per_batch.max().unsqueeze(0) if num_edges_per_batch.max() > 0 else num_edges
            else:
                # 如果没有关系族信息，回退到全局采样
                all_query_edge_index = None
                all_query_edge_batch = None
                all_query_edge_family = None
                num_edges = (num_nodes * (num_nodes - 1) / 2).long()
        
        # 如果不是异质图或没有关系族信息，使用全局采样
        if not (self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets") and len(self.dataset_info.edge_family_offsets) > 0) or (all_query_edge_index is None):
            # If we had one graph, we will iterate on all edges for each step
            # we also make sure that the non existing edge number remains the same with the training process
            (
                all_condensed_index,
                all_edge_batch,
                all_edge_mask,
            ) = sampled_condensed_indices_uniformly(
                max_condensed_value=num_edges,
                num_edges_to_sample=num_edges,
                return_mask=True,
            )  # double checked
            all_query_edge_index = None  # 标记使用condensed索引
            all_query_edge_batch = None
        
        # number of edges used per loop for each graph
        num_edges_per_loop = torch.ceil(self.edge_fraction * num_edges)  # (bs, )
        len_loop = math.ceil(1.0 / self.edge_fraction)

        new_edge_index, new_edge_attr, new_charge = (
            torch.zeros((2, 0), device=self.device, dtype=torch.long),
            torch.zeros(0, device=self.device),
            torch.zeros(0, device=self.device, dtype=torch.long),
        )
        # 异质边类型 mask 用「当前最佳」节点类型：首轮用噪声 node，后续用上轮采样 new_node
        new_node = node.argmax(dim=-1) if node.dim() > 1 else node

        # create the new data for calculation
        sparse_noisy_data = {
            "node_t": node,
            "edge_index_t": edge_index,
            "edge_attr_t": edge_attr,
            "batch": batch,
            "y_t": y,
            "ptr": ptr,
            "charge_t": charge,
            "t_int": (t_float * self.T).int(),
            "t_float": t_float,
        }

        for i in range(len_loop):
            if self.autoregressive and i != 0:
                sparse_noisy_data["edge_index_t"] = new_edge_index
                sparse_noisy_data["edge_attr_t"] = new_edge_attr

            # 初始化 edges_to_keep_mask_sorted（用于最后一个循环的过滤）
            edges_to_keep_mask_sorted = None
            
            # 检查是否为异质图模式，使用不同的采样方式
            if self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets") and len(self.dataset_info.edge_family_offsets) > 0 and all_query_edge_index is not None:
                # 异质图模式：使用按关系族采样的查询边
                # 对所有关系族的查询边进行均匀采样（每个循环采样一部分）
                num_query_edges_total = num_edges_total
                
                # 计算每个循环要处理的边数（按edge_fraction比例）
                if i == 0:
                    # 第一次循环，打乱所有查询边（实现均匀采样）
                    perm = torch.randperm(num_query_edges_total, device=self.device)
                    all_query_edge_index = all_query_edge_index[:, perm]
                    all_query_edge_batch = all_query_edge_batch[perm]
                    if all_query_edge_family is not None:
                        all_query_edge_family = all_query_edge_family[perm]  # 同时打乱关系族信息
                
                # 计算当前循环要采样的边索引范围
                num_query_edges_per_loop = int(math.ceil(num_query_edges_total * self.edge_fraction))
                start_idx = i * num_query_edges_per_loop
                end_idx = min((i + 1) * num_query_edges_per_loop, num_query_edges_total)
                
                if start_idx < num_query_edges_total:
                    # 选择当前循环的边
                    triu_query_edge_index = all_query_edge_index[:, start_idx:end_idx]
                    query_edge_batch = all_query_edge_batch[start_idx:end_idx]
                    query_edge_family = all_query_edge_family[start_idx:end_idx] if all_query_edge_family is not None else None
                else:
                    # 如果已经采样完所有边，使用空边
                    triu_query_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                    query_edge_batch = torch.empty((0,), dtype=torch.long, device=self.device)
                    query_edge_family = None
            else:
                # 同质图模式或异质图但无查询边：使用 condensed 索引采样；关系族信息不适用
                query_edge_family = None
                # the last loop might have less edges, we need to make sure that each loop has the same number of edges
                if i == len_loop - 1:
                    edges_to_consider_mask = all_edge_mask >= (
                        num_edges[all_edge_batch] - num_edges_per_loop[all_edge_batch]
                    )
                    edges_to_keep_mask = torch.logical_and(
                        all_edge_mask >= num_edges_per_loop[all_edge_batch] * i,
                        all_edge_mask < num_edges_per_loop[all_edge_batch] * (i + 1),
                    )

                    triu_query_edge_index_condensed = all_condensed_index[edges_to_consider_mask]
                    query_edge_batch = all_edge_batch[edges_to_consider_mask]
                    condensed_query_edge_index = (
                        triu_query_edge_index_condensed + num_edges_ptr[query_edge_batch]
                    )
                    condensed_query_edge_index, condensed_query_edge_index_argsort = (
                        condensed_query_edge_index.sort()
                    )
                    edges_to_keep_mask_sorted = edges_to_keep_mask[edges_to_consider_mask][
                        condensed_query_edge_index_argsort
                    ]
                else:
                    # [0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1]
                    # all_condensed_index is not sorted inside the graph, but it sorted for graph batch
                    edges_to_consider_mask = torch.logical_and(
                        all_edge_mask >= num_edges_per_loop[all_edge_batch] * i,
                        all_edge_mask < num_edges_per_loop[all_edge_batch] * (i + 1),
                    )

                # get query edges and pass to matrix index
                triu_query_edge_index_condensed = all_condensed_index[edges_to_consider_mask]
                query_edge_batch = all_edge_batch[edges_to_consider_mask]
                # the order of edges does not change
                triu_query_edge_index = condensed_to_matrix_index_batch(
                    condensed_index=triu_query_edge_index_condensed,
                    num_nodes=num_nodes,
                    edge_batch=query_edge_batch,
                    ptr=ptr,
                ).long()

            # concatenate query edges and existing edges together to get the computational graph
            # clean_edge_attr has the priority
            # 异质图的关系是有向的，不应该转换为无向边
            # 采样时：保持有向边结构（for_message_passing=False）
            # 训练时：使用双向边支持消息传递（for_message_passing=True）
            query_mask, comp_edge_index, comp_edge_attr = get_computational_graph(
                triu_query_edge_index=triu_query_edge_index,
                clean_edge_index=sparse_noisy_data["edge_index_t"],
                clean_edge_attr=sparse_noisy_data["edge_attr_t"],
                heterogeneous=self.heterogeneous,
                for_message_passing=False,  # 采样时保持有向边结构
            )
            
            # 异质图：在循环内部生成 edge_type_mask，直接使用 query_edge_family
            # 每条边的可预测空间 = no-edge + 该关系族的子类型（如隶属仅 no-edge/隶属→存在性；撰写为 no-edge/一作/二作/通信→存在性+子类别）
            edge_type_mask = None
            if (self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets")
                    and len(getattr(self.dataset_info, "edge_family_offsets", {})) > 0
                    and query_edge_family is not None):
                # 使用查询边的关系族信息直接生成mask
                num_query_edges = query_mask.sum().item()
                if num_query_edges > 0:
                    edge_family_offsets = self.dataset_info.edge_family_offsets
                    edge_family2id = getattr(self.dataset_info, "edge_family2id", {})
                    id2edge_family = {v: k for k, v in edge_family2id.items()}
                    
                    # 获取查询边在comp_edge_index中的位置（query_mask标记的边）
                    query_edge_indices = torch.where(query_mask)[0]  # 在comp_edge_index中的索引
                    num_q = len(query_edge_indices)
                    
                    # 构建关系族ID到边类型范围的映射
                    fam_ranges = {}
                    for fam_name, offset in edge_family_offsets.items():
                        next_o = self.out_dims.E
                        for o2 in edge_family_offsets.values():
                            if o2 > offset and o2 < next_o:
                                next_o = o2
                        fam_ranges[fam_name] = (offset, min(next_o, self.out_dims.E))
                    
                    # 为每条查询边生成mask（按 comp 中 query 边的顺序，即 query_mask_indices 顺序）
                    mask = torch.zeros((num_q, self.out_dims.E), device=self.device)
                    
                    # coalesce 会重排边顺序，query_mask 中 True 的顺序 ≠ triu_query_edge_index 的顺序，
                    # 必须用 (batch, src, dst) 把 comp 中的查询边与 query_edge_family 对齐
                    query_mask_indices = torch.where(query_mask)[0]  # 所有查询边在 comp_edge_index 中的位置
                    comp_batch = batch[comp_edge_index[0]]  # (num_comp_edges,)
                    comp_src = comp_edge_index[0]
                    comp_dst = comp_edge_index[1]
                    # 当前轮查询边的 (batch, src, dst) -> fam_id；to_undirected 后 comp 含 (s,d) 与 (d,s)，反向也需同族约束
                    key_to_fam = {}
                    for k in range(triu_query_edge_index.shape[1]):
                        b = query_edge_batch[k].item()
                        s = triu_query_edge_index[0, k].item()
                        d = triu_query_edge_index[1, k].item()
                        fam_id = query_edge_family[k].item()
                        key_to_fam[(b, s, d)] = fam_id
                        key_to_fam[(b, d, s)] = fam_id
                    # 按 comp 中查询边顺序填 mask，并统计对齐失败数以便判断约束是否生效
                    unmatched = 0
                    for idx_in_mask, idx_in_comp in enumerate(query_mask_indices):
                        b = comp_batch[idx_in_comp].item()
                        s = comp_src[idx_in_comp].item()
                        d = comp_dst[idx_in_comp].item()
                        fam_id = key_to_fam.get((b, s, d), None)
                        if fam_id is not None:
                            fam_name = id2edge_family.get(fam_id, None)
                            if fam_name and fam_name in fam_ranges:
                                st, en = fam_ranges[fam_name]
                                mask[idx_in_mask, 0] = 1.0  # no-edge 始终允许
                                for gid in range(st, en):
                                    mask[idx_in_mask, gid] = 1.0
                            else:
                                mask[idx_in_mask, :] = 1.0
                        else:
                            # 该边是合并时与已有边重复、被标成 query 的，或跨轮重复，允许所有类型
                            unmatched += 1
                            mask[idx_in_mask, :] = 1.0
                    if num_q > 0:
                        unmatched_ratio = unmatched / num_q
                        print(f"[DEBUG] edge_type_mask 对齐: unmatched={unmatched}, total={num_q}, 约束生效比例={1 - unmatched_ratio:.2%}")
                        # 超过阈值说明关系族约束在退化，直接报错便于排查
                        if unmatched_ratio > 0.01:
                            raise ValueError(
                                f"edge_type_mask 对齐失败过多: unmatched={unmatched}, total={num_q}, "
                                f"比例={unmatched_ratio:.2%} > 1%。请检查 key_to_fam 与 comp 中查询边来源。"
                            )
                    edge_type_mask = mask

            # add computational graph
            sparse_noisy_data["comp_edge_index_t"] = comp_edge_index
            sparse_noisy_data["comp_edge_attr_t"] = comp_edge_attr
            sparse_pred = self.forward(sparse_noisy_data)

            # get_s_and_t_given_0_E for computational edges: (NE, de, de)
            # 检查是否为异质图模式，如果是，需要为每个关系族计算独立的转移概率
            if self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets") and len(self.dataset_info.edge_family_offsets) > 0:
                # 异质图模式：为每个关系族计算独立的转移概率
                # 根据边的全局 ID 推断 edge_family
                comp_edge_attr_discrete = comp_edge_attr.argmax(dim=-1)  # (NE,)
                edge_family_offsets = self.dataset_info.edge_family_offsets
                edge_family2id = getattr(self.dataset_info, "edge_family2id", {})
                id2edge_family = {v: k for k, v in edge_family2id.items()}
                
                # 获取所有关系族的转移矩阵
                all_family_qt = self.transition_model.get_all_family_Qt(beta_t, device=self.device)
                all_family_qsb = self.transition_model.get_all_family_Qt_bar(alpha_s_bar, device=self.device)
                all_family_qtb = self.transition_model.get_all_family_Qt_bar(alpha_t_bar, device=self.device)
                
                # 为每个关系族计算转移概率
                p_s_and_t_given_0_E_list = []
                comp_edge_batch = batch[comp_edge_index[0]]
                
                # 获取节点类型信息（用于确定 no-edge 位置属于哪个关系族）
                type_offsets = getattr(self.dataset_info, "type_offsets", {})
                fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
                current_node_subtype = anchor_node_subtype
                
                # 推断每个节点的类型
                node_type_ids = torch.zeros_like(current_node_subtype) - 1  # -1 表示未知
                if type_offsets:
                    sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                    for i, (t_name, off) in enumerate(sorted_types):
                        if i + 1 < len(sorted_types):
                            next_offset = sorted_types[i + 1][1]
                            type_mask = (current_node_subtype >= off) & (current_node_subtype < next_offset)
                        else:
                            type_mask = current_node_subtype >= off
                        node_type_ids[type_mask] = i
                
                for fam_id, fam_name in id2edge_family.items():
                    # 判断哪些边属于这个关系族
                    offset = edge_family_offsets.get(fam_name, 0)
                    # 获取该关系族的边数（从 offset 到下一个 offset 或全局边数）
                    if fam_name not in edge_family_offsets:
                        continue
                    
                    # 计算该关系族的边 ID 范围
                    next_offset = self.out_dims.E
                    for other_fam_name, other_offset in edge_family_offsets.items():
                        if other_offset > offset and other_offset < next_offset:
                            next_offset = other_offset
                    
                    # 获取该关系族的端点类型
                    if fam_name not in fam_endpoints:
                        continue
                    src_type = fam_endpoints[fam_name]["src_type"]
                    dst_type = fam_endpoints[fam_name]["dst_type"]
                    
                    # 找到该关系族对应的节点类型索引
                    src_type_idx = None
                    dst_type_idx = None
                    if type_offsets:
                        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                        for idx, (t_name, _) in enumerate(sorted_types):
                            if t_name == src_type:
                                src_type_idx = idx
                            if t_name == dst_type:
                                dst_type_idx = idx
                    
                    if src_type_idx is None or dst_type_idx is None:
                        continue
                    
                    # 判断哪些边属于这个关系族
                    # 对于有边的位置：使用 comp_edge_attr_discrete 判断（>= offset & < next_offset）
                    # 对于 no-edge 位置：根据 (src_type, dst_type) 判断是否属于该关系族
                    # 这样每个边位置只属于一个关系族，避免多族覆盖
                    has_edge_mask = (comp_edge_attr_discrete >= offset) & (comp_edge_attr_discrete < next_offset)  # (num_comp_edges,)
                    
                    # no-edge 位置：根据节点类型判断是否属于该关系族
                    src_nodes = comp_edge_index[0]  # (num_comp_edges,)
                    dst_nodes = comp_edge_index[1]  # (num_comp_edges,)
                    src_types = node_type_ids[src_nodes]  # (num_comp_edges,)
                    dst_types = node_type_ids[dst_nodes]  # (num_comp_edges,)
                    no_edge_mask = (comp_edge_attr_discrete == 0)  # (num_comp_edges,)
                    type_match_mask = (src_types == src_type_idx) & (dst_types == dst_type_idx)  # (num_comp_edges,)
                    no_edge_in_fam = no_edge_mask & type_match_mask  # (num_comp_edges,)
                    
                    # 合并：有边且属于该族，或 no-edge 且节点类型匹配该族
                    fam_mask = has_edge_mask | no_edge_in_fam
                    
                    if not fam_mask.any():
                        continue
                    
                    # 获取该关系族的边
                    fam_comp_edge_attr = comp_edge_attr[fam_mask]  # (num_edges_fam, de)
                    fam_comp_edge_batch = comp_edge_batch[fam_mask]  # (num_edges_fam,)
                    
                    # 将全局边属性 ID 转换为关系族内的局部 ID
                    fam_comp_edge_attr_discrete = comp_edge_attr_discrete[fam_mask]  # (num_edges_fam,)
                    fam_local_attr = fam_comp_edge_attr_discrete.clone()
                    non_zero_mask = fam_local_attr != 0
                    if non_zero_mask.any():
                        fam_local_attr[non_zero_mask] = fam_local_attr[non_zero_mask] - offset + 1
                    
                    # 转换为 one-hot 编码（使用局部 ID）
                    num_fam_states = all_family_qt[fam_name].E.shape[-1]
                    fam_local_attr_onehot = F.one_hot(fam_local_attr.long(), num_classes=num_fam_states).float()  # (num_edges_fam, num_states)
                    
                    # 获取该关系族的转移矩阵（按 batch 维保留）。
                    # 注意：compute_sparse_batched_over0_posterior_distribution 内部会用 batch 再索引一次，
                    # 这里不能预先按 fam_comp_edge_batch 取，否则会出现双重索引导致错位。
                    Qt_fam = all_family_qt[fam_name].E  # (bs, num_states, num_states)
                    Qsb_fam = all_family_qsb[fam_name].E  # (bs, num_states, num_states)
                    Qtb_fam = all_family_qtb[fam_name].E  # (bs, num_states, num_states)
                    
                    # 计算该关系族的转移概率
                    p_s_and_t_given_0_E_fam = diffusion_utils.compute_sparse_batched_over0_posterior_distribution(
                        input_data=fam_local_attr_onehot,
                        batch=fam_comp_edge_batch,
                        Qt=Qt_fam,
                        Qsb=Qsb_fam,
                        Qtb=Qtb_fam,
                    )  # (num_edges_fam, num_states, num_states)
                    
                    # 转换回全局 ID 空间（扩展维度以匹配全局边数）
                    # 需要创建一个全零的 p_s_and_t_given_0_E，然后填充对应位置
                    p_s_and_t_given_0_E_list.append((fam_mask, p_s_and_t_given_0_E_fam, offset, num_fam_states))
                
                # 合并所有关系族的转移概率
                num_comp_edges = comp_edge_attr.shape[0]
                num_global_states = self.out_dims.E
                p_s_and_t_given_0_E = torch.zeros(
                    (num_comp_edges, num_global_states, num_global_states),
                    device=self.device
                )
                
                for fam_mask, p_s_and_t_given_0_E_fam, offset, num_fam_states in p_s_and_t_given_0_E_list:
                    # 将局部状态空间映射回全局状态空间
                    # p_s_and_t_given_0_E_fam: (num_edges_fam, num_fam_states, num_fam_states)
                    # 需要映射到: (num_edges_fam, num_global_states, num_global_states)
                    # 局部状态 0 -> 全局状态 0 (no-edge)
                    # 局部状态 1, 2, ... -> 全局状态 offset, offset+1, ...
                    num_edges_fam = p_s_and_t_given_0_E_fam.shape[0]
                    
                    # 创建映射：从局部状态到全局状态
                    for local_from in range(num_fam_states):
                        if local_from == 0:
                            global_from = 0
                        else:
                            global_from = offset + local_from - 1
                        
                        if global_from < num_global_states:
                            for local_to in range(num_fam_states):
                                if local_to == 0:
                                    global_to = 0
                                else:
                                    global_to = offset + local_to - 1
                                
                                if global_to < num_global_states:
                                    # 复制转移概率
                                    p_s_and_t_given_0_E[fam_mask, global_from, global_to] = p_s_and_t_given_0_E_fam[:, local_from, local_to]
            else:
                # 同质图模式：使用全局转移矩阵
                p_s_and_t_given_0_E = (
                    diffusion_utils.compute_sparse_batched_over0_posterior_distribution(
                        input_data=comp_edge_attr,
                        batch=batch[comp_edge_index[0]],
                        Qt=Qt.E,
                        Qsb=Qsb.E,
                        Qtb=Qtb.E,
                    )
                )

            # 生成节点类型mask：限制每个节点只能在其所属类型的子类别范围内采样（禁止主类型变化，只允许子类型变化）
            node_type_mask = None
            if self.heterogeneous and hasattr(self.dataset_info, "type_offsets"):
                type_offsets = self.dataset_info.type_offsets
                if type_offsets:
                    # 获取当前节点的子类别ID
                    current_node_subtype = sparse_noisy_data["node_t"]
                    if current_node_subtype.dim() > 1:
                        current_node_subtype = current_node_subtype.argmax(dim=-1)  # (N,)
                    else:
                        current_node_subtype = current_node_subtype.long()  # (N,)
                    
                    num_nodes = current_node_subtype.shape[0]
                    num_subtypes = self.out_dims.X
                    node_type_mask = torch.zeros((num_nodes, num_subtypes), device=self.device)
                    
                    # 计算每个类型的size
                    sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                    type_sizes = {}
                    for i, (t_name, off) in enumerate(sorted_types):
                        if i + 1 < len(sorted_types):
                            type_sizes[t_name] = sorted_types[i + 1][1] - off
                        else:
                            type_sizes[t_name] = num_subtypes - off
                    
                    # 为每个节点生成mask：只允许在其当前主类型内的子类别范围内采样
                    for t_name, offset in sorted_types:
                        type_size = type_sizes.get(t_name, 0)
                        if type_size <= 0:
                            continue
                        # 找到属于该类型的节点
                        if t_name == sorted_types[-1][0]:
                            # 最后一个类型
                            type_mask = current_node_subtype >= offset
                        else:
                            next_offset = sorted_types[sorted_types.index((t_name, offset)) + 1][1]
                            type_mask = (current_node_subtype >= offset) & (current_node_subtype < next_offset)
                        
                        if type_mask.any():
                            # 只允许该类型范围内的所有子类别（禁止跨主类型）
                            node_type_mask[type_mask, offset:offset + type_size] = 1.0

            # 采样顺序：先节点后边
            # 1. 先采样节点子类别（主类型不变，只允许子类型变化）
            # 2. 再采样边类型：边类型编码节点子类别之间的关系（如"学生"和"应用文"之间的关系）
            #    模型基于新采样的节点子类别预测边类型，后验分布也会考虑节点子类别组合，两者共同决定边采样
            # 注意：edge_type_mask 只基于关系族（主类型）做粗粒度限制，细粒度的子类别-边类型关系由模型和后验学习
            (
                sampled_node,
                sampled_edge_attr,
                sampled_charge,
            ) = self.sample_sparse_node_edge(
                sparse_pred.node,
                sparse_pred.edge_attr[query_mask],
                p_s_and_t_given_0_X,
                p_s_and_t_given_0_E[query_mask],
                sparse_pred.charge,
                p_s_and_t_given_0_charge,
                edge_type_mask,  # 粗粒度限制：基于 query_edge_family（主类型级别）
                node_type_mask,  # 节点类型 mask（禁止主类型变化，只允许子类型变化）
            )
            
            # 调试信息：记录模型预测的边类型分布和mask信息（每10%进度记录一次）
            t_val = t_float[0].item()
            if hasattr(self, 'local_rank') and self.local_rank == 0 and abs(t_val * 10 - round(t_val * 10)) < 0.05:
                pred_edge_attr_discrete = sparse_pred.edge_attr[query_mask].argmax(dim=-1) if sparse_pred.edge_attr[query_mask].dim() > 1 else sparse_pred.edge_attr[query_mask]
                num_pred_edges = pred_edge_attr_discrete.shape[0]
                num_pred_no_edge = (pred_edge_attr_discrete == 0).sum().item()
                num_pred_has_edge = num_pred_edges - num_pred_no_edge
                if num_pred_edges > 0:
                    # 当前查询边在噪声图中的已有边占比
                    current_edge_attr_discrete = comp_edge_attr.argmax(dim=-1)
                    query_current = current_edge_attr_discrete[query_mask]
                    num_query_current_nonzero = (query_current != 0).sum().item()
                    pct_query_current_nonzero = (num_query_current_nonzero / num_pred_edges * 100) if num_pred_edges > 0 else 0.0

                    # 检查边类型mask
                    mask_info = ""
                    if edge_type_mask is not None:
                        num_allowed_types = edge_type_mask.sum(dim=1).float().mean().item()  # 平均每个边允许的类型数
                        num_edges_with_only_noedge = (edge_type_mask.sum(dim=1) == 1).sum().item()  # 只允许no-edge的边数
                        pct_only_noedge = (num_edges_with_only_noedge / num_pred_edges * 100) if num_pred_edges > 0 else 0.0
                        mask_info = f", mask允许类型数={num_allowed_types:.1f}, 只允许no-edge的边数={num_edges_with_only_noedge} ({pct_only_noedge:.1f}%)"
                    # 本步参与预测的「候选边位」= 有边位 + 无边位；当前有边/无边指噪声图里该位置是否已有边
                    num_query_no_edge = num_pred_edges - num_query_current_nonzero
                    print(
                        f"[DEBUG] t={t_float[0].item():.2f}: 候选边位={num_pred_edges}个(当前有边={num_query_current_nonzero}, 当前无边={num_query_no_edge}), "
                        f"预测no-edge={num_pred_no_edge} ({num_pred_no_edge/num_pred_edges*100:.1f}%), 预测有边={num_pred_has_edge} ({num_pred_has_edge/num_pred_edges*100:.1f}%)"
                        f"{mask_info}"
                    )

                    # 统计已有边被预测为no-edge的比例
                    if num_query_current_nonzero > 0:
                        pred_noedge_on_existing = (pred_edge_attr_discrete[query_current != 0] == 0).sum().item()
                        pct_noedge_on_existing = pred_noedge_on_existing / num_query_current_nonzero * 100
                        print(
                            f"[DEBUG] t={t_float[0].item():.2f}: 已有边被预测为no-edge={pred_noedge_on_existing} "
                            f"({pct_noedge_on_existing:.1f}%)"
                        )
                    
                    # 检查模型预测概率分布（应用softmax前）
                    if sparse_pred.edge_attr[query_mask].dim() > 1:
                        pred_logits = sparse_pred.edge_attr[query_mask]  # (E, num_edge_types) - 这是logits
                        pred_probs = F.softmax(pred_logits, dim=-1)  # 应用softmax
                        prob_no_edge = pred_probs[:, 0].mean().item()
                        prob_has_edge = pred_probs[:, 1:].sum(dim=1).mean().item()
                        max_prob_has_edge = pred_probs[:, 1:].max(dim=1)[0].mean().item()  # 平均最大有边概率
                        print(f"[DEBUG] t={t_float[0].item():.2f}: 平均预测概率 - no-edge={prob_no_edge:.4f}, 有边={prob_has_edge:.4f}, 最大有边概率={max_prob_has_edge:.4f}")
            # get nodes, charges adn edge index
            new_node = sampled_node
            new_charge = sampled_charge if self.use_charge else charge
            sampled_edge_index = comp_edge_index[:, query_mask]

            # update edges iteratively
            # 异质图：有向关系 (src,dst) 必须与 edge_attr 对应，undirected_to_directed 的 triu 会
            # 把 (v,u) 变成 (min,max) 导致 Paper->Author 误带 author_of，故跳过
            if not self.heterogeneous:
                sampled_edge_index, sampled_edge_attr = utils.undirected_to_directed(
                    sampled_edge_index, sampled_edge_attr
                )

            if i == len_loop - 1:
                # print('before filter', sampled_edge_index.shape)
                sampled_edge_batch = batch[sampled_edge_index[0]]
                if self.heterogeneous:
                    # Directed heterogeneous edges: sort directly by (batch, u, v)
                    # instead of condensed upper-triangle index (which assumes i < j).
                    sampled_edge_index_no_batch = sampled_edge_index - ptr[sampled_edge_batch]
                    max_nodes = int(num_nodes.max().item()) if num_nodes.numel() > 0 else 1
                    sort_key = (
                        sampled_edge_batch.to(torch.int64) * (max_nodes * max_nodes)
                        + sampled_edge_index_no_batch[0].to(torch.int64) * max_nodes
                        + sampled_edge_index_no_batch[1].to(torch.int64)
                    )
                    _, sampled_condensed_edge_index_argsort = sort_key.sort()
                else:
                    sampled_edge_index_no_batch = (
                        sampled_edge_index - ptr[sampled_edge_batch]
                    )
                    sampled_condensed_edge_index = matrix_to_condensed_index_batch(
                        matrix_index=sampled_edge_index_no_batch,
                        num_nodes=num_nodes,
                        edge_batch=sampled_edge_batch,
                    )
                    sampled_condensed_edge_index = (
                        sampled_condensed_edge_index + num_edges_ptr[sampled_edge_batch]
                    )
                    (
                        sampled_condensed_edge_index,
                        sampled_condensed_edge_index_argsort,
                    ) = sampled_condensed_edge_index.sort()
                sampled_edge_attr = sampled_edge_attr[
                    sampled_condensed_edge_index_argsort
                ]
                sampled_edge_index = sampled_edge_index[
                    :, sampled_condensed_edge_index_argsort
                ]

                # edges_to_keep_mask_sorted 的作用：
                # 在同质图模式下，边被分成多个循环处理，每个循环处理 edge_fraction 比例的边
                # 在最后一个循环中，需要确保处理的边数与其他循环一致
                # edges_to_keep_mask_sorted 用于标记当前循环应该保留的边（排序后）
                # 在异质图模式下，使用不同的采样方式（all_query_edge_index），不需要这个过滤
                if edges_to_keep_mask_sorted is not None:
                    # 同质图模式：使用 edges_to_keep_mask_sorted 过滤（确保每个循环处理的边数一致）
                    sampled_edge_attr = sampled_edge_attr[edges_to_keep_mask_sorted]
                    sampled_edge_index = sampled_edge_index[:, edges_to_keep_mask_sorted]
                # 异质图模式：edges_to_keep_mask_sorted 为 None，不需要过滤，直接使用所有采样的边
                # print('after filter', sampled_edge_index.shape)

            exist_edge_pos = sampled_edge_attr != 0
            num_sampled_edges = sampled_edge_attr.shape[0] if sampled_edge_attr.numel() > 0 else 0
            num_exist_edges = exist_edge_pos.sum().item() if exist_edge_pos.numel() > 0 else 0
            new_edge_index = torch.hstack(
                [new_edge_index, sampled_edge_index[:, exist_edge_pos]]
            )
            new_edge_attr = torch.hstack(
                [new_edge_attr, sampled_edge_attr[exist_edge_pos]]
            )
            
            # 调试信息：记录采样后的边数（每10%进度记录一次）
            t_val = t_float[0].item()
            if hasattr(self, 'local_rank') and self.local_rank == 0 and abs(t_val * 10 - round(t_val * 10)) < 0.05:
                print(f"[DEBUG] t={t_float[0].item():.2f}: 采样边数={num_sampled_edges}, 保留边数={num_exist_edges}, 当前累计边数={new_edge_index.shape[1]}")

        # 异质图：保留「未在查询集中的现有边」，避免每步丢边导致 Disconnected 100%
        # 仅保留 (src_type,dst_type) 与 edge_type 在 fam_endpoints 下合法的边，避免带过非法类型的噪声边
        # 注意：这里应该保留的是上一步的边（data.edge_index），而不是当前步骤已采样的边（new_edge_index）
        # 因为查询边集是针对当前步骤的，上一步的边如果不在查询集中，应该被保留
        if (self.heterogeneous and all_query_edge_index is not None
                and data.edge_index.shape[1] > 0):
            qu = set()
            for e in range(all_query_edge_index.shape[1]):
                u = all_query_edge_index[0, e].item()
                v = all_query_edge_index[1, e].item()
                b = all_query_edge_batch[e].item()
                # 异质图：关系是有向的，使用 (b, u, v) 而不是 (b, min(u,v), max(u,v))
                # 例如：Author->Paper 和 Paper->Author 是不同的边，不应该被合并
                qu.add((b, u, v))
            edge_batch_e = data.batch[data.edge_index[0].long()]
            edge_attr_d = (data.edge_attr.argmax(dim=-1) if data.edge_attr.dim() > 1
                           else data.edge_attr.long())
            
            # 调试信息：记录查询边集和现有边的关系（每10%进度记录一次）
            t_val = t_float[0].item()
            if hasattr(self, 'local_rank') and self.local_rank == 0 and abs(t_val * 10 - round(t_val * 10)) < 0.05:
                num_existing_edges = data.edge_index.shape[1]
                num_query_edges = len(qu)
                edges_in_query = 0
                for e in range(num_existing_edges):
                    u = data.edge_index[0, e].item()
                    v = data.edge_index[1, e].item()
                    b = edge_batch_e[e].item()
                    # 异质图：使用 (b, u, v) 而不是 (b, min(u,v), max(u,v))
                    if (b, u, v) in qu:
                        edges_in_query += 1
                pct_in_query = (edges_in_query/num_existing_edges*100) if num_existing_edges > 0 else 0.0
                print(f"[DEBUG] t={t_float[0].item():.2f}: 上一步图内边数={num_existing_edges}, 本步候选边位数={num_query_edges}, 上一步边落在候选边位中={edges_in_query}条 ({pct_in_query:.1f}%)")
            # 若存在 fam_endpoints/type_offsets，则对保留边做 (src_type,dst_type,edge_type) 合法性过滤
            fam_endpoints = getattr(self.dataset_info, "fam_endpoints", {})
            type_offsets = getattr(self.dataset_info, "type_offsets", {})
            edge_family_offsets = getattr(self.dataset_info, "edge_family_offsets", {})
            node_t = anchor_node_subtype
            type_sizes = {}
            if type_offsets:
                sorted_ty = sorted(type_offsets.items(), key=lambda x: x[1])
                for i, (t, off) in enumerate(sorted_ty):
                    type_sizes[t] = (sorted_ty[i + 1][1] - off) if i + 1 < len(sorted_ty) else (self.out_dims.X - off)
            fam_ranges = {}
            for fam_name, offset in (edge_family_offsets or {}).items():
                next_o = self.out_dims.E
                for o2 in (edge_family_offsets or {}).values():
                    if o2 > offset and o2 < next_o:
                        next_o = o2
                fam_ranges[fam_name] = (offset, min(next_o, self.out_dims.E))

            def _allowed(src_t: str, dst_t: str, typ: int) -> bool:
                if typ == 0:
                    return True
                if not fam_endpoints or not type_offsets:
                    return True
                for fam_name, (st, en) in (fam_ranges or {}).items():
                    ep = fam_endpoints.get(fam_name, {})
                    if ep.get("src_type") == src_t and ep.get("dst_type") == dst_t and st <= typ < en:
                        return True
                return False

            def _subtype_to_type(st: int):
                for t, off in sorted(type_offsets.items(), key=lambda x: x[1]):
                    if off <= st < off + type_sizes.get(t, 0):
                        return t
                return None

            # 检查 new_edge_index 中是否已经包含了上一步的边（避免重复添加）
            # 构建 new_edge_index 中已有的边集合
            new_edge_set = set()
            if new_edge_index.shape[1] > 0:
                new_edge_batch = batch[new_edge_index[0].long()]
                for e in range(new_edge_index.shape[1]):
                    u = new_edge_index[0, e].item()
                    v = new_edge_index[1, e].item()
                    b = new_edge_batch[e].item()
                    new_edge_set.add((b, u, v))
            
            keep = []
            for e in range(data.edge_index.shape[1]):
                b = edge_batch_e[e].item()
                u = data.edge_index[0, e].item()
                v = data.edge_index[1, e].item()
                typ = edge_attr_d[e].item()
                # 异质图：使用 (b, u, v) 而不是 (b, min(u,v), max(u,v))，因为关系是有向的
                # 只保留：1) 不在查询集中的边 2) 不是no-edge 3) 不在new_edge_index中（避免重复）
                if (b, u, v) not in qu and typ != 0 and (b, u, v) not in new_edge_set:
                    if fam_endpoints and type_offsets:
                        src_t = _subtype_to_type(int(node_t[u].item()))
                        dst_t = _subtype_to_type(int(node_t[v].item()))
                        if not _allowed(src_t, dst_t, typ):
                            continue
                    keep.append(e)
            if keep:
                keep = torch.tensor(keep, device=self.device, dtype=torch.long)
                new_edge_index = torch.hstack([new_edge_index, data.edge_index[:, keep]])
                new_edge_attr = torch.hstack([new_edge_attr, edge_attr_d[keep]])
                
                # 调试信息：记录保留的现有边数（每10%进度记录一次）
                t_val = t_float[0].item()
                if hasattr(self, 'local_rank') and self.local_rank == 0 and abs(t_val * 10 - round(t_val * 10)) < 0.05:
                    print(f"[DEBUG] t={t_float[0].item():.2f}: 保留现有边数={len(keep)}, 合并后总边数={new_edge_index.shape[1]}")

        # 边界检查：确保所有节点索引都在有效范围内
        if new_edge_index.shape[1] > 0:
            max_node_idx = ptr[-1].item() if len(ptr) > 0 else 0
            valid_edge_mask = (new_edge_index[0] >= 0) & (new_edge_index[0] < max_node_idx) & \
                             (new_edge_index[1] >= 0) & (new_edge_index[1] < max_node_idx)
            if not valid_edge_mask.all():
                # 过滤无效边
                new_edge_index = new_edge_index[:, valid_edge_mask]
                new_edge_attr = new_edge_attr[valid_edge_mask]
        
        # there is maximum edges of repeatation maximum for twice
        t_val = t_float[0].item()
        debug_print = hasattr(self, 'local_rank') and self.local_rank == 0 and abs(t_val * 10 - round(t_val * 10)) < 0.05
        if debug_print:
            before_dedup = new_edge_index.shape[1]
        if new_edge_index.shape[1] > 0:
            new_edge_index, new_edge_attr = utils.delete_repeated_twice_edges(
                new_edge_index, new_edge_attr
            )
            if debug_print:
                after_dedup = new_edge_index.shape[1]
                print(f"[DEBUG] t={t_val:.2f}: 去重前={before_dedup}, 去重后={after_dedup}")
        else:
            # 如果没有有效边，创建空的边索引和属性
            new_edge_index = torch.zeros((2, 0), device=self.device, dtype=torch.long)
            new_edge_attr = torch.zeros((0, self.out_dims.E), device=self.device)
        # 同质图：to_undirected 合并无向边；异质图：有向关系 (u,v) 与 (v,u) 语义不同，不复制
        if not self.heterogeneous:
            new_edge_index, new_edge_attr = utils.to_undirected(
                new_edge_index, new_edge_attr
            )

        # 异质图：最终剔除 (src_type,dst_type,edge_type) 非法边。虽然采样顺序是先节点后边，
        # 但节点类型 mask 只允许子类型变化，主类型不应变化；此处做兜底过滤以防万一（理论上不应有非法边）。
        if (self.heterogeneous and new_edge_index.shape[1] > 0):
            fam = getattr(self.dataset_info, "fam_endpoints", {})
            toff = getattr(self.dataset_info, "type_offsets", {})
            efo = getattr(self.dataset_info, "edge_family_offsets", {})
            if fam and toff and efo:
                st = new_node if new_node.dim() == 1 else new_node.argmax(dim=-1)
                tys = {}
                for i, (t, off) in enumerate(sorted(toff.items(), key=lambda x: x[1])):
                    lst = sorted(toff.items(), key=lambda x: x[1])
                    tys[t] = (lst[i+1][1] - off) if i+1 < len(lst) else (self.out_dims.X - off)
                fr = {}
                for fn, o in efo.items():
                    no = self.out_dims.E
                    for o2 in efo.values():
                        if o2 > o and o2 < no: no = o2
                    fr[fn] = (o, min(no, self.out_dims.E))
                def ok(s, d, typ):
                    if typ == 0: return True
                    for fn, (lo, hi) in fr.items():
                        ep = fam.get(fn, {})
                        if ep.get("src_type") == s and ep.get("dst_type") == d and lo <= typ < hi:
                            return True
                    return False
                def st2t(v):
                    for t, off in sorted(toff.items(), key=lambda x: x[1]):
                        if off <= v < off + tys.get(t, 0): return t
                    return None
                keep = []
                for j in range(new_edge_index.shape[1]):
                    u, v = int(new_edge_index[0, j].item()), int(new_edge_index[1, j].item())
                    typ = int(new_edge_attr[j].item())
                    src, dst = st2t(int(st[u].item())), st2t(int(st[v].item()))
                    if ok(src, dst, typ):
                        keep.append(j)
                if len(keep) < new_edge_index.shape[1]:
                    if debug_print:
                        before_filter = new_edge_index.shape[1]
                    keep = torch.tensor(keep, device=self.device, dtype=torch.long)
                    new_edge_index = new_edge_index[:, keep]
                    new_edge_attr = new_edge_attr[keep]
                    if debug_print:
                        after_filter = new_edge_index.shape[1]
                        print(f"[DEBUG] t={t_val:.2f}: 合法性过滤(因节点类型更新导致(src,dst,边类型)不匹配): 前={before_filter}, 后={after_filter}, 删除={before_filter - after_filter}")

        new_node = F.one_hot(new_node, num_classes=self.out_dims.X)
        new_charge = (
            F.one_hot(new_charge, num_classes=self.out_dims.charge)
            if self.use_charge
            else new_charge
        )
        new_edge_attr = F.one_hot(new_edge_attr.long(), num_classes=self.out_dims.E)
        
        # 调试信息：记录最终边数（每10%进度记录一次）
        t_val = t_float[0].item()
        if hasattr(self, 'local_rank') and self.local_rank == 0 and abs(t_val * 10 - round(t_val * 10)) < 0.05:
            final_edge_count = new_edge_index.shape[1]
            final_edge_attr_discrete = new_edge_attr.argmax(dim=-1) if new_edge_attr.dim() > 1 else new_edge_attr
            final_nonzero_edges = (final_edge_attr_discrete != 0).sum().item() if final_edge_attr_discrete.numel() > 0 else 0
            print(f"[DEBUG] t={t_float[0].item():.2f}: 最终边数={final_edge_count}, 非零边数={final_nonzero_edges}")

        # 检查边是否为空，如果为空则跳过断言
        if new_edge_attr.numel() > 0:
            assert torch.argmax(new_edge_attr, -1).min() > 0
            assert new_edge_attr.max() < 2

        data.node = new_node
        data.edge_index = new_edge_index
        data.edge_attr = new_edge_attr
        data.charge = new_charge

        return data

    def compute_sparse_extra_data(self, sparse_noisy_data):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""
        return utils.SparsePlaceHolder(
            node=sparse_noisy_data["X_t"],
            edge_index=sparse_noisy_data["edge_index_t"],
            edge_attr=sparse_noisy_data["edge_attr_t"],
            y=sparse_noisy_data["y_t"],
        )

    def compute_extra_data(self, sparse_noisy_data):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""
        # get extra features
        extra_data = self.extra_features(sparse_noisy_data)
        if type(extra_data) == tuple:
            extra_data = extra_data[0]
        extra_mol_data = self.domain_features(sparse_noisy_data)
        if type(extra_mol_data) == tuple:
            extra_mol_data = extra_mol_data[0]

        # get necessary parameters
        t_float = sparse_noisy_data["t_float"]
        ptr = sparse_noisy_data["ptr"]
        batch = sparse_noisy_data["batch"]
        n_node = ptr.diff().max()
        node_mask = utils.ptr_to_node_mask(ptr, batch, n_node)

        # get extra data to correct places
        edge_batch = sparse_noisy_data["batch"][
            sparse_noisy_data["comp_edge_index_t"][0].long()
        ]
        edge_batch = edge_batch.long()
        dense_comp_edge_index = (
            sparse_noisy_data["comp_edge_index_t"]
            - ptr[edge_batch]
            + edge_batch * n_node
        )
        comp_edge_index0 = dense_comp_edge_index[0] % n_node
        comp_edge_index1 = dense_comp_edge_index[1] % n_node

        extraE = extra_data.E[
            edge_batch, comp_edge_index0.long(), comp_edge_index1.long()
        ]
        # 按当前 batch 的 n_node 对齐：extra 可能用 dataset 的 max_n_nodes，导致与 node_mask (bs,n_node) 不一致
        def _align_extra_X(X, n_node, node_mask):
            if X.dim() < 2:
                return X.flatten(end_dim=1)[node_mask.flatten(end_dim=1)]
            if X.dim() == 2:
                # (N, feat) 稀疏格式：已按节点对齐，直接返回
                return X
            bs, n_max, feat = X.shape[0], X.shape[1], X.shape[2]
            if n_max >= n_node:
                slice_X = X[:, :n_node, :]
            else:
                pad = X.new_zeros(bs, n_node - n_max, feat)
                slice_X = torch.cat([X, pad], dim=1)
            return slice_X.flatten(end_dim=1)[node_mask.flatten(end_dim=1)]

        extraX = _align_extra_X(extra_data.X, n_node, node_mask)

        # 兼容 PlaceHolder (X, E, y) 和旧格式 (node, edge_attr, y)；与 node_mask 对齐
        if hasattr(extra_mol_data, 'X'):
            extra_mol_X = _align_extra_X(extra_mol_data.X, n_node, node_mask)
            extra_mol_E = extra_mol_data.E[
                edge_batch, comp_edge_index0.long(), comp_edge_index1.long()
            ]
            extra_mol_y = extra_mol_data.y
        elif hasattr(extra_mol_data, 'node'):
            extra_mol_X = _align_extra_X(extra_mol_data.node, n_node, node_mask)
            if hasattr(extra_mol_data, 'edge_attr') and extra_mol_data.edge_attr is not None:
                ea = extra_mol_data.edge_attr
                num_comp = sparse_noisy_data["comp_edge_index_t"].shape[1]
                if ea.dim() == 2:
                    if ea.shape[0] >= num_comp:
                        extra_mol_E = ea[:num_comp].to(dtype=extraE.dtype, device=extraE.device)
                    else:
                        pad = ea.new_zeros(num_comp - ea.shape[0], ea.shape[1])
                        extra_mol_E = torch.cat([ea, pad], dim=0).to(dtype=extraE.dtype, device=extraE.device)
                else:
                    extra_mol_E = ea[
                        edge_batch, comp_edge_index0.long(), comp_edge_index1.long()
                    ]
            else:
                extra_mol_E = torch.zeros_like(extraE)
            extra_mol_y = extra_mol_data.y
        else:
            # 如果没有额外特征，使用零张量，确保 dtype 和 device 一致
            extra_mol_X = torch.zeros_like(extraX)
            extra_mol_E = torch.zeros_like(extraE)
            extra_mol_y = torch.zeros_like(extra_data.y)
        
        # 确保所有张量的 dtype 和 device 一致
        extra_mol_X = extra_mol_X.to(dtype=extraX.dtype, device=extraX.device)
        extra_mol_E = extra_mol_E.to(dtype=extraE.dtype, device=extraE.device)
        extra_mol_y = extra_mol_y.to(dtype=extra_data.y.dtype, device=extra_data.y.device)

        # scale extra data when self.scaling_layer is true
        extraX, extraE, extra_y = self.scale_extra_data(
            torch.hstack([extra_mol_X, extraX]),
            torch.hstack([extraE, extra_mol_E]),
            torch.hstack([extra_data.y, extra_mol_y]),
        )

        # append extra information
        # 确保所有张量的 dtype 和形状一致
        node_t = sparse_noisy_data["node_t"].to(dtype=torch.float32)
        extraX = extraX.to(dtype=torch.float32)
        
        # 根据 use_charge 决定是否包含 charge_t
        if self.use_charge:
            charge_t = sparse_noisy_data["charge_t"]
            # 确保 charge_t 是 2D 的 (N, dc) 或 (N, 0)
            if charge_t.dim() == 1:
                charge_t = charge_t.unsqueeze(-1)  # (N,) -> (N, 1)
            charge_t = charge_t.to(dtype=torch.float32)
            node = torch.hstack([node_t, charge_t, extraX])
        else:
            # 不使用 charge，直接拼接 node_t 和 extraX
            node = torch.hstack([node_t, extraX])
        
        # 确保 edge 的 dtype 一致
        comp_edge_attr_t = sparse_noisy_data["comp_edge_attr_t"].to(dtype=torch.float32)
        extraE = extraE.to(dtype=torch.float32)
        comp_edge_attr = torch.hstack([comp_edge_attr_t, extraE])
        
        # extra_data.y contains at least the time step
        y_t = sparse_noisy_data["y_t"].to(dtype=torch.float32)
        t_float = t_float.to(dtype=torch.float32)
        extra_y = extra_y.to(dtype=torch.float32)
        y = torch.hstack((y_t, t_float, extra_y)).float()

        # get the input for the forward function
        # TODO: change to PlaceHolder
        extra_sparse_noisy_data = {
            "node_t": node,
            "edge_index_t": sparse_noisy_data["comp_edge_index_t"],
            "edge_attr_t": comp_edge_attr,
            "y_t": y,
            "batch": sparse_noisy_data["batch"],
            "charge_t": sparse_noisy_data["charge_t"],
        }

        return extra_sparse_noisy_data

    def get_scaling_layers(self):
        node_scaling_layer, edge_scaling_layer, graph_scaling_layer = None, None, None
        if self.scaling_layer:
            extra_dim = self.in_dims.X - self.out_dims.X
            if extra_dim > 0:
                node_scaling_layer = nn.Conv1d(
                    in_channels=extra_dim,
                    out_channels=extra_dim,
                    kernel_size=1,
                    dilation=1,
                    bias=False,
                    groups=extra_dim,
                )
            extra_dim = self.in_dims.E - self.out_dims.E
            if extra_dim > 0:
                edge_scaling_layer = nn.Conv1d(
                    in_channels=extra_dim,
                    out_channels=extra_dim,
                    kernel_size=1,
                    dilation=1,
                    bias=False,
                    groups=extra_dim,
                )
            extra_dim = self.in_dims.y - self.out_dims.y - 1
            if extra_dim > 0:
                graph_scaling_layer = nn.Conv1d(
                    in_channels=extra_dim,
                    out_channels=extra_dim,
                    kernel_size=1,
                    dilation=1,
                    bias=False,
                    groups=extra_dim,
                )

        return node_scaling_layer, edge_scaling_layer, graph_scaling_layer

    def scale_extra_data(self, extraX, extraE, extra_y):
        if self.node_scaling_layer is not None:
            extraX = self.node_scaling_layer(extraX.permute(1, 0)).permute(1, 0)
        if self.edge_scaling_layer is not None:
            extraE = self.edge_scaling_layer(extraE.permute(1, 0)).permute(1, 0)
        if self.graph_scaling_layer is not None:
            extra_y = self.graph_scaling_layer(extra_y.permute(1, 0)).permute(1, 0)

        return extraX, extraE, extra_y

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )
