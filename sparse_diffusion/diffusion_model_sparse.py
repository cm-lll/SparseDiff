import time
import os
import os.path as osp
import math
import pickle
import json

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

        self.train_loss = TrainLossDiscrete(cfg.model.lambda_train, self.edge_fraction)
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

        self.model = GraphTransformerConv(
            n_layers=cfg.model.n_layers,
            input_dims=self.in_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.out_dims,
            sn_hidden_dim=cfg.model.sn_hidden_dim,
            output_y=cfg.model.output_y,
            dropout=cfg.model.dropout,
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
            node_type2id = getattr(self.dataset_info, "node_type2id", {})
            type_offsets = getattr(self.dataset_info, "type_offsets", {})
            node_type_names = getattr(self.dataset_info, "node_type_names", [])
            
            # 如果 type_offsets 不存在，尝试从 vocab.json 加载
            if not type_offsets and node_type_names:
                import os.path as osp
                import json
                vocab_path = osp.join(getattr(self.dataset_info, "vocab_path", ""), "vocab.json")
                if not vocab_path or not osp.exists(vocab_path):
                    if hasattr(self.dataset_info, "datamodule") and hasattr(self.dataset_info.datamodule, "inner"):
                        vocab_path = osp.join(self.dataset_info.datamodule.inner.processed_dir, "vocab.json")
                
                if osp.exists(vocab_path):
                    with open(vocab_path, "r", encoding="utf-8") as f:
                        vocab = json.load(f)
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
                for other_fam_name, other_offset in edge_family_offsets.items():
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
                            type_sizes[t] = node_t.max().item() + 1 - off
                    
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
                        
                        # 生成所有可能的边对（排除自环）
                        if src_type == dst_type:
                            # 同类型：排除自环
                            src_indices = batch_src_nodes.unsqueeze(1).expand(-1, num_dst).flatten()
                            dst_indices = batch_dst_nodes.unsqueeze(0).expand(num_src, -1).flatten()
                            valid_mask = src_indices != dst_indices
                            if not valid_mask.any():
                                continue
                            all_fam_edges = torch.stack([src_indices[valid_mask], dst_indices[valid_mask]], dim=0)
                        else:
                            # 不同类型：生成所有可能的边对
                            src_indices = batch_src_nodes.unsqueeze(1).expand(-1, num_dst).flatten()
                            dst_indices = batch_dst_nodes.unsqueeze(0).expand(num_src, -1).flatten()
                            all_fam_edges = torch.stack([src_indices, dst_indices], dim=0)
                        
                        num_fam_possible_edges = all_fam_edges.shape[1]
                        
                        # 使用类似 sampled_condensed_indices_uniformly 的方式均匀采样
                        if num_fam_possible_edges <= num_query_edges_fam:
                            sampled_indices = torch.arange(num_fam_possible_edges, device=self.device)
                        else:
                            perm = torch.randperm(num_fam_possible_edges, device=self.device)
                            sampled_indices = perm[:num_query_edges_fam]
                        
                        # 选择采样的边
                        fam_query_edge_index = all_fam_edges[:, sampled_indices]
                        
                        if fam_query_edge_index.shape[1] > 0:
                            fam_query_edge_batch = torch.full((fam_query_edge_index.shape[1],), b, dtype=torch.long, device=self.device)
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
        )

        # pass sparse comp_graph to dense comp_graph for ease calculation
        sparse_noisy_data["comp_edge_index_t"] = comp_edge_index
        sparse_noisy_data["comp_edge_attr_t"] = comp_edge_attr
        sparse_pred = self.forward(sparse_noisy_data)

        # Compute the loss on the query edges only
        sparse_pred.edge_attr = sparse_pred.edge_attr[query_mask]
        sparse_pred.edge_index = comp_edge_index[:, query_mask]

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

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        epoch_loss = self.train_loss.log_epoch_metrics()
        # self.log("train_epoch/x_CE", epoch_loss["train_epoch/x_CE"], sync_dist=False)
        self.print(
            f"Epoch {self.current_epoch} finished: X: {epoch_loss['train_epoch/x_CE'] :.2f} -- "
            f"E: {epoch_loss['train_epoch/E_CE'] :.2f} --"
            f"charge: {epoch_loss['train_epoch/charge_CE'] :.2f} --"
            f"y: {epoch_loss['train_epoch/y_CE'] :.2f}"
        )
        epoch_node_metrics, epoch_edge_metrics = self.train_metrics.log_epoch_metrics()

        if wandb.run:
            wandb.log({"epoch": self.current_epoch}, commit=True)

    def on_validation_epoch_start(self) -> None:
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

        # prepare sparse information
        num_nodes = ptr.diff().long()
        num_edges = (num_nodes * (num_nodes - 1) / 2).long()
        num_edges_ptr = torch.hstack(
            [torch.tensor([0]).to(self.device), num_edges.cumsum(-1)]
        ).long()

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

        # make a loop for all query edges
        for i in range(len_loop):
            # the last loop might have less edges, we need to make sure that each loop has the same number of edges
            if i == len_loop - 1:
                edges_to_consider_mask = all_edge_mask >= (
                    num_edges[all_edge_batch] - num_edges_per_loop[all_edge_batch]
                )
                edges_to_keep_mask = torch.logical_and(
                    all_edge_mask >= num_edges_per_loop[all_edge_batch] * i,
                    all_edge_mask < num_edges_per_loop[all_edge_batch] * (i + 1),
                )

                triu_query_edge_index = all_condensed_index[edges_to_consider_mask]
                query_edge_batch = all_edge_batch[edges_to_consider_mask]
                condensed_query_edge_index = (
                    triu_query_edge_index + num_edges_ptr[query_edge_batch]
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

            triu_query_edge_index = all_condensed_index[edges_to_consider_mask]
            query_edge_batch = all_edge_batch[edges_to_consider_mask]
            # the order of edges does not change

            # Sample the query edges and build the computational graph = union(noisy graph, query edges)
            triu_query_edge_index = condensed_to_matrix_index_batch(
                condensed_index=triu_query_edge_index,
                num_nodes=num_nodes,
                edge_batch=query_edge_batch,
                ptr=ptr,
            ).long()
            query_mask, comp_edge_index, comp_edge_attr = get_computational_graph(
                triu_query_edge_index=triu_query_edge_index,
                clean_edge_index=sparse_noisy_data["edge_index_t"],
                clean_edge_attr=sparse_noisy_data["edge_attr_t"],
            )

            # pass sparse comp_graph to dense comp_graph for ease calculation
            sparse_noisy_data["comp_edge_index_t"] = comp_edge_index
            sparse_noisy_data["comp_edge_attr_t"] = comp_edge_attr
            sparse_pred = self.forward(sparse_noisy_data)
            all_node = sparse_pred.node
            all_charge = sparse_pred.charge
            new_edge_attr = sparse_pred.edge_attr[query_mask]
            new_edge_index = comp_edge_index[:, query_mask]

            new_edge_index, new_edge_attr = utils.undirected_to_directed(
                new_edge_index, new_edge_attr
            )

            if i == len_loop - 1:
                new_edge_batch = batch[new_edge_index[0]]
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
        metrics = [
            self.val_nll.compute(),
            self.val_X_kl.compute() * self.T,
            self.val_E_kl.compute() * self.T,
            self.val_X_logp.compute(),
            self.val_E_logp.compute(),
        ]

        if self.use_charge:
            metrics += [
                self.val_charge_kl.compute() * self.T,
                self.val_charge_logp.compute(),
            ]
        else:
            metrics += [-1, -1]

        if self.val_nll.compute() < self.best_nll:
            self.best_epoch = self.current_epoch
            self.best_nll = self.val_nll.compute()
        metrics += [self.best_epoch, self.best_nll]

        if wandb.run:
            wandb.log(
                {
                    "val/epoch_NLL": metrics[0],
                    "val/X_kl": metrics[1],
                    "val/E_kl": metrics[2],
                    "val/X_logp": metrics[3],
                    "val/E_logp": metrics[4],
                    "val/charge_kl": metrics[5],
                    "val/charge_logp": metrics[6],
                    "val/best_nll_epoch": metrics[7],
                    "val/best_nll": metrics[8],
                },
                commit=False,
            )

        self.print(
            f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
            f"Val Edge type KL: {metrics[2] :.2f}",
        )

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=False)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print(
            "Val loss: %.4f \t Best val loss:  %.4f\n" % (val_nll, self.best_val_nll)
        )

        self.val_counter += 1
        if hasattr(self, 'local_rank') and self.local_rank == 0:
            print("Starting to sample")
        if self.val_counter % self.cfg.general.sample_every_val == 0:
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
        if self.local_rank == 0:
            utils.setup_wandb(
                self.cfg
            )  # Initialize wandb only on one process to log metrics only once
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

                sampled_batch = self.sample_batch(
                    batch_id=id,
                    batch_size=to_generate,
                    num_nodes=None,
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
        probN = data.x.unsqueeze(1) @ Qtb.X[data.batch]  # (N, dx)
        node_t = probN.squeeze(1).multinomial(1).flatten()  # (N, )
        # count node numbers and edge numbers for existing edges for each graph
        num_nodes = data.ptr.diff().long()
        batch_edge = data.batch[data.edge_index[0]]
        num_edges = torch.zeros(num_nodes.shape).to(self.device)
        unique, counts = torch.unique(batch_edge, sorted=True, return_counts=True)
        num_edges[unique] = counts.float()
        # count number of non-existing edges for each graph
        num_neg_edge = ((num_nodes - 1) * num_nodes - num_edges) / 2  # (bs, )

        # Step1: diffuse on existing edges
        # get edges defined in the top triangle of the adjacency matrix
        dir_edge_index, dir_edge_attr = utils.undirected_to_directed(
            data.edge_index, data.edge_attr
        )
        
        # 检查是否为异质图模式且有 edge_family 信息
        has_edge_family = hasattr(data, 'edge_family') and data.edge_family is not None
        if self.heterogeneous and has_edge_family:
            # 异质图模式：按关系族隔离处理
            # 将 edge_family 也转换为有向边（undirected_to_directed 只返回上三角部分）
            top_mask = data.edge_index[0] < data.edge_index[1]
            dir_edge_family = data.edge_family[top_mask]
            
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
                fam_dir_edge_attr = dir_edge_attr[fam_mask]  # 应该是 (num_edges_fam,)
                fam_batch_edge = data.batch[fam_dir_edge_index[0]]
                
                # 确保 fam_dir_edge_attr 是 1D 的，且长度与边数匹配
                num_edges_fam = fam_dir_edge_index.shape[1]
                if fam_dir_edge_attr.dim() > 1:
                    # 如果是 2D 或更高维度，reshape 为 (num_edges_fam,)
                    fam_dir_edge_attr = fam_dir_edge_attr.view(num_edges_fam, -1)[:, 0]  # 取第一列
                elif len(fam_dir_edge_attr) != num_edges_fam:
                    # 如果长度不匹配，可能是索引问题，重新索引
                    fam_dir_edge_attr = dir_edge_attr[fam_mask]
                    if fam_dir_edge_attr.dim() > 1:
                        fam_dir_edge_attr = fam_dir_edge_attr.view(num_edges_fam, -1)[:, 0]
                
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
                                # 最后一个类型，需要从 node_t 的最大值推断
                                type_sizes[t] = node_t.max().item() + 1 - off
                        
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
        E_t_attr = E_t_attr[mask]
        E_t_index = E_t_index[:, mask]
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
            num_global_states = self.out_dims.E
            
            # 初始化全局状态空间的概率
            prob_true_E = torch.zeros_like(E)  # (bs, n, n, de)
            prob_pred_E = torch.zeros_like(pred_probs_E)  # (bs, n, n, de)
            
            # 为每个关系族计算后验分布
            for fam_name in all_family_qt.keys():
                offset = edge_family_offsets.get(fam_name, 0)
                
                # 找到下一个关系族的offset
                next_offset = num_global_states
                for other_fam_name, other_offset in edge_family_offsets.items():
                    if other_offset > offset and other_offset < next_offset:
                        next_offset = other_offset
                
                # 判断哪些边属于这个关系族（使用真实的 E，而不是噪声后的 E_t）
                # 因为我们要为真实的边计算后验分布，应该使用真实边所属关系族的转移矩阵
                fam_mask = (E_discrete == 0) | ((E_discrete >= offset) & (E_discrete < next_offset))  # (bs, n, n)
                
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
                    
                    pred_E_b_fam_discrete = pred_E_b_fam.argmax(dim=-1)  # (num_edges_b_fam,)
                    pred_E_b_fam_local = pred_E_b_fam_discrete.clone()
                    # 只处理属于该关系族的边
                    valid_mask_pred = (pred_E_b_fam_discrete == 0) | ((pred_E_b_fam_discrete >= offset) & (pred_E_b_fam_discrete < next_offset))
                    non_zero_valid_mask_pred = valid_mask_pred & (pred_E_b_fam_local != 0)
                    if non_zero_valid_mask_pred.any():
                        pred_E_b_fam_local[non_zero_valid_mask_pred] = pred_E_b_fam_local[non_zero_valid_mask_pred] - offset + 1
                    # 对于不在范围内的边，设置为0（无边）
                    pred_E_b_fam_local[~valid_mask_pred] = 0
                    # 确保局部状态值在有效范围内
                    pred_E_b_fam_local = torch.clamp(pred_E_b_fam_local, 0, num_fam_states - 1)
                    
                    # 转换为局部状态的one-hot编码
                    E_t_b_fam_local_onehot = F.one_hot(E_t_b_fam_local.long(), num_classes=num_fam_states).float()  # (num_edges_b_fam, num_fam_states)
                    E_b_fam_local_onehot = F.one_hot(E_b_fam_local.long(), num_classes=num_fam_states).float()  # (num_edges_b_fam, num_fam_states)
                    pred_E_b_fam_local_onehot = F.one_hot(pred_E_b_fam_local.long(), num_classes=num_fam_states).float()  # (num_edges_b_fam, num_fam_states)
                    
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
                        M=pred_E_b_fam_local_onehot.unsqueeze(0),  # (1, num_edges_b_fam, num_fam_states)
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
                    for local_state in range(num_fam_states):
                        if local_state == 0:
                            global_state = 0
                        else:
                            global_state = offset + local_state - 1
                        if global_state < num_global_states:
                            # 正确的方式：先用mask索引得到 (num_edges_b_fam, de)，然后设置对应的列
                            prob_true_E[b][batch_fam_mask][:, global_state] = prob_true_E_b_fam_local[:, local_state]
                            prob_pred_E[b][batch_fam_mask][:, global_state] = prob_pred_E_b_fam_local[:, local_state]
                    
                    # 释放中间张量以节省内存
                    del E_t_b_fam_local_onehot, E_b_fam_local_onehot, pred_E_b_fam_local_onehot
                    del prob_true_E_b_fam_local, prob_pred_E_b_fam_local
                    del E_t_b_fam_local, E_b_fam_local, pred_E_b_fam_local
            
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

        # Sample noise  -- z has size ( num_samples, num_nodes, num_features)
        sparse_sampled_data = diffusion_utils.sample_sparse_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )

        assert number_chain_steps < self.T
        chain = utils.SparseChainPlaceHolder(keep_chain=keep_chain)

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

            # keep_chain can be very small, e.g., 1
            if ((s_int * number_chain_steps) % self.T == 0) and (keep_chain != 0):
                chain.append(sparse_sampled_data)

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

    def sample_sparse_node(self, pred_node, p_s_and_t_given_0_X):
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

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        X_t = prob_X.multinomial(1)[:, 0]

        return X_t

    def sample_sparse_edge(self, pred_edge, p_s_and_t_given_0_E):
        # Normalize predictions
        pred_E = F.softmax(pred_edge, dim=-1)  # N, d0
        # Dim of the second tensor: N, d0, dt-1
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E  # N, d0, dt-1
        unnormalized_prob_E = weighted_E.sum(dim=1)  # N, dt-1
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(
            unnormalized_prob_E, dim=-1, keepdim=True
        )

        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()
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
    ):
        sampled_node = self.sample_sparse_node(pred_node, p_s_and_t_given_0_X).long()
        sampled_edge = self.sample_sparse_edge(pred_edge, p_s_and_t_given_0_E).long()

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

        beta_t = self.noise_schedule(t_normalized=t_float, skip=True)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float, skip=True)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float, skip=True)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Prior distribution
        # (N, dx, dx)
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

        # 检查是否为异质图模式，如果是，需要按关系族分别进行均匀采样
        # 根据图片要求：|Eq| = km，其中 m 是真实边数，k 是倍数（通过 edge_fraction 控制）
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
            
            # 如果 type_offsets 不存在，尝试从 vocab.json 加载
            if not type_offsets and node_type_names:
                import os.path as osp
                import json
                vocab_path = osp.join(getattr(self.dataset_info, "vocab_path", ""), "vocab.json")
                if not vocab_path or not osp.exists(vocab_path):
                    if hasattr(self.dataset_info, "datamodule") and hasattr(self.dataset_info.datamodule, "inner"):
                        vocab_path = osp.join(self.dataset_info.datamodule.inner.processed_dir, "vocab.json")
                
                if osp.exists(vocab_path):
                    with open(vocab_path, "r", encoding="utf-8") as f:
                        vocab = json.load(f)
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
            
            # 为每个关系族分别生成查询边
            # 根据图片要求：|Eq| = km，其中 m 是该关系族的真实边数，k 是倍数（通过 edge_fraction 控制）
            # 首先统计每个关系族的真实边数
            edge_attr_discrete = edge_attr.argmax(dim=-1) if edge_attr.dim() > 1 else edge_attr  # (E,)
            edge_family_offsets = self.dataset_info.edge_family_offsets
            
            all_query_edge_index_list = []
            all_query_edge_batch_list = []
            
            bs = len(num_nodes)
            node_t = node.argmax(dim=-1) if node.dim() > 1 else node  # (N,) - 全局子类别ID
            
            for fam_id, fam_name in id2edge_family.items():
                if fam_name not in fam_endpoints:
                    continue
                
                src_type = fam_endpoints[fam_name]["src_type"]
                dst_type = fam_endpoints[fam_name]["dst_type"]
                
                # 计算该关系族的offset范围
                offset = edge_family_offsets.get(fam_name, 0)
                next_offset = self.out_dims.E
                for other_fam_name, other_offset in edge_family_offsets.items():
                    if other_offset > offset and other_offset < next_offset:
                        next_offset = other_offset
                
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
                            type_sizes[t] = node_t.max().item() + 1 - off
                    
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
                        k = self.edge_fraction  # 倍数
                        num_query_edges_fam = int(math.ceil(k * m_fam)) if m_fam > 0 else 0
                        num_query_edges_fam = min(num_query_edges_fam, num_fam_possible_edges)
                        
                        if num_query_edges_fam == 0:
                            continue
                        
                        # 生成所有可能的边对（排除自环）
                        # 使用向量化操作生成所有可能的边对
                        if src_type == dst_type:
                            # 同类型：排除自环
                            src_indices = batch_src_nodes.unsqueeze(1).expand(-1, num_dst).flatten()  # (num_src * num_dst,)
                            dst_indices = batch_dst_nodes.unsqueeze(0).expand(num_src, -1).flatten()  # (num_src * num_dst,)
                            # 过滤自环
                            valid_mask = src_indices != dst_indices
                            if not valid_mask.any():
                                continue
                            all_fam_edges = torch.stack([src_indices[valid_mask], dst_indices[valid_mask]], dim=0)  # (2, num_valid_edges)
                        else:
                            # 不同类型：生成所有可能的边对（笛卡尔积）
                            src_indices = batch_src_nodes.unsqueeze(1).expand(-1, num_dst).flatten()  # (num_src * num_dst,)
                            dst_indices = batch_dst_nodes.unsqueeze(0).expand(num_src, -1).flatten()  # (num_src * num_dst,)
                            all_fam_edges = torch.stack([src_indices, dst_indices], dim=0)  # (2, num_src * num_dst)
                        
                        num_fam_possible_edges = all_fam_edges.shape[1]
                        
                        # 使用类似 sampled_condensed_indices_uniformly 的方式均匀采样
                        # 随机采样 num_query_edges_fam 条边
                        if num_fam_possible_edges <= num_query_edges_fam:
                            # 如果可能的边数 <= 要采样的边数，采样所有边
                            sampled_indices = torch.arange(num_fam_possible_edges, device=self.device)
                        else:
                            # 均匀采样 num_query_edges_fam 条边
                            perm = torch.randperm(num_fam_possible_edges, device=self.device)
                            sampled_indices = perm[:num_query_edges_fam]
                        
                        # 选择采样的边
                        fam_query_edge_index = all_fam_edges[:, sampled_indices]  # (2, num_query_edges_fam)
                        
                        if fam_query_edge_index.shape[1] > 0:
                            fam_query_edge_batch = torch.full((fam_query_edge_index.shape[1],), b, dtype=torch.long, device=self.device)
                            all_query_edge_index_list.append(fam_query_edge_index)
                            all_query_edge_batch_list.append(fam_query_edge_batch)
            
            # 合并所有关系族的查询边
            if len(all_query_edge_index_list) > 0:
                all_query_edge_index = torch.cat(all_query_edge_index_list, dim=1)  # (2, E_query)
                all_query_edge_batch = torch.cat(all_query_edge_batch_list)  # (E_query,)
                
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

            # 检查是否为异质图模式，使用不同的采样方式
            if self.heterogeneous and hasattr(self.dataset_info, "edge_family_offsets") and len(self.dataset_info.edge_family_offsets) > 0 and all_query_edge_index is not None:
                # 异质图模式：使用按关系族采样的查询边
                # 对所有关系族的查询边进行均匀采样（每个循环采样一部分）
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
                    # 选择当前循环的边
                    triu_query_edge_index = all_query_edge_index[:, start_idx:end_idx]
                    query_edge_batch = all_query_edge_batch[start_idx:end_idx]
                else:
                    # 如果已经采样完所有边，使用空边
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
            query_mask, comp_edge_index, comp_edge_attr = get_computational_graph(
                triu_query_edge_index=triu_query_edge_index,
                clean_edge_index=sparse_noisy_data["edge_index_t"],
                clean_edge_attr=sparse_noisy_data["edge_attr_t"],
            )

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
                
                for fam_id, fam_name in id2edge_family.items():
                    # 判断哪些边属于这个关系族
                    offset = edge_family_offsets.get(fam_name, 0)
                    # 获取该关系族的边数（从 offset 到下一个 offset 或全局边数）
                    if fam_name in edge_family_offsets:
                        # 计算该关系族的边 ID 范围
                        # 对于有子类别的关系族，全局 ID 范围是 [offset, offset + num_subtypes)
                        # 对于没有子类别的关系族，只有 offset（即只有 no-edge 和 edge 两种状态）
                        # 我们需要根据 edge_attr 的全局 ID 来判断
                        # 如果 edge_attr == 0，属于 no-edge
                        # 如果 edge_attr >= offset 且 < next_offset，属于该关系族
                        next_offset = None
                        for other_fam_name, other_offset in edge_family_offsets.items():
                            if other_offset > offset and (next_offset is None or other_offset < next_offset):
                                next_offset = other_offset
                        
                        if next_offset is None:
                            # 这是最后一个关系族，使用全局边数
                            fam_mask = (comp_edge_attr_discrete == 0) | ((comp_edge_attr_discrete >= offset) & (comp_edge_attr_discrete < self.out_dims.E))
                        else:
                            fam_mask = (comp_edge_attr_discrete == 0) | ((comp_edge_attr_discrete >= offset) & (comp_edge_attr_discrete < next_offset))
                    else:
                        # 如果没有找到 offset，跳过（不应该发生）
                        continue
                    
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
                    
                    # 获取该关系族的转移矩阵
                    Qt_fam = all_family_qt[fam_name].E[fam_comp_edge_batch]  # (num_edges_fam, num_states, num_states)
                    Qsb_fam = all_family_qsb[fam_name].E[fam_comp_edge_batch]  # (num_edges_fam, num_states, num_states)
                    Qtb_fam = all_family_qtb[fam_name].E[fam_comp_edge_batch]  # (num_edges_fam, num_states, num_states)
                    
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

            # sample nodes and edges
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
            )
            # get nodes, charges adn edge index
            new_node = sampled_node
            new_charge = sampled_charge if self.use_charge else charge
            sampled_edge_index = comp_edge_index[:, query_mask]

            # update edges iteratively

            sampled_edge_index, sampled_edge_attr = utils.undirected_to_directed(
                sampled_edge_index, sampled_edge_attr
            )

            if i == len_loop - 1:
                # print('before filter', sampled_edge_index.shape)
                sampled_edge_batch = batch[sampled_edge_index[0]]
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

                sampled_edge_attr = sampled_edge_attr[edges_to_keep_mask_sorted]
                sampled_edge_index = sampled_edge_index[:, edges_to_keep_mask_sorted]
                # print('after filter', sampled_edge_index.shape)

            exist_edge_pos = sampled_edge_attr != 0
            new_edge_index = torch.hstack(
                [new_edge_index, sampled_edge_index[:, exist_edge_pos]]
            )
            new_edge_attr = torch.hstack(
                [new_edge_attr, sampled_edge_attr[exist_edge_pos]]
            )

        # there is maximum edges of repeatation maximum for twice
        new_edge_index, new_edge_attr = utils.delete_repeated_twice_edges(
            new_edge_index, new_edge_attr
        )
        # concat the last new_edge_attr and new sampled edges
        new_edge_index, new_edge_attr = utils.to_undirected(
            new_edge_index, new_edge_attr
        )

        new_node = F.one_hot(new_node, num_classes=self.out_dims.X)
        new_charge = (
            F.one_hot(new_charge, num_classes=self.out_dims.charge)
            if self.use_charge
            else new_charge
        )
        new_edge_attr = F.one_hot(new_edge_attr.long(), num_classes=self.out_dims.E)

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
            egde_index=sparse_noisy_data["edge_index_t"],
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
        extraX = extra_data.X.flatten(end_dim=1)[node_mask.flatten(end_dim=1)]

        # 兼容 PlaceHolder (X, E, y) 和旧格式 (node, edge_attr, y)
        # 确保 dtype 和 device 一致
        if hasattr(extra_mol_data, 'X'):
            extra_mol_X = extra_mol_data.X.flatten(end_dim=1)[node_mask.flatten(end_dim=1)]
            extra_mol_E = extra_mol_data.E[
                edge_batch, comp_edge_index0.long(), comp_edge_index1.long()
            ]
            extra_mol_y = extra_mol_data.y
        elif hasattr(extra_mol_data, 'node'):
            extra_mol_X = extra_mol_data.node.flatten(end_dim=1)[node_mask.flatten(end_dim=1)]
            extra_mol_E = extra_mol_data.edge_attr[
                edge_batch, comp_edge_index0.long(), comp_edge_index1.long()
            ] if hasattr(extra_mol_data, 'edge_attr') else torch.zeros_like(extraE)
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
