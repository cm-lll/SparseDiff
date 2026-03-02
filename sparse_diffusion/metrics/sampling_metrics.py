from collections import Counter

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import wandb
import networkx as nx
from torchmetrics import MeanMetric, MaxMetric, Metric, MeanAbsoluteError
import torch
from torch import Tensor
from torch_geometric.utils import to_scipy_sparse_matrix
import torch_geometric as pyg

import sparse_diffusion.utils as utils
from sparse_diffusion.metrics.metrics_utils import (
    counter_to_tensor,
    wasserstein1d,
    total_variation1d,
)

class SamplingMetrics(nn.Module):
    def __init__(self, dataset_infos, test, dataloaders=None):
        super().__init__()
        self.dataset_infos = dataset_infos
        self.test = test

        self.disconnected = MeanMetric()
        self.mean_components = MeanMetric()
        self.max_components = MaxMetric()
        self.num_nodes_w1 = MeanMetric()
        self.node_types_tv = MeanMetric()
        self.edge_types_tv = MeanMetric()
        self._reference_graphs = {"val": [], "test": []}
        self._has_ref_graphs = False

        self.domain_metrics = None
        if dataset_infos.is_molecular:
            from sparse_diffusion.metrics.molecular_metrics import (
                SamplingMolecularMetrics,
            )

            self.domain_metrics = SamplingMolecularMetrics(
                dataset_infos.train_smiles,
                dataset_infos.test_smiles if test else dataset_infos.val_smiles,
                dataset_infos,
                test
            )

        elif dataset_infos.spectre:
            from sparse_diffusion.metrics.spectre_utils import (
                Comm20SamplingMetrics,
                PlanarSamplingMetrics,
                SBMSamplingMetrics,
                ProteinSamplingMetrics,
                PointCloudSamplingMetrics,
                EgoSamplingMetrics
            )

            if dataset_infos.dataset_name == "comm20":
                self.domain_metrics = Comm20SamplingMetrics(dataloaders=dataloaders, test=test)
            elif dataset_infos.dataset_name == "planar":
                self.domain_metrics = PlanarSamplingMetrics(dataloaders=dataloaders, test=test)
            elif dataset_infos.dataset_name == "sbm":
                self.domain_metrics = SBMSamplingMetrics(dataloaders=dataloaders, test=test)
            elif dataset_infos.dataset_name == "protein":
                self.domain_metrics = ProteinSamplingMetrics(dataloaders=dataloaders, test=test)
            elif dataset_infos.dataset_name == "point_cloud":
                self.domain_metrics = PointCloudSamplingMetrics(dataloaders=dataloaders, test=test)
            elif dataset_infos.dataset_name == "ego":
                self.domain_metrics = EgoSamplingMetrics(dataloaders=dataloaders, test=test)
            elif dataset_infos.dataset_name == "acm_subgraphs":
                # acm_subgraphs 使用通用指标，不设置 domain_metrics
                self.domain_metrics = None
            else:
                raise ValueError(
                    "Dataset {} not implemented".format(dataset_infos.dataset_name)
                )

        self._prepare_reference_graphs(dataloaders)

    def reset(self):
        for metric in [
            self.mean_components,
            self.max_components,
            self.disconnected,
            self.num_nodes_w1,
            self.node_types_tv,
            self.edge_types_tv,
        ]:
            metric.reset()
        if self.domain_metrics is not None:
            self.domain_metrics.reset()

    def _prepare_reference_graphs(self, dataloaders):
        """Extract val/test reference graphs once for real-vs-generated structural comparison."""
        if not isinstance(dataloaders, dict):
            return
        try:
            for split in ("val", "test"):
                loader = dataloaders.get(split, None)
                if loader is None:
                    continue
                graphs = []
                for batch in loader:
                    graphs.extend(_batch_to_nx_graphs(batch))
                self._reference_graphs[split] = graphs
            self._has_ref_graphs = (
                len(self._reference_graphs.get("val", [])) > 0
                and len(self._reference_graphs.get("test", [])) > 0
            )
        except Exception as _e:
            self._has_ref_graphs = False
            print(f"[GRAPH-CMP] failed to build reference graphs: {_e}", flush=True)

    def compute_all_metrics(self, generated_graphs: list, current_epoch, local_rank, key_suffix="", chart_title_suffix="", true_conditioned_pred_edge_counts=None):
        """Compare statistics of the generated data with statistics of the val/test set.
        - 全局统计（NumNodesW1, NodeTypesTV, EdgeTypesTV, Disconnected, MeanComponents, MaxComponents）始终在整张预测图上算。
        - 边分布图按关系族聚焦：target/generate 均为「该族内子类别」的条件分布，不含 no-edge，不被 none 边稀释。
        true_conditioned_pred_edge_counts: 可选，fam_name -> 该族真实边上的预测子类型计数；用于边分布图 generate 列（与训练指标一致）。
        Wandb Custom Chart 需绑定的 key（chart_title_suffix 为「 预测」时）：
          - 节点分布: "node distribution (子类别) 预测" 或 "node distribution 预测"
          - 边分布: "edge distribution ({fam_name}) 子类别 (仅实际边) 预测" / "(生成图无该族可能边位) 预测"
        """
        stat = (
            self.dataset_infos.statistics["test"]
            if self.test
            else self.dataset_infos.statistics["val"]
        )

        # Number of nodes
        self.num_nodes_w1(number_nodes_distance(generated_graphs, stat.num_nodes))

        # Node types（仅 rank 0 写 wandb 柱状图，避免多 GPU 重复）
        # Wandb Custom Chart 需绑定 key: "node distribution (子类别)"+suffix 或 "node distribution"+suffix
        node_type_tv, node_tv_per_class = node_types_distance(
            generated_graphs, stat.node_types, save_histogram=(local_rank == 0), dataset_infos=self.dataset_infos,
            chart_title_suffix=chart_title_suffix,
            current_epoch=current_epoch,
        )
        self.node_types_tv(node_type_tv)

        # Edge types（Wandb Custom Chart 需绑定 key: "edge distribution ({fam_name}) 子类别 (...)"+suffix）
        edge_types_tv, edge_tv_per_class = bond_types_distance(
            generated_graphs, stat.bond_types, save_histogram=(local_rank == 0), dataset_infos=self.dataset_infos,
            chart_title_suffix=chart_title_suffix,
            true_conditioned_pred_edge_counts=true_conditioned_pred_edge_counts,
            current_epoch=current_epoch,
        )
        self.edge_types_tv(edge_types_tv)

        # Components
        device = self.disconnected.device
        connected_comp = connected_components(generated_graphs).to(device)
        self.disconnected(connected_comp > 1)
        self.mean_components(connected_comp)
        self.max_components(connected_comp)

        key = "val" if not self.test else "test"
        to_log = {
            f"{key}{key_suffix}/NumNodesW1": self.num_nodes_w1.compute().item(),
            f"{key}{key_suffix}/NodeTypesTV": self.node_types_tv.compute().item(),
            f"{key}{key_suffix}/EdgeTypesTV": self.edge_types_tv.compute().item(),
            f"{key}{key_suffix}/Disconnected": self.disconnected.compute().item() * 100,
            f"{key}{key_suffix}/MeanComponents": self.mean_components.compute().item(),
            f"{key}{key_suffix}/MaxComponents": self.max_components.compute().item(),
        }
        # 真实图（reference）的连通性：若每张图本身就可能不连通，可与生成图对比
        split_name = "test" if self.test else "val"
        ref_graphs = self._reference_graphs.get(split_name, [])
        if ref_graphs and local_rank == 0:
            ref_disc, ref_mean_comp, ref_max_comp = _connected_component_stats_from_nx(ref_graphs)
            to_log[f"{key}{key_suffix}/Disconnected_real"] = float(ref_disc)
            to_log[f"{key}{key_suffix}/MeanComponents_real"] = float(ref_mean_comp)
            to_log[f"{key}{key_suffix}/MaxComponents_real"] = float(ref_max_comp)

        if self.domain_metrics is not None:
            do_metrics = self.domain_metrics.forward(
                generated_graphs, current_epoch, local_rank
            )
            to_log.update(do_metrics)

        # 真实图 vs 生成图结构对比指标（聚类系数、三角形、LCC、边重叠、幂律 alpha、同配、度分布 MMD）
        if local_rank == 0:
            split_name = "test" if self.test else "val"
            cmp_metrics = graph_comparison_metrics(
                generated_graphs=generated_graphs,
                reference_graphs=self._reference_graphs.get(split_name, []),
                key_prefix=f"{key}{key_suffix}",
            )
            to_log.update(cmp_metrics)

        if wandb.run and local_rank == 0:
            wandb.log(to_log, commit=False)
        if local_rank == 0:
            print(
                f"Sampling metrics{chart_title_suffix}", {k: round(v, 5) for k, v in to_log.items()}
            )

        return to_log, edge_tv_per_class


def number_nodes_distance(generated_graphs, dataset_counts):
    max_number_nodes = max(dataset_counts.keys())
    reference_n = torch.zeros(
        max_number_nodes + 1, device=generated_graphs.batch.device
    )
    for n, count in dataset_counts.items():
        reference_n[n] = count

    c = Counter()
    for i in range(generated_graphs.batch.max() + 1):
        c[int((generated_graphs.batch == i).sum())] += 1

    generated_n = counter_to_tensor(c).to(reference_n.device)
    return wasserstein1d(generated_n, reference_n)


def node_types_distance(generated_graphs, target, save_histogram=True, dataset_infos=None, chart_title_suffix="", current_epoch=None):
    generated_distribution = torch.zeros_like(target)

    # generated_graphs.node 可能是 one-hot 编码 (N, dx) 或离散值 (N,)
    node_types = generated_graphs.node
    if node_types.dim() > 1 and node_types.shape[-1] > 1:
        # one-hot 编码，需要 argmax
        node_types = torch.argmax(node_types, dim=-1)
    n_classes = generated_distribution.numel()
    for node in node_types:
        idx = int(node.item()) if node.numel() == 1 else int(node)
        idx = max(0, min(idx, n_classes - 1))
        generated_distribution[idx] += 1

    if save_histogram:
        if wandb.run:
            g_sum = generated_distribution.sum().item()
            t_sum = target.sum().item()
            gen_norm = (generated_distribution / g_sum).tolist() if g_sum > 0 else [0.0] * len(generated_distribution)
            tgt_norm = (target / t_sum).tolist() if t_sum > 0 else [0.0] * len(target)
            # 若有子类别名称则用「子类别」表格 + 柱状图，否则沿用原 histogram
            subtype_names = getattr(dataset_infos, "node_subtype_names", None) if dataset_infos else None
            title = "node distribution (子类别)" + chart_title_suffix
            log_kw = {"step": current_epoch} if current_epoch is not None else {}
            if subtype_names is not None and len(subtype_names) >= len(gen_norm):
                data = [[subtype_names[i], tgt_norm[i], gen_norm[i]] for i in range(len(gen_norm))]
                table = wandb.Table(data=data, columns=["子类别", "target", "generate"])
                wandb.log({title: wandb.plot.bar(table, "子类别", ["target", "generate"], title=title)}, **log_kw)
            else:
                data = [[i, tgt_norm[i], gen_norm[i]] for i in range(len(gen_norm))]
                table = wandb.Table(data=data, columns=["index", "target", "generate"])
                wandb.log({"node distribution" + chart_title_suffix: wandb.plot.histogram(table, "index", title=title)}, **log_kw)

            # 异质图：按节点类型隔离的子类别分布（每种类型一张表，target/generate 均为该类型内的条件分布）
            type_offsets = getattr(dataset_infos, "type_offsets", None) if dataset_infos else None
            node_subtype_by_type = getattr(dataset_infos, "node_subtype_by_type", None) if dataset_infos else None
            if (
                type_offsets
                and node_subtype_by_type
                and subtype_names is not None
                and getattr(dataset_infos, "output_dims", None) is not None
            ):
                sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
                type_names_ordered = [t for t, _ in sorted_types]
                n_global = generated_distribution.shape[0]
                for i, (tname, off) in enumerate(sorted_types):
                    size = (sorted_types[i + 1][1] - off) if (i + 1 < len(sorted_types)) else max(1, n_global - off)
                    if size <= 0 or tname not in node_subtype_by_type:
                        continue
                    tgt_counts = node_subtype_by_type[tname]
                    if isinstance(tgt_counts, torch.Tensor):
                        tgt_counts = tgt_counts.to(generated_distribution.device)
                    else:
                        tgt_counts = torch.as_tensor(tgt_counts, device=generated_distribution.device, dtype=generated_distribution.dtype)
                    if tgt_counts.numel() != size:
                        continue
                    # 生成图该类型节点数：全局 id 在 [off, off+size) 的节点
                    gen_counts = generated_distribution[off : off + size].clone()
                    g_sum_t = gen_counts.sum().item()
                    t_sum_t = tgt_counts.sum().item()
                    gen_norm_t = (gen_counts / g_sum_t).tolist() if g_sum_t > 0 else [0.0] * size
                    tgt_norm_t = (tgt_counts / t_sum_t).tolist() if t_sum_t > 0 else [0.0] * size
                    names_t = subtype_names[off : off + size]
                    data_t = [[names_t[j], tgt_norm_t[j], gen_norm_t[j]] for j in range(size)]
                    table_t = wandb.Table(data=data_t, columns=["子类别", "target", "generate"])
                    title_t = f"node distribution (子类别) {tname}" + chart_title_suffix
                    wandb.log({title_t: wandb.plot.bar(table_t, "子类别", ["target", "generate"], title=title_t)}, **log_kw)

        np.save("generated_node_types.npy", generated_distribution.cpu().numpy())

    return total_variation1d(generated_distribution, target)


def bond_types_distance(generated_graphs, target, save_histogram=True, dataset_infos=None, chart_title_suffix="", true_conditioned_pred_edge_counts=None, current_epoch=None):
    device = generated_graphs.batch.device
    log_kw = {"step": current_epoch} if current_epoch is not None else {}
    if true_conditioned_pred_edge_counts is None:
        true_conditioned_pred_edge_counts = {}
    generated_distribution = torch.zeros_like(target).to(device)
    edge_index, edge_attr = utils.undirected_to_directed(
        generated_graphs.edge_index, generated_graphs.edge_attr
    )
    # edge_attr 可能是 one-hot 编码 (M, de) 或离散值 (M,)
    if edge_attr.dim() > 1 and edge_attr.shape[-1] > 1:
        edge_attr = torch.argmax(edge_attr, dim=-1)

    if (
        dataset_infos is not None
        and getattr(dataset_infos, "heterogeneous", False)
        and getattr(dataset_infos, "edge_family_marginals", None)
        and getattr(dataset_infos, "edge_family_offsets", None)
        and getattr(dataset_infos, "fam_endpoints", None)
        and getattr(dataset_infos, "type_offsets", None)
    ):
        # per-family distribution
        type_offsets = dataset_infos.type_offsets
        sorted_types = sorted(type_offsets.items(), key=lambda x: x[1])
        type_names_ordered = [t for t, _ in sorted_types]
        type_sizes = {}
        for i, (t, off) in enumerate(sorted_types):
            if i + 1 < len(sorted_types):
                type_sizes[t] = sorted_types[i + 1][1] - off
            else:
                type_sizes[t] = max(1, dataset_infos.output_dims.X - off)

        node_types = generated_graphs.node
        if node_types.dim() > 1 and node_types.shape[-1] > 1:
            node_types = torch.argmax(node_types, dim=-1)
        node_type_ids = node_types.new_full(node_types.shape, -1)
        for tidx, tname in enumerate(type_names_ordered):
            off = type_offsets[tname]
            size = type_sizes.get(tname, 0)
            mask = (node_types >= off) & (node_types < off + size)
            node_type_ids[mask] = tidx

        fam_list = sorted(dataset_infos.edge_family_marginals.keys())
        fam_offsets = dataset_infos.edge_family_offsets
        fam_endpoints = dataset_infos.fam_endpoints

        # total possible edges per family
        total_possible = {f: 0.0 for f in fam_list}
        edge_counts = {f: torch.zeros(len(dataset_infos.edge_family_marginals[f]), device=device) for f in fam_list}

        num_graphs = int(generated_graphs.batch.max() + 1)
        edge_batch = generated_graphs.batch[edge_index[0]]

        for b in range(num_graphs):
            node_mask = generated_graphs.batch == b
            if not node_mask.any():
                continue
            types_b = node_type_ids[node_mask]
            for fam_name in fam_list:
                ep = fam_endpoints.get(fam_name, {})
                src_type = ep.get("src_type")
                dst_type = ep.get("dst_type")
                if src_type not in type_names_ordered or dst_type not in type_names_ordered:
                    continue
                src_idx = type_names_ordered.index(src_type)
                dst_idx = type_names_ordered.index(dst_type)
                n_src = (types_b == src_idx).sum().item()
                n_dst = (types_b == dst_idx).sum().item()
                if src_idx == dst_idx:
                    total_possible[fam_name] += max(n_src * (n_src - 1), 0)
                else:
                    total_possible[fam_name] += n_src * n_dst

        # count existing edges per family and per subtype
        for fam_name in fam_list:
            offset = fam_offsets.get(fam_name, 0)
            next_offset = len(target)
            for _, o in fam_offsets.items():
                if o > offset and o < next_offset:
                    next_offset = o
            fam_mask = (edge_attr >= offset) & (edge_attr < next_offset)
            fam_edges = edge_attr[fam_mask]
            if fam_edges.numel() > 0:
                local_ids = (fam_edges - offset + 1).clamp(min=1)
                one_hot = F.one_hot(
                    local_ids, num_classes=edge_counts[fam_name].numel()
                ).float()
                edge_counts[fam_name] += one_hot.sum(dim=0)

        tvs = []
        subtype_names_by_fam = getattr(dataset_infos, "edge_subtype_names_by_family", None)
        for fam_name in fam_list:
            total = total_possible[fam_name]
            fam_counts = edge_counts[fam_name]
            if total <= 0:
                # 该关系族在生成图中「可能边位」为 0（例如没有 Author/Org 节点，则 author_of/affiliated_with 为 0）
                # 仅当该族有真实子类别（非仅 __none__）时才记录到 wandb；无子类别的不显示
                if save_histogram and wandb.run and subtype_names_by_fam and fam_name in subtype_names_by_fam:
                    names = subtype_names_by_fam[fam_name]
                    n = min(len(names), len(dataset_infos.edge_family_marginals[fam_name]))
                    if n > 1:
                        names_edge_only = [names[j] for j in range(1, n)]
                        if not all(("__none__" in str(s)) for s in names_edge_only):
                            target_fam = dataset_infos.edge_family_marginals[fam_name].to(device)
                            tgt_full = (target_fam / target_fam.sum()).tolist() if target_fam.sum() > 0 else [0.0] * target_fam.numel()
                            sum_tgt_edge = sum(tgt_full[1:n]) if n <= len(tgt_full) else 0.0
                            tgt_cond = [tgt_full[j] / sum_tgt_edge if sum_tgt_edge > 0 else 0.0 for j in range(1, n)] if n <= len(tgt_full) else [0.0] * (n - 1)
                            data = [[names[j], tgt_cond[j - 1], 0.0] for j in range(1, n)]
                            if data:
                                table = wandb.Table(data=data, columns=["子类别", "target", "generate"])
                                title_fam0 = f"edge distribution ({fam_name}) 子类别 (生成图无该族可能边位)" + chart_title_suffix
                                wandb.log({title_fam0: wandb.plot.bar(table, "子类别", ["target", "generate"], title=title_fam0)}, **log_kw)
                continue
            fam_counts[0] = max(total - fam_counts[1:].sum().item(), 0.0)
            target_fam = dataset_infos.edge_family_marginals[fam_name].to(device)
            tv, _ = total_variation1d(fam_counts, target_fam)
            # total_variation1d 返回的第一个值是 float (.item())，需要转换为 tensor
            tvs.append(torch.tensor(tv, device=device))
            # wandb：按关系族记录边子类别分布；聚焦到该族，仅显示子类别（不含 no-edge），不被 none 边稀释
            if save_histogram and wandb.run and subtype_names_by_fam and fam_name in subtype_names_by_fam:
                names = subtype_names_by_fam[fam_name]
                n = min(len(names), fam_counts.numel())
                if n > 1:
                    names_edge_only = [names[i] for i in range(1, n)]
                    if not all(("__none__" in str(s)) for s in names_edge_only):
                        # target / generate 均为「该族内」条件分布（只含子类别 1..n-1，不含 no-edge）
                        tgt_full = (target_fam / target_fam.sum()).tolist() if target_fam.sum() > 0 else [0.0] * fam_counts.numel()
                        tgt_full = (tgt_full + [0.0] * n)[:n]
                        sum_tgt_edge = sum(tgt_full[1:n])
                        tgt_cond = [tgt_full[i] / sum_tgt_edge if sum_tgt_edge > 0 else 0.0 for i in range(1, n)]
                        # generate：有 true_conditioned 时用「真实边属于该族时模型预测的子类型」分布；否则用整张预测图里该族的边统计
                        gen_counts_tensor = true_conditioned_pred_edge_counts.get(fam_name)
                        if gen_counts_tensor is not None:
                            gen_counts_tensor = gen_counts_tensor.to(device)
                            num_gen = min(gen_counts_tensor.numel(), n)
                            sum_gen_edge = gen_counts_tensor[1:num_gen].sum().item()
                            gen_cond = [gen_counts_tensor[i].item() / sum_gen_edge if sum_gen_edge > 0 else 0.0 for i in range(1, num_gen)]
                            gen_cond = (gen_cond + [0.0] * (n - 1))[: n - 1]
                        else:
                            gen_full = (fam_counts / fam_counts.sum()).tolist() if fam_counts.sum() > 0 else [0.0] * fam_counts.numel()
                            gen_full = (gen_full + [0.0] * n)[:n]
                            sum_gen_edge = sum(gen_full[1:n])
                            gen_cond = [gen_full[i] / sum_gen_edge if sum_gen_edge > 0 else 0.0 for i in range(1, n)]
                        names_edge = (names[1:n] + [str(i) for i in range(1, n)])[: n - 1]
                        data = [[names_edge[i], tgt_cond[i], gen_cond[i]] for i in range(len(names_edge))]
                        table = wandb.Table(data=data, columns=["子类别", "target", "generate"])
                        title_fam = f"edge distribution ({fam_name}) 子类别 (仅实际边)" + chart_title_suffix
                        wandb.log({title_fam: wandb.plot.bar(table, "子类别", ["target", "generate"], title=title_fam)}, **log_kw)
        if tvs:
            edge_types_tv = torch.stack(tvs).mean()
        else:
            edge_types_tv = torch.tensor(0.0, device=device)
        return edge_types_tv, None

    # fallback: global distribution
    for edge in edge_attr:
        generated_distribution[edge] += 1

    n_nodes = pyg.nn.pool.global_add_pool(
        torch.ones_like(generated_graphs.batch).unsqueeze(-1), generated_graphs.batch
    ).flatten()
    generated_distribution[0] = (n_nodes * (n_nodes - 1) / 2).sum()
    generated_distribution[0] = (
        generated_distribution[0] - generated_distribution[1:].sum()
    )

    if save_histogram:
        if wandb.run:
            data = [[k, l] for k, l in zip(target, generated_distribution/generated_distribution.sum())]
            table = wandb.Table(data=data, columns=["target", "generate"])
            wandb.log({'edge distribution': wandb.plot.histogram(table, 'types', title="edge distribution")})

        np.save("generated_bond_types.npy", generated_distribution.cpu().numpy())

    tv, tv_per_class = total_variation1d(generated_distribution, target.to(device))
    return tv, tv_per_class


def _build_ptr(batch: torch.Tensor) -> torch.Tensor:
    if batch.numel() == 0:
        return torch.tensor([0], device=batch.device, dtype=torch.long)
    num_graphs = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=num_graphs).long()
    return torch.cat(
        [torch.tensor([0], device=batch.device, dtype=torch.long), counts.cumsum(0)]
    )


def _batch_to_nx_graphs(data) -> list:
    """Convert a PyG batch / sparse placeholder to a list of undirected NetworkX graphs."""
    if not hasattr(data, "edge_index"):
        return []
    if hasattr(data, "batch") and data.batch is not None:
        batch = data.batch
    else:
        n_nodes = int(data.node.shape[0] if hasattr(data, "node") else data.x.shape[0])
        batch = torch.zeros(n_nodes, dtype=torch.long, device=data.edge_index.device)

    ptr = getattr(data, "ptr", None)
    if ptr is None:
        ptr = _build_ptr(batch)
    edge_batch = batch[data.edge_index[0]]
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    graphs = []
    for i in range(num_graphs):
        node_mask = batch == i
        n = int(node_mask.sum().item())
        edge_mask = edge_batch == i
        if edge_mask.any():
            edge_index = (data.edge_index[:, edge_mask] - ptr[i]).detach().cpu().numpy()
            undirected_edges = set()
            for u, v in edge_index.T.tolist():
                if u == v:
                    continue
                a, b = (u, v) if u < v else (v, u)
                undirected_edges.add((a, b))
        else:
            undirected_edges = set()
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(undirected_edges)
        graphs.append(g)
    return graphs


def _safe_mean(values):
    arr = np.array(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def _power_law_alpha_from_degrees(degrees):
    degrees = np.asarray(degrees, dtype=float)
    degrees = degrees[degrees > 0]
    if degrees.size < 2:
        return np.nan
    try:
        import powerlaw  # optional

        # 度数为整数，使用 discrete=True 避免 powerlaw 的 UserWarning
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
        return float(fit.power_law.alpha)
    except Exception:
        xmin = max(1.0, float(degrees.min()))
        normed = degrees / xmin
        normed = normed[normed > 1.0]
        if normed.size < 2:
            return np.nan
        denom = np.log(normed).sum()
        if denom <= 0:
            return np.nan
        return float(1.0 + normed.size / denom)


def _connected_component_stats_from_nx(graphs):
    """Given a list of NetworkX graphs, return (disconnected_pct, mean_components, max_components)."""
    if not graphs:
        return np.nan, np.nan, np.nan
    num_components_list = []
    for g in graphs:
        if g.number_of_nodes() == 0:
            num_components_list.append(0)
        else:
            num_components_list.append(nx.number_connected_components(g))
    arr = np.array(num_components_list, dtype=float)
    disconnected_pct = (arr > 1).mean() * 100.0
    mean_comp = float(np.mean(arr))
    max_comp = float(np.max(arr))
    return disconnected_pct, mean_comp, max_comp


def _set_structural_stats(graphs):
    clustering = []
    triangles = []
    lcc_sizes = []
    assortativity = []
    alpha = []
    all_degrees = []
    for g in graphs:
        if g.number_of_nodes() == 0:
            continue
        clustering.append(nx.average_clustering(g))
        tri = sum(nx.triangles(g).values()) / 3.0
        triangles.append(float(tri))
        if g.number_of_nodes() > 0:
            lcc_sizes.append(float(max(len(c) for c in nx.connected_components(g))))
        else:
            lcc_sizes.append(0.0)
        try:
            assort = nx.degree_assortativity_coefficient(g)
        except Exception:
            assort = np.nan
        assortativity.append(float(assort) if assort is not None else np.nan)
        deg = np.array([d for _, d in g.degree()], dtype=float)
        all_degrees.extend(deg.tolist())
        alpha.append(_power_law_alpha_from_degrees(deg))

    return {
        "clustering": _safe_mean(clustering),
        "triangles": _safe_mean(triangles),
        "lcc_size": _safe_mean(lcc_sizes),
        "power_law_alpha": _safe_mean(alpha),
        "assortativity": _safe_mean(assortativity),
        "degrees": np.asarray(all_degrees, dtype=float),
    }


def _paired_edge_overlap_rate(reference_graphs, generated_graphs):
    if len(reference_graphs) == 0 or len(generated_graphs) == 0:
        return np.nan
    n = min(len(reference_graphs), len(generated_graphs))
    overlaps = []
    for i in range(n):
        real_edges = set(reference_graphs[i].edges())
        gen_edges = set(generated_graphs[i].edges())
        denom = max(len(real_edges), 1)
        overlaps.append(len(real_edges.intersection(gen_edges)) / denom)
    return _safe_mean(overlaps)


def _degree_mmd(real_deg, gen_deg):
    real_deg = np.asarray(real_deg, dtype=float).reshape(-1, 1)
    gen_deg = np.asarray(gen_deg, dtype=float).reshape(-1, 1)
    if real_deg.size == 0 or gen_deg.size == 0:
        return np.nan

    z = np.vstack([real_deg, gen_deg])
    dmat = np.abs(z - z.T)
    sigma = np.median(dmat[dmat > 0]) if np.any(dmat > 0) else 1.0
    sigma = float(max(sigma, 1e-6))

    def rbf(a, b):
        diff = a - b.T
        return np.exp(-(diff * diff) / (2 * sigma * sigma))

    k_xx = rbf(real_deg, real_deg)
    k_yy = rbf(gen_deg, gen_deg)
    k_xy = rbf(real_deg, gen_deg)

    m = real_deg.shape[0]
    n = gen_deg.shape[0]
    if m > 1:
        xx = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
    else:
        xx = 0.0
    if n > 1:
        yy = (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
    else:
        yy = 0.0
    xy = k_xy.mean()
    return float(max(xx + yy - 2.0 * xy, 0.0))


def _safe_log_value(v):
    try:
        f = float(v)
    except Exception:
        return -1.0
    if np.isnan(f) or np.isinf(f):
        return -1.0
    return f


def graph_comparison_metrics(generated_graphs, reference_graphs, key_prefix):
    """Compute real-vs-generated graph structural comparison metrics."""
    if len(reference_graphs) == 0:
        return {}
    gen_graphs = _batch_to_nx_graphs(generated_graphs)
    if len(gen_graphs) == 0:
        return {}

    n = min(len(reference_graphs), len(gen_graphs))
    # 配对连通性：按图对比「是否不连通」，比整图聚合的 Disconnected 更有意义
    real_n_comp = [
        nx.number_connected_components(g) if g.number_of_nodes() > 0 else 0
        for g in reference_graphs[:n]
    ]
    gen_n_comp = [
        nx.number_connected_components(g) if g.number_of_nodes() > 0 else 0
        for g in gen_graphs[:n]
    ]
    real_disconn = np.array([c > 1 for c in real_n_comp], dtype=float)
    gen_disconn = np.array([c > 1 for c in gen_n_comp], dtype=float)
    connectivity_agreement = float((real_disconn == gen_disconn).mean() * 100.0) if n > 0 else np.nan

    real_stats = _set_structural_stats(reference_graphs)
    gen_stats = _set_structural_stats(gen_graphs)
    edge_overlap = _paired_edge_overlap_rate(reference_graphs, gen_graphs)
    degree_mmd = _degree_mmd(real_stats["degrees"], gen_stats["degrees"])

    metrics = {
        f"{key_prefix}/Graph/ConnectivityAgreement": _safe_log_value(connectivity_agreement),
        f"{key_prefix}/Graph/Clustering_real": _safe_log_value(real_stats["clustering"]),
        f"{key_prefix}/Graph/Clustering_gen": _safe_log_value(gen_stats["clustering"]),
        f"{key_prefix}/Graph/Triangles_real": _safe_log_value(real_stats["triangles"]),
        f"{key_prefix}/Graph/Triangles_gen": _safe_log_value(gen_stats["triangles"]),
        f"{key_prefix}/Graph/LCCSize_real": _safe_log_value(real_stats["lcc_size"]),
        f"{key_prefix}/Graph/LCCSize_gen": _safe_log_value(gen_stats["lcc_size"]),
        f"{key_prefix}/Graph/EdgeOverlapRate": _safe_log_value(edge_overlap),
        f"{key_prefix}/Graph/PowerLawAlpha_real": _safe_log_value(real_stats["power_law_alpha"]),
        f"{key_prefix}/Graph/PowerLawAlpha_gen": _safe_log_value(gen_stats["power_law_alpha"]),
        f"{key_prefix}/Graph/DegreeAssortativity_real": _safe_log_value(real_stats["assortativity"]),
        f"{key_prefix}/Graph/DegreeAssortativity_gen": _safe_log_value(gen_stats["assortativity"]),
        f"{key_prefix}/Graph/DegreeMMD": _safe_log_value(degree_mmd),
    }
    metrics[f"{key_prefix}/Graph/Clustering_abs_gap"] = abs(
        metrics[f"{key_prefix}/Graph/Clustering_gen"] - metrics[f"{key_prefix}/Graph/Clustering_real"]
    )
    metrics[f"{key_prefix}/Graph/Triangles_abs_gap"] = abs(
        metrics[f"{key_prefix}/Graph/Triangles_gen"] - metrics[f"{key_prefix}/Graph/Triangles_real"]
    )
    metrics[f"{key_prefix}/Graph/LCCSize_abs_gap"] = abs(
        metrics[f"{key_prefix}/Graph/LCCSize_gen"] - metrics[f"{key_prefix}/Graph/LCCSize_real"]
    )
    metrics[f"{key_prefix}/Graph/PowerLawAlpha_abs_gap"] = abs(
        metrics[f"{key_prefix}/Graph/PowerLawAlpha_gen"] - metrics[f"{key_prefix}/Graph/PowerLawAlpha_real"]
    )
    metrics[f"{key_prefix}/Graph/DegreeAssortativity_abs_gap"] = abs(
        metrics[f"{key_prefix}/Graph/DegreeAssortativity_gen"] - metrics[f"{key_prefix}/Graph/DegreeAssortativity_real"]
    )
    return metrics


def connected_components(generated_graphs):
    num_graphs = int(generated_graphs.batch.max() + 1)
    all_num_components = torch.zeros(num_graphs)
    batch = generated_graphs.batch
    edge_batch = batch[generated_graphs.edge_index[0]]
    ptr = getattr(generated_graphs, "ptr", None)
    if ptr is None:
        ptr = torch.cat([
            torch.tensor([0], device=batch.device, dtype=torch.long),
            torch.unique(batch, return_counts=True)[1].cumsum(0, dtype=torch.long)
        ])
    for i in range(num_graphs):
        # get the graph
        node_mask = batch == i
        edge_mask = edge_batch == i
        node_types = generated_graphs.node[node_mask]
        # 如果是 one-hot 编码，需要 argmax（但这里只需要节点数量，不需要实际类型）
        if node_types.dim() > 1 and node_types.shape[-1] > 1:
            node = torch.argmax(node_types, dim=-1)
        else:
            node = node_types
        edge_index = generated_graphs.edge_index[:, edge_mask] - ptr[i]
        # DENSE OPERATIONS
        sp_adj = to_scipy_sparse_matrix(edge_index, num_nodes=len(node))
        num_components, component = sp.csgraph.connected_components(sp_adj.toarray())
        all_num_components[i] = num_components

    return all_num_components


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        target = self.target_histogram.to(pred.device)
        super().update(pred, target)


class CEPerClass(Metric):
    full_state_update = True

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id
        self.add_state("total_ce", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.binary_cross_entropy = torch.nn.BCELoss(reduction="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.0).any(dim=-1)

        prob = self.softmax(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        output = self.binary_cross_entropy(prob, target)
        self.total_ce += output
        self.total_samples += prob.numel()

    def compute(self):
        return self.total_ce / self.total_samples


class MeanNumberEdge(Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("total_edge", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples
