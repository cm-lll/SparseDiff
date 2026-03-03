import torch
import torch.nn as nn
import wandb
from sparse_diffusion.metrics.abstract_metrics import CrossEntropyMetric, MaskedAccuracy


class TrainLossDiscrete(nn.Module):
    """Train with Cross entropy"""

    def __init__(
        self,
        lambda_train,
        edge_fraction,
        dataset_info=None,
        relation_matrix_loss_weight=0.0,
        relation_matrix_loss_normalize=True,
        use_edge_subtype_ce=True,
        metapath2_loss_weight=0.0,
        metapath2_loss_normalize=True,
        structure_only=False,
        subtype_degree_loss_weight=0.0,
        subtype_degree_loss_normalize=True,
        subtype_degree_max=10,
    ):
        super().__init__()
        self.structure_only = bool(structure_only)
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.edge_pos_acc = MaskedAccuracy(ignore_index=0)
        self.y_loss = CrossEntropyMetric()
        self.charge_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train
        self.lambda_train[0] = self.lambda_train[0] / edge_fraction
        self.pos_weight_per_edge_type = None
        self.edge_family_ranges = {}
        self.edge_family_multisub = set()
        self.edge_focal_gamma = 2.0
        self.edge_focal_gamma_pos = 2.0
        self.edge_focal_gamma_neg = 2.0
        self.edge_alpha_neg = 0.8
        self.pos_weight_scale = 1.0
        self.focal_gamma_scale = 1.0
        self.focal_alpha_neg_scale = 1.0
        # 基于性能的自适应权重分配（不使用MoE风格）
        self.base_alpha_neg = 0.8  # 基础alpha_neg值
        self.adaptive_alpha_neg = 0.8  # 当前alpha_neg（会自适应调整）
        self.adaptive_smoothing = 0.1  # 自适应调整的平滑系数
        # 历史指标（用于自适应调整）
        self.prev_pos_ce = None
        self.prev_neg_ce = None
        self.prev_pos_exist_acc = None
        self.edge_exist_weight = 1.0
        self.edge_subtype_weight = 1.0
        self.edge_exist_correct = None
        self.edge_exist_total = None
        self.edge_pos_exist_correct = None
        self.edge_pos_exist_total = None
        self.edge_pos_class_correct = None
        self.edge_pos_class_total = None
        self.edge_pos_class_given_exist_correct = None
        self.edge_pos_class_given_exist_total = None
        # 细粒度损失指标
        self.edge_exist_loss_total = None
        self.edge_subtype_loss_total = None
        self.edge_pos_ce_total = None
        self.edge_pos_ce_samples = None
        self.edge_neg_ce_total = None
        self.edge_neg_ce_samples = None
        self.relation_matrix_loss_weight = float(relation_matrix_loss_weight)
        self.relation_matrix_loss_normalize = bool(relation_matrix_loss_normalize)
        self.relation_matrix_loss_total = None
        self.relation_matrix_loss_steps = None
        self.use_edge_subtype_ce = bool(use_edge_subtype_ce)
        self.metapath2_loss_weight = float(metapath2_loss_weight)
        self.metapath2_loss_normalize = bool(metapath2_loss_normalize)
        self.metapath2_loss_total = None
        self.metapath2_loss_steps = None
        self.subtype_degree_loss_weight = float(subtype_degree_loss_weight)
        self.subtype_degree_loss_normalize = bool(subtype_degree_loss_normalize)
        self.subtype_degree_max = int(subtype_degree_max)
        self.subtype_degree_loss_total = None
        self.subtype_degree_loss_steps = None

        if (
            dataset_info is not None
            and getattr(dataset_info, "edge_family_marginals", None)
            and getattr(dataset_info, "edge_family_offsets", None)
        ):
            edge_family_marginals = dataset_info.edge_family_marginals
            edge_family_offsets = dataset_info.edge_family_offsets
            num_edge_types = getattr(dataset_info, "output_dims", None)
            if num_edge_types is not None:
                num_edge_types = num_edge_types.E
            if not num_edge_types:
                num_edge_types = len(getattr(dataset_info, "bond_types", []))
            if num_edge_types:
                pos_weight = torch.ones(num_edge_types, dtype=torch.float)
                for fam_name, marginals in edge_family_marginals.items():
                    if not isinstance(marginals, torch.Tensor):
                        marginals = torch.tensor(marginals, dtype=torch.float)
                    u0 = float(marginals[0].item()) if marginals.numel() > 0 else 0.0
                    u1 = max(1.0 - u0, 1e-6)
                    w = (1.0 - u1) / u1
                    w = float(min(max(w, 1.0), 100.0))
                    offset = edge_family_offsets.get(fam_name, 0)
                    next_offset = num_edge_types
                    for _, o in edge_family_offsets.items():
                        if o > offset and o < next_offset:
                            next_offset = o
                    for gid in range(offset, next_offset):
                        if 0 <= gid < num_edge_types:
                            pos_weight[gid] = w
                pos_weight[0] = 1.0
                # Build family ranges for per-family metrics
                edge_family_offsets = dataset_info.edge_family_offsets
                fam_sorted = sorted(edge_family_offsets.items(), key=lambda x: x[1])
                for fam_name, offset in fam_sorted:
                    next_offset = num_edge_types
                    for _, off2 in fam_sorted:
                        if off2 > offset and off2 < next_offset:
                            next_offset = off2
                    self.edge_family_ranges[fam_name] = (offset, next_offset)
                    if (next_offset - offset) > 1:
                        self.edge_family_multisub.add(fam_name)

    def _relation_matrix_loss(self, pred, true_data):
        """Subtype-level outgoing relation matrix loss on query edges."""
        if pred.edge_attr.numel() == 0 or true_data.edge_attr.numel() == 0:
            return pred.edge_attr.sum() * 0.0
        if true_data.node.numel() == 0:
            return pred.edge_attr.sum() * 0.0

        num_node_subtypes = pred.node.shape[-1]
        num_edge_types = pred.edge_attr.shape[-1]
        src = pred.edge_index[0].long()
        dst = pred.edge_index[1].long()

        src_sub = true_data.node[src].long()
        dst_sub = true_data.node[dst].long()

        # Build target matrix from true edge labels: M_true[src_sub, dst_sub, edge_type].
        true_labels = true_data.edge_attr.long()
        true_flat_idx = (
            (src_sub * num_node_subtypes + dst_sub) * num_edge_types + true_labels
        )
        true_hist = torch.zeros(
            num_node_subtypes * num_node_subtypes * num_edge_types,
            device=pred.edge_attr.device,
            dtype=pred.edge_attr.dtype,
        )
        true_hist.scatter_add_(
            0, true_flat_idx, torch.ones_like(true_flat_idx, dtype=pred.edge_attr.dtype)
        )

        # Build predicted matrix from edge-type probabilities.
        pred_prob = torch.softmax(pred.edge_attr, dim=-1)
        base_pair = (src_sub * num_node_subtypes + dst_sub) * num_edge_types
        type_offsets = torch.arange(num_edge_types, device=pred.edge_attr.device).view(1, -1)
        pred_flat_idx = (base_pair.view(-1, 1) + type_offsets).reshape(-1)
        pred_hist = torch.zeros_like(true_hist)
        pred_hist.scatter_add_(0, pred_flat_idx, pred_prob.reshape(-1))

        if self.relation_matrix_loss_normalize:
            true_hist = true_hist / true_hist.sum().clamp(min=1.0)
            pred_hist = pred_hist / pred_hist.sum().clamp(min=1.0)

        return torch.nn.functional.l1_loss(pred_hist, true_hist, reduction="mean")

    def _metapath2_subtype_loss(self, pred, true_data):
        """Two-hop subtype transition loss using subtype adjacency composition (A @ A)."""
        if pred.edge_attr.numel() == 0 or true_data.edge_attr.numel() == 0:
            return pred.edge_attr.sum() * 0.0
        if true_data.node.numel() == 0:
            return pred.edge_attr.sum() * 0.0

        src = pred.edge_index[0].long()
        dst = pred.edge_index[1].long()
        if src.numel() == 0:
            return pred.edge_attr.sum() * 0.0

        node_sub = true_data.node.long()
        num_sub = pred.node.shape[-1]
        device = pred.edge_attr.device
        dtype = pred.edge_attr.dtype

        # Predicted edge existence probability P(edge exists) = 1 - P(no-edge).
        pred_prob = torch.softmax(pred.edge_attr, dim=-1)
        pred_exist = 1.0 - pred_prob[:, 0]

        # True edge existence indicator.
        true_labels = true_data.edge_attr.long()
        true_exist = (true_labels > 0).to(dtype=dtype)
        src_sub = node_sub[src]
        dst_sub = node_sub[dst]

        # Build subtype-level adjacency matrices A_true / A_pred with O(E) scatter.
        flat_idx = src_sub * num_sub + dst_sub
        A_true = torch.zeros(num_sub * num_sub, device=device, dtype=dtype)
        A_pred = torch.zeros(num_sub * num_sub, device=device, dtype=dtype)
        A_true.scatter_add_(0, flat_idx, true_exist)
        A_pred.scatter_add_(0, flat_idx, pred_exist)
        A_true = A_true.view(num_sub, num_sub)
        A_pred = A_pred.view(num_sub, num_sub)

        if self.metapath2_loss_normalize:
            A_true = A_true / A_true.sum().clamp(min=1.0)
            A_pred = A_pred / A_pred.sum().clamp(min=1.0)

        # Two-hop subtype transitions.
        B_true = A_true @ A_true
        B_pred = A_pred @ A_pred

        if self.metapath2_loss_normalize:
            B_true = B_true / B_true.sum().clamp(min=1.0)
            B_pred = B_pred / B_pred.sum().clamp(min=1.0)

        return torch.nn.functional.l1_loss(B_pred, B_true, reduction="mean")

    def _subtype_degree_hist_loss(self, pred, true_data):
        """Subtype-level degree distribution loss (histogram over total degree)."""
        if pred.edge_attr.numel() == 0 or true_data.edge_attr.numel() == 0:
            return pred.edge_attr.sum() * 0.0
        if true_data.node.numel() == 0:
            return pred.edge_attr.sum() * 0.0

        src = pred.edge_index[0].long()
        dst = pred.edge_index[1].long()
        if src.numel() == 0:
            return pred.edge_attr.sum() * 0.0

        node_sub = true_data.node.long()
        num_sub = pred.node.shape[-1]
        num_nodes = node_sub.shape[0]
        device = pred.edge_attr.device
        dtype = pred.edge_attr.dtype
        num_bins = self.subtype_degree_max + 2  # 0..max_degree plus overflow

        pred_prob = torch.softmax(pred.edge_attr, dim=-1)
        pred_exist = 1.0 - pred_prob[:, 0]
        true_labels = true_data.edge_attr.long()
        true_exist = (true_labels > 0).to(dtype=dtype)

        true_deg = torch.zeros(num_nodes, device=device, dtype=dtype)
        pred_deg = torch.zeros(num_nodes, device=device, dtype=dtype)
        true_deg.scatter_add_(0, src, true_exist)
        true_deg.scatter_add_(0, dst, true_exist)
        pred_deg.scatter_add_(0, src, pred_exist)
        pred_deg.scatter_add_(0, dst, pred_exist)

        true_bin = true_deg.round().long().clamp(min=0, max=num_bins - 1)
        true_hist = torch.zeros(num_sub * num_bins, device=device, dtype=dtype)
        true_flat_idx = node_sub * num_bins + true_bin
        true_hist.scatter_add_(
            0, true_flat_idx, torch.ones_like(true_flat_idx, dtype=dtype)
        )
        true_hist = true_hist.view(num_sub, num_bins)

        pred_hist = torch.zeros(num_sub * num_bins, device=device, dtype=dtype)
        deg_cap = float(num_bins - 1)
        pred_deg_cap = pred_deg.clamp(min=0.0, max=deg_cap)
        low = torch.floor(pred_deg_cap).long()
        high = torch.clamp(low + 1, max=num_bins - 1)
        w_high = (pred_deg_cap - low.to(dtype=dtype)).clamp(min=0.0, max=1.0)
        w_low = 1.0 - w_high
        low_idx = node_sub * num_bins + low
        high_idx = node_sub * num_bins + high
        pred_hist.scatter_add_(0, low_idx, w_low)
        pred_hist.scatter_add_(0, high_idx, w_high)
        pred_hist = pred_hist.view(num_sub, num_bins)

        active_rows = true_hist.sum(dim=1) > 0
        if not active_rows.any():
            return pred.edge_attr.sum() * 0.0

        if self.subtype_degree_loss_normalize:
            true_row_sum = true_hist.sum(dim=1, keepdim=True).clamp(min=1.0)
            pred_row_sum = pred_hist.sum(dim=1, keepdim=True).clamp(min=1.0)
            true_hist = true_hist / true_row_sum
            pred_hist = pred_hist / pred_row_sum

        return torch.nn.functional.l1_loss(
            pred_hist[active_rows], true_hist[active_rows], reduction="mean"
        )

    def forward(self, pred, true_data, log: bool):
        if self.structure_only:
            loss_X = 0.0
            loss_E = 0.0
            relation_matrix_loss = self._relation_matrix_loss(pred, true_data)
            metapath2_loss = self._metapath2_subtype_loss(pred, true_data)
            subtype_degree_loss = self._subtype_degree_hist_loss(pred, true_data)
            device = relation_matrix_loss.device
            if self.relation_matrix_loss_total is None:
                self.relation_matrix_loss_total = torch.tensor(0.0, device=device)
                self.relation_matrix_loss_steps = torch.tensor(0.0, device=device)
            if self.metapath2_loss_total is None:
                self.metapath2_loss_total = torch.tensor(0.0, device=device)
                self.metapath2_loss_steps = torch.tensor(0.0, device=device)
            if self.subtype_degree_loss_total is None:
                self.subtype_degree_loss_total = torch.tensor(0.0, device=device)
                self.subtype_degree_loss_steps = torch.tensor(0.0, device=device)
            self.relation_matrix_loss_total += relation_matrix_loss.detach()
            self.relation_matrix_loss_steps += 1.0
            self.metapath2_loss_total += metapath2_loss.detach()
            self.metapath2_loss_steps += 1.0
            self.subtype_degree_loss_total += subtype_degree_loss.detach()
            self.subtype_degree_loss_steps += 1.0
            if log:
                to_log = {}
                if self.relation_matrix_loss_weight > 0:
                    to_log["train_loss/relation_matrix_L1"] = relation_matrix_loss.detach()
                if self.metapath2_loss_weight > 0:
                    to_log["train_loss/metapath2_L1"] = metapath2_loss.detach()
                if self.subtype_degree_loss_weight > 0:
                    to_log["train_loss/subtype_degree_L1"] = subtype_degree_loss.detach()
                if wandb.run:
                    wandb.log(to_log, commit=True)
            return (
                self.relation_matrix_loss_weight * relation_matrix_loss
                + self.metapath2_loss_weight * metapath2_loss
                + self.subtype_degree_loss_weight * subtype_degree_loss
            )
        loss_X = (
            self.node_loss(pred.node, true_data.node)
            if true_data.node.numel() > 0
            else 0.0
        )
        pred_edge = pred.edge_attr
        true_edge = true_data.edge_attr
        if true_edge.dim() > 1:
            true_labels = true_edge.argmax(dim=-1)
        else:
            true_labels = true_edge
        pred_edge_flat = pred_edge.reshape(-1, pred_edge.shape[-1])
        true_labels_flat = true_labels.reshape(-1)
        pred_labels_flat = pred_edge_flat.argmax(dim=-1)
        true_is_pos = true_labels_flat > 0
        pred_is_pos = pred_labels_flat > 0

        # Initialize counters lazily on correct device
        if self.edge_exist_correct is None:
            device = true_labels_flat.device
            self.edge_exist_correct = torch.tensor(0.0, device=device)
            self.edge_exist_total = torch.tensor(0.0, device=device)
            self.edge_pos_exist_correct = torch.tensor(0.0, device=device)
            self.edge_pos_exist_total = torch.tensor(0.0, device=device)
            self.edge_pos_class_correct = torch.tensor(0.0, device=device)
            self.edge_pos_class_total = torch.tensor(0.0, device=device)
            self.edge_pos_class_given_exist_correct = torch.tensor(0.0, device=device)
            self.edge_pos_class_given_exist_total = torch.tensor(0.0, device=device)
            # 细粒度损失指标
            self.edge_exist_loss_total = torch.tensor(0.0, device=device)
            self.edge_subtype_loss_total = torch.tensor(0.0, device=device)
            self.edge_pos_ce_total = torch.tensor(0.0, device=device)
            self.edge_pos_ce_samples = torch.tensor(0.0, device=device)
            self.edge_neg_ce_total = torch.tensor(0.0, device=device)
            self.edge_neg_ce_samples = torch.tensor(0.0, device=device)
            self.relation_matrix_loss_total = torch.tensor(0.0, device=device)
            self.relation_matrix_loss_steps = torch.tensor(0.0, device=device)
            self.metapath2_loss_total = torch.tensor(0.0, device=device)
            self.metapath2_loss_steps = torch.tensor(0.0, device=device)
            self.subtype_degree_loss_total = torch.tensor(0.0, device=device)
            self.subtype_degree_loss_steps = torch.tensor(0.0, device=device)
            # 负边准确率指标（用于自适应权重调整）
            self.edge_neg_exist_correct = torch.tensor(0.0, device=device)
            self.edge_neg_exist_total = torch.tensor(0.0, device=device)

        # Existence accuracy on all query edges
        self.edge_exist_correct += (pred_is_pos == true_is_pos).sum()
        self.edge_exist_total += true_labels_flat.numel()
        
        # 计算负边存在性准确率（用于自适应权重调整）
        neg_mask = ~true_is_pos  # 负边mask（真实标签为0）
        if neg_mask.any():
            # 负边预测为不存在（pred_is_pos == False）才算正确
            if self.edge_neg_exist_correct is None:
                device = true_labels_flat.device
                self.edge_neg_exist_correct = torch.tensor(0.0, device=device)
                self.edge_neg_exist_total = torch.tensor(0.0, device=device)
            self.edge_neg_exist_correct += (~pred_is_pos[neg_mask]).sum()
            self.edge_neg_exist_total += neg_mask.sum()

        # Existence accuracy on true positives only (recall for existence)
        if true_is_pos.any():
            self.edge_pos_exist_correct += pred_is_pos[true_is_pos].sum()
            self.edge_pos_exist_total += true_is_pos.sum()

        # Class accuracy conditioned on true positives
        if true_is_pos.any():
            self.edge_pos_class_correct += (pred_labels_flat[true_is_pos] == true_labels_flat[true_is_pos]).sum()
            self.edge_pos_class_total += true_is_pos.sum()
        # Class accuracy conditioned on (true positive AND predicted exists)
        # 只计算多子类别关系族（如隶属关系只有一种子类别，不应该参与类别准确率计算）
        pos_and_pred = true_is_pos & pred_is_pos
        if pos_and_pred.any() and self.edge_family_ranges and self.edge_family_multisub:
            # 只计算属于多子类别关系族的边
            multisub_mask = torch.zeros_like(pos_and_pred, dtype=torch.bool)
            for fam_name, (st, en) in self.edge_family_ranges.items():
                if fam_name in self.edge_family_multisub:
                    fam_mask = (true_labels_flat >= st) & (true_labels_flat < en)
                    multisub_mask |= fam_mask
            pos_and_pred_multisub = pos_and_pred & multisub_mask
            if pos_and_pred_multisub.any():
                self.edge_pos_class_given_exist_correct += (
                    pred_labels_flat[pos_and_pred_multisub] == true_labels_flat[pos_and_pred_multisub]
                ).sum()
                self.edge_pos_class_given_exist_total += pos_and_pred_multisub.sum()
        elif pos_and_pred.any() and not (self.edge_family_ranges and self.edge_family_multisub):
            # 如果没有关系族信息，使用原来的逻辑（向后兼容）
            self.edge_pos_class_given_exist_correct += (
                pred_labels_flat[pos_and_pred] == true_labels_flat[pos_and_pred]
            ).sum()
            self.edge_pos_class_given_exist_total += pos_and_pred.sum()
        # Hierarchical edge loss: 先存在性（Focal）再子类型（CE）。存在性梯度更强，故去噪时「有无边」变化大、
        # 「边存在时子类型」变化相对少；若希望子类型更易变，可增大 edge_subtype_weight。
        # 按关系族可预测空间：仅一种子类别的关系族（如隶属）等价于只做存在性；多子类别关系族（如撰写）做存在性+子类型（仅 edge_family_multisub 参与 subtype CE）。
        exist_logits = torch.stack(
            [pred_edge_flat[:, 0], torch.logsumexp(pred_edge_flat[:, 1:], dim=-1)],
            dim=-1,
        )
        exist_targets = (true_labels_flat > 0).long()
        ce_exist = torch.nn.functional.cross_entropy(
            exist_logits, exist_targets, reduction="none"
        )
        pt_exist = torch.exp(-ce_exist)
        gamma_pos = self.edge_focal_gamma_pos * self.focal_gamma_scale
        gamma_neg = self.edge_focal_gamma_neg * self.focal_gamma_scale
        # 使用自适应调整的alpha_neg，而不是固定的
        alpha_neg = self.adaptive_alpha_neg * self.focal_alpha_neg_scale
        gamma_exist = torch.where(
            exist_targets == 1,
            torch.full_like(pt_exist, gamma_pos),
            torch.full_like(pt_exist, gamma_neg),
        )
        alpha_exist = torch.where(
            exist_targets == 1,
            torch.ones_like(pt_exist),
            torch.full_like(pt_exist, alpha_neg),
        )
        
        # 传统Focal Loss（不使用MoE风格）
        # 根据性能自适应调整alpha_neg来控制正负边平衡
        loss_exist = (alpha_exist * (1.0 - pt_exist) ** gamma_exist * ce_exist).sum()

        loss_subtype = pred_edge_flat.sum() * 0.0
        if self.use_edge_subtype_ce and self.edge_family_ranges and self.edge_family_multisub:
            for fam_name, (st, en) in self.edge_family_ranges.items():
                if fam_name not in self.edge_family_multisub:
                    continue
                fam_mask = (true_labels_flat >= st) & (true_labels_flat < en)
                if fam_mask.any():
                    local_targets = true_labels_flat[fam_mask] - st
                    class_weight = None
                    if self.pos_weight_per_edge_type is not None:
                        weight_vec = self.pos_weight_per_edge_type.to(true_labels_flat.device)
                        class_weight = weight_vec[st:en] * self.pos_weight_scale
                    ce_sub = torch.nn.functional.cross_entropy(
                        pred_edge_flat[fam_mask][:, st:en],
                        local_targets,
                        reduction="sum",
                        weight=class_weight,
                    )
                    loss_subtype = loss_subtype + ce_sub

        loss_E = self.edge_exist_weight * loss_exist + self.edge_subtype_weight * loss_subtype
        # Keep logging metrics on full edge distribution
        self.edge_loss.update(pred_edge_flat, true_labels_flat)
        self.edge_pos_acc.update(pred.edge_attr, true_data.edge_attr)
        
        # 记录细粒度损失指标
        if self.edge_exist_loss_total is not None:
            self.edge_exist_loss_total += loss_exist.detach()
            self.edge_subtype_loss_total += loss_subtype.detach()
            # 分别计算正边和负边的CE
            pos_mask = (true_labels_flat > 0)
            neg_mask = (true_labels_flat == 0)
            if pos_mask.any():
                pos_ce = torch.nn.functional.cross_entropy(
                    pred_edge_flat[pos_mask], true_labels_flat[pos_mask], reduction="sum"
                )
                self.edge_pos_ce_total += pos_ce.detach()
                self.edge_pos_ce_samples += pos_mask.sum().float()
            if neg_mask.any():
                neg_ce = torch.nn.functional.cross_entropy(
                    pred_edge_flat[neg_mask], true_labels_flat[neg_mask], reduction="sum"
                )
                self.edge_neg_ce_total += neg_ce.detach()
                self.edge_neg_ce_samples += neg_mask.sum().float()
        loss_y = 0.0
        loss_charge = self.charge_loss(pred.charge, true_data.charge) if pred.charge.numel() > 0 else 0.0
        relation_matrix_loss = self._relation_matrix_loss(pred, true_data)
        if self.relation_matrix_loss_total is not None:
            self.relation_matrix_loss_total += relation_matrix_loss.detach()
            self.relation_matrix_loss_steps += 1.0
        metapath2_loss = self._metapath2_subtype_loss(pred, true_data)
        if self.metapath2_loss_total is not None:
            self.metapath2_loss_total += metapath2_loss.detach()
            self.metapath2_loss_steps += 1.0
        subtype_degree_loss = self._subtype_degree_hist_loss(pred, true_data)
        if self.subtype_degree_loss_total is not None:
            self.subtype_degree_loss_total += subtype_degree_loss.detach()
            self.subtype_degree_loss_steps += 1.0

        if log:
            to_log = {}
            # 精简指标：只保留需要的
            if self.node_loss.total_samples > 0:
                to_log["train_loss/x_CE"] = self.node_loss.compute()
            if self.edge_loss.total_samples > 0:
                to_log["train_loss/E_CE"] = self.edge_loss.compute()
            # 细粒度损失指标
            if self.edge_pos_ce_samples is not None and self.edge_pos_ce_samples > 0:
                to_log["train_loss/E_pos_CE"] = (self.edge_pos_ce_total / self.edge_pos_ce_samples).item()
            if self.edge_neg_ce_samples is not None and self.edge_neg_ce_samples > 0:
                to_log["train_loss/E_neg_CE"] = (self.edge_neg_ce_total / self.edge_neg_ce_samples).item()
            if self.relation_matrix_loss_weight > 0:
                to_log["train_loss/relation_matrix_L1"] = relation_matrix_loss.detach()
            if self.metapath2_loss_weight > 0:
                to_log["train_loss/metapath2_L1"] = metapath2_loss.detach()
            if self.subtype_degree_loss_weight > 0:
                to_log["train_loss/subtype_degree_L1"] = subtype_degree_loss.detach()
            # 准确率指标
            if self.edge_pos_exist_total is not None and self.edge_pos_exist_total > 0:
                to_log["train_acc/E_pos_exist_acc"] = self.edge_pos_exist_correct / self.edge_pos_exist_total
            if (
                self.edge_pos_class_given_exist_total is not None
                and self.edge_pos_class_given_exist_total > 0
            ):
                to_log["train_acc/E_pos_class_given_exist_acc"] = (
                    self.edge_pos_class_given_exist_correct
                    / self.edge_pos_class_given_exist_total
                )
            if wandb.run:
                wandb.log(to_log, commit=True)

        return (
            loss_X
            + self.lambda_train[0] * loss_E
            + self.lambda_train[1] * loss_y
            + self.lambda_train[2] * loss_charge
            + self.relation_matrix_loss_weight * relation_matrix_loss
            + self.metapath2_loss_weight * metapath2_loss
            + self.subtype_degree_loss_weight * subtype_degree_loss
        )

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.edge_pos_acc, self.y_loss]:
            metric.reset()
        self.edge_exist_correct = None
        self.edge_exist_total = None
        self.edge_pos_exist_correct = None
        self.edge_pos_exist_total = None
        self.edge_pos_class_correct = None
        self.edge_pos_class_total = None
        self.edge_pos_class_given_exist_correct = None
        self.edge_pos_class_given_exist_total = None
        # 细粒度损失指标
        self.edge_exist_loss_total = None
        self.edge_subtype_loss_total = None
        self.edge_pos_ce_total = None
        self.edge_pos_ce_samples = None
        self.edge_neg_ce_total = None
        self.edge_neg_ce_samples = None
        self.relation_matrix_loss_total = None
        self.relation_matrix_loss_steps = None
        self.metapath2_loss_total = None
        self.metapath2_loss_steps = None
        self.subtype_degree_loss_total = None
        self.subtype_degree_loss_steps = None

    def log_epoch_metrics(self, log_step=None):
        """log_step: 若传入（如 current_epoch），wandb 横轴为该值，便于「每 epoch 一条」显示为 0,1,2,..."""
        epoch_node_loss = (
            self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        )
        epoch_edge_loss = (
            self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        )
        epoch_edge_pos_acc = (
            self.edge_pos_acc.compute()
            if hasattr(self.edge_pos_acc, "total") and self.edge_pos_acc.total > 0
            else -1
        )
        epoch_edge_exist_acc = (
            (self.edge_exist_correct / self.edge_exist_total)
            if self.edge_exist_total is not None and self.edge_exist_total > 0
            else -1
        )
        epoch_edge_pos_exist_acc = (
            (self.edge_pos_exist_correct / self.edge_pos_exist_total)
            if self.edge_pos_exist_total is not None and self.edge_pos_exist_total > 0
            else -1
        )
        # 计算负边存在性准确率
        epoch_edge_neg_exist_acc = (
            (self.edge_neg_exist_correct / self.edge_neg_exist_total)
            if self.edge_neg_exist_total is not None and self.edge_neg_exist_total > 0
            else -1
        )
        
        # 基于性能的自适应alpha_neg调整（不使用MoE风格）
        if (epoch_edge_pos_exist_acc != -1 and epoch_edge_neg_exist_acc != -1 and 
            self.edge_pos_ce_samples is not None and self.edge_pos_ce_samples > 0 and
            self.edge_neg_ce_samples is not None and self.edge_neg_ce_samples > 0):
            
            # 获取当前epoch的CE
            current_pos_ce = (self.edge_pos_ce_total / self.edge_pos_ce_samples).item()
            current_neg_ce = (self.edge_neg_ce_total / self.edge_neg_ce_samples).item()
            
            # 方案1: 直接响应负边CE变化（主要机制）
            # 如果负边CE上升，直接增加alpha_neg（更关注负边）
            if self.prev_neg_ce is not None and current_neg_ce > self.prev_neg_ce * 1.02:  # 负边CE上升超过2%
                # 负边性能恶化，增加alpha_neg（更关注负边）
                alpha_adjustment = self.adaptive_smoothing * (current_neg_ce / self.prev_neg_ce - 1.0)
                self.adaptive_alpha_neg = min(2.0, self.adaptive_alpha_neg + alpha_adjustment)
            elif self.prev_neg_ce is not None and current_neg_ce < self.prev_neg_ce * 0.98:  # 负边CE下降超过2%
                # 负边性能改善，可以稍微减少alpha_neg（但不要减少太多）
                alpha_adjustment = self.adaptive_smoothing * (1.0 - current_neg_ce / self.prev_neg_ce) * 0.5  # 减少幅度减半
                self.adaptive_alpha_neg = max(0.1, self.adaptive_alpha_neg - alpha_adjustment)
            
            # 方案2: 基于损失比例调整（辅助机制，用于平衡）
            # 如果正边CE相对上升很多，稍微减少alpha_neg（但不要太多，因为负边可能也在恶化）
            if self.prev_pos_ce is not None and self.prev_neg_ce is not None:
                prev_ce_ratio = self.prev_pos_ce / self.prev_neg_ce if self.prev_neg_ce > 0 else 1.0
                current_ce_ratio = current_pos_ce / current_neg_ce if current_neg_ce > 0 else 1.0
                
                # 如果正边CE相对上升很多（超过10%），稍微减少alpha_neg
                if current_ce_ratio > prev_ce_ratio * 1.1:  # 正边CE相对上升超过10%
                    alpha_adjustment = self.adaptive_smoothing * (current_ce_ratio / prev_ce_ratio - 1.0) * 0.3  # 减少幅度减小
                    self.adaptive_alpha_neg = max(0.1, self.adaptive_alpha_neg - alpha_adjustment)
                # 如果变化很小，保持当前alpha_neg（轻微衰减到基础值）
                elif abs(current_ce_ratio - prev_ce_ratio) / prev_ce_ratio < 0.05:  # 变化小于5%
                    # 轻微衰减到基础值（平滑）
                    self.adaptive_alpha_neg = 0.99 * self.adaptive_alpha_neg + 0.01 * self.base_alpha_neg
            
            # 方案2: 基于准确率差异调整alpha_neg（备选）
            # 如果正边准确率低，减少alpha_neg（更关注正边）；如果负边准确率低，增加alpha_neg（更关注负边）
            acc_diff = epoch_edge_pos_exist_acc - epoch_edge_neg_exist_acc
            if abs(acc_diff) > 0.05:  # 准确率差异超过5%
                if acc_diff < 0:  # 正边准确率更低
                    # 减少alpha_neg，让正边损失权重相对增加
                    alpha_adjustment = self.adaptive_smoothing * abs(acc_diff)
                    self.adaptive_alpha_neg = max(0.1, self.adaptive_alpha_neg - alpha_adjustment)
                else:  # 负边准确率更低
                    # 增加alpha_neg，让负边损失权重相对增加
                    alpha_adjustment = self.adaptive_smoothing * abs(acc_diff)
                    self.adaptive_alpha_neg = min(2.0, self.adaptive_alpha_neg + alpha_adjustment)
            
            # 更新历史值
            self.prev_pos_ce = current_pos_ce
            self.prev_neg_ce = current_neg_ce
        elif self.prev_pos_ce is None:
            # 初始化历史值
            if self.edge_pos_ce_samples is not None and self.edge_pos_ce_samples > 0:
                self.prev_pos_ce = (self.edge_pos_ce_total / self.edge_pos_ce_samples).item()
            if self.edge_neg_ce_samples is not None and self.edge_neg_ce_samples > 0:
                self.prev_neg_ce = (self.edge_neg_ce_total / self.edge_neg_ce_samples).item()
        epoch_edge_pos_class_acc = (
            (self.edge_pos_class_correct / self.edge_pos_class_total)
            if self.edge_pos_class_total is not None and self.edge_pos_class_total > 0
            else -1
        )
        epoch_edge_pos_class_given_exist_acc = (
            (self.edge_pos_class_given_exist_correct / self.edge_pos_class_given_exist_total)
            if (
                self.edge_pos_class_given_exist_total is not None
                and self.edge_pos_class_given_exist_total > 0
            )
            else -1
        )
        epoch_y_loss = (
            self.y_loss.compute() if self.y_loss.total_samples > 0 else -1
        )
        epoch_charge_loss = (
            self.charge_loss.compute() if self.charge_loss.total_samples > 0 else -1
        )

        to_log = {}
        # 精简指标：只保留需要的
        if epoch_node_loss != -1:
            to_log["train_epoch/x_CE"] = epoch_node_loss
        if epoch_edge_loss != -1:
            to_log["train_epoch/E_CE"] = epoch_edge_loss
        # 细粒度损失指标（epoch级别）
        if self.edge_pos_ce_samples is not None and self.edge_pos_ce_samples > 0:
            to_log["train_epoch/E_pos_CE"] = (self.edge_pos_ce_total / self.edge_pos_ce_samples).item()
        if self.edge_neg_ce_samples is not None and self.edge_neg_ce_samples > 0:
            to_log["train_epoch/E_neg_CE"] = (self.edge_neg_ce_total / self.edge_neg_ce_samples).item()
        if (
            self.relation_matrix_loss_weight > 0
            and self.relation_matrix_loss_total is not None
            and self.relation_matrix_loss_steps is not None
            and self.relation_matrix_loss_steps > 0
        ):
            to_log["train_epoch/relation_matrix_L1"] = (
                self.relation_matrix_loss_total / self.relation_matrix_loss_steps
            ).item()
        if (
            self.metapath2_loss_weight > 0
            and self.metapath2_loss_total is not None
            and self.metapath2_loss_steps is not None
            and self.metapath2_loss_steps > 0
        ):
            to_log["train_epoch/metapath2_L1"] = (
                self.metapath2_loss_total / self.metapath2_loss_steps
            ).item()
        if (
            self.subtype_degree_loss_weight > 0
            and self.subtype_degree_loss_total is not None
            and self.subtype_degree_loss_steps is not None
            and self.subtype_degree_loss_steps > 0
        ):
            to_log["train_epoch/subtype_degree_L1"] = (
                self.subtype_degree_loss_total / self.subtype_degree_loss_steps
            ).item()
        # 准确率指标
        if epoch_edge_pos_exist_acc != -1:
            to_log["train_epoch/E_pos_exist_acc"] = epoch_edge_pos_exist_acc
        if epoch_edge_pos_class_given_exist_acc != -1:
            to_log["train_epoch/E_pos_class_given_exist_acc"] = (
                epoch_edge_pos_class_given_exist_acc
            )
        # 记录自适应权重信息
        to_log["train_epoch/adaptive_alpha_neg"] = self.adaptive_alpha_neg
        if epoch_edge_neg_exist_acc != -1:
            to_log["train_epoch/E_neg_exist_acc"] = epoch_edge_neg_exist_acc
        if wandb.run:
            if log_step is not None:
                wandb.log(to_log, step=log_step, commit=False)
            else:
                wandb.log(to_log, commit=False)

        return to_log
