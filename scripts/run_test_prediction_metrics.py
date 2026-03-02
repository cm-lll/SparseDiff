#!/usr/bin/env python3
"""
独立脚本：在测试集上做「加噪 → 一步去噪预测」（与训练时 val_pred 流程一致），得到 test_pred 指标。

流程：测试集 batch → 加噪(apply_sparse_noise) → 多批遍历所有可能边、每批 forward 预测 → 合并为完整预测图 → 与测试集真实图对比。
验证/本脚本使用全量边覆盖（异质图每关系族所有可能 (u,v)），与完整去噪时的多批预测一致；扩散采样（test/）在原项目里跑即可。
"""

import argparse
import os
import sys

# 保证从项目根运行时可导入 sparse_diffusion 及其内部相对导入（models, diffusion_model_sparse 等）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SPARSE_DIR = os.path.join(_ROOT, "sparse_diffusion")
if _SPARSE_DIR not in sys.path:
    sys.path.insert(0, _SPARSE_DIR)
os.chdir(_ROOT)

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

# 延迟导入，避免未用到的依赖
def _build_cfg():
    with initialize_config_dir(config_dir=os.path.join(_ROOT, "configs"), version_base="1.3"):
        return compose(config_name="config", overrides=["+experiment=acm_train"])


def _build_acm_components(cfg):
    from sparse_diffusion.datasets.acm_subgraphs_dataset import ACMSubgraphsDataModule, ACMSubgraphsInfos
    from sparse_diffusion.metrics.abstract_metrics import TrainAbstractMetricsDiscrete
    from sparse_diffusion.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
    from sparse_diffusion.metrics.sampling_metrics import SamplingMetrics

    pl.seed_everything(cfg.train.seed)
    datamodule = ACMSubgraphsDataModule(cfg)
    dataset_infos = ACMSubgraphsInfos(datamodule)
    train_metrics = TrainAbstractMetricsDiscrete()
    domain_features = DummyExtraFeatures()
    dataloaders = datamodule.dataloaders

    ef = cfg.model.extra_features
    edge_f = cfg.model.edge_features
    extra_features = (
        ExtraFeatures(
            eigenfeatures=cfg.model.eigenfeatures,
            edge_features_type=edge_f,
            dataset_info=dataset_infos,
            num_eigenvectors=cfg.model.num_eigenvectors,
            num_eigenvalues=cfg.model.num_eigenvalues,
            num_degree=cfg.model.num_degree,
            dist_feat=cfg.model.dist_feat,
            use_positional=cfg.model.positional_encoding,
        )
        if ef is not None
        else DummyExtraFeatures()
    )
    dataset_infos.compute_input_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features,
    )

    val_sampling_metrics = SamplingMetrics(dataset_infos, test=False, dataloaders=dataloaders)
    test_sampling_metrics = SamplingMetrics(dataset_infos, test=True, dataloaders=dataloaders)

    return (
        datamodule,
        dataset_infos,
        train_metrics,
        extra_features,
        domain_features,
        val_sampling_metrics,
        test_sampling_metrics,
    )


def main():
    parser = argparse.ArgumentParser(description="Test-set prediction metrics (no sampling).")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (e.g. checkpoints/acm_train/last.ckpt)")
    parser.add_argument("--output", type=str, default=None, help="Optional output file for metrics dict (one line per key=value)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda:0 if available else cpu)")
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    print("Loading config and building components...")
    cfg = _build_cfg()
    (
        datamodule,
        dataset_infos,
        train_metrics,
        extra_features,
        domain_features,
        val_sampling_metrics,
        test_sampling_metrics,
    ) = _build_acm_components(cfg)

    from sparse_diffusion.diffusion_model_sparse import DiscreteDenoisingDiffusion
    import sparse_diffusion.utils as utils

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "extra_features": extra_features,
        "domain_features": domain_features,
        "val_sampling_metrics": val_sampling_metrics,
        "test_sampling_metrics": test_sampling_metrics,
    }
    model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # 旧 checkpoint 可能来自不同 hidden_dim / 噪声步数的模型，这里只加载形状完全匹配的参数，避免 size mismatch 报错
    loaded_state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []
    for k, v in loaded_state.items():
        if k in model_state and isinstance(v, type(model_state[k])) and hasattr(v, "shape") and hasattr(
            model_state[k], "shape"
        ):
            if v.shape == model_state[k].shape:
                filtered_state[k] = v
            else:
                skipped_keys.append(k)
        elif k in model_state:
            # 非 tensor 类型的 buffer，直接尝试加载
            filtered_state[k] = v
        else:
            skipped_keys.append(k)
    missing, unexp = model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded checkpoint with {len(filtered_state)} matching keys, skipped {len(skipped_keys)} mismatched keys.")
    if missing or unexp:
        print("Strict=False load summary -> missing:", missing, "unexpected:", unexp)

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    test_loader = datamodule.dataloaders["test"]
    # 加噪验证：对第一个 batch 做一次加噪并检查 clean vs noisy 是否不同
    with torch.no_grad():
        first_batch = next(iter(test_loader)).to(device)
        data = dataset_infos.to_one_hot(first_batch)
        sparse_noisy = model.apply_sparse_noise(data)
        clean_node = data.x.argmax(dim=-1) if data.x.dim() > 1 else data.x
        noisy_node = sparse_noisy["node_t"] if sparse_noisy["node_t"].dim() == 1 else sparse_noisy["node_t"].argmax(dim=-1)
        node_diff = (clean_node != noisy_node).float().mean().item()
        clean_edges = set(map(tuple, data.edge_index.t().tolist()))
        noisy_ei = sparse_noisy["edge_index_t"]
        if noisy_ei.dim() == 2 and noisy_ei.shape[1] > 0:
            noisy_edges = set(map(tuple, noisy_ei.t().tolist()))
        else:
            noisy_edges = set()
        edge_overlap = len(clean_edges & noisy_edges) / len(clean_edges) if clean_edges else 0.0
        print(f"[加噪验证] 节点变化比例={node_diff:.2%}, 边与原始重叠比例={edge_overlap:.2%} (加噪后应均<100%)")
    if edge_overlap >= 0.999 and node_diff < 0.001:
        print("[警告] 加噪几乎未改变数据，请检查 apply_sparse_noise 或数据格式。")

    predicted_list = []
    n_batches = 0
    print("测试集：加噪后多批全边预测并合并（与 val_pred 全量边一致）...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            model._val_predicted_graphs_list = []
            model.validation_step(batch, i)
            predicted_list.extend(model._val_predicted_graphs_list)
            n_batches += 1
    print(f"Processed {n_batches} test batches, {len(predicted_list)} placeholder(s).")

    if not predicted_list:
        print("No predicted graphs collected. Aborting.")
        sys.exit(1)

    pred_concat = utils.concat_sparse_graphs(predicted_list)
    num_graphs = int(pred_concat.batch.max().item()) + 1
    print(f"Concatenated {num_graphs} graphs.")

    test_sampling_metrics.reset()
    to_log, _ = test_sampling_metrics.compute_all_metrics(
        pred_concat,
        current_epoch=0,
        local_rank=0,
        key_suffix="_pred",
        chart_title_suffix=" 预测",
    )

    print("\n--- test_pred ---")
    for k, v in sorted(to_log.items()):
        print(f"  {k}: {v}")

    if args.output:
        with open(args.output, "w") as f:
            for k, v in sorted(to_log.items()):
                f.write(f"{k}={v}\n")
        print(f"\nWrote metrics to {args.output}")


if __name__ == "__main__":
    main()
