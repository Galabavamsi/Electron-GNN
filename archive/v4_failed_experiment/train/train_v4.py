import argparse
import atexit
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mace_net import SpectralEquivariantGNN
from train.dataset import SpectrumDataset
from train.losses_v4 import v4_loss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_partial_state_dict(model, ckpt_path, device):
    """Load compatible weights from a checkpoint and skip mismatched tensors."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        return 0, 0

    loaded = torch.load(ckpt_path, map_location=device, weights_only=True)
    current = model.state_dict()
    compatible = {k: v for k, v in loaded.items() if k in current and current[k].shape == v.shape}
    current.update(compatible)
    model.load_state_dict(current)
    return len(compatible), len(current)


def make_train_val_indices(num_samples, val_ratio, seed):
    """Deterministic split for tiny datasets while keeping at least one train sample."""
    if num_samples < 2 or val_ratio <= 0.0:
        idx = list(range(num_samples))
        return idx, idx

    n_val = int(round(num_samples * val_ratio))
    n_val = max(1, min(num_samples - 1, n_val))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _init_metric_sums():
    return {
        "total": 0.0,
        "loss_bip": 0.0,
        "loss_spectrum": 0.0,
        "loss_physics": 0.0,
        "s_kk": 0.0,
        "s_pos": 0.0,
        "s_sum": 0.0,
        "s_smooth": 0.0,
        "s_stab": 0.0,
        "trust_mean": 0.0,
        "trust_max": 0.0,
    }


def _accumulate(metric_sums, metrics):
    for key in metric_sums.keys():
        metric_sums[key] += _to_float(metrics[key])


def _average(metric_sums, n_batches):
    return {k: v / max(1, n_batches) for k, v in metric_sums.items()}


def train_epoch(model, dataloader, optimizer, device, loss_kwargs, grad_clip=1.0):
    model.train()
    sums = _init_metric_sums()

    for batch in tqdm(dataloader, desc="Train V4"):
        batch = batch.to(device)
        optimizer.zero_grad()

        pred_dict = model(batch)
        total, metrics = v4_loss(pred_dict, batch, **loss_kwargs)

        total.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        _accumulate(sums, metrics)

    return _average(sums, len(dataloader))


def evaluate_epoch(model, dataloader, device, loss_kwargs):
    model.eval()
    sums = _init_metric_sums()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval V4"):
            batch = batch.to(device)
            pred_dict = model(batch)
            _, metrics = v4_loss(pred_dict, batch, **loss_kwargs)
            _accumulate(sums, metrics)

    return _average(sums, len(dataloader))


def main():
    parser = argparse.ArgumentParser(description="Train V4 predictor + verifier stack")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_file", type=str, default="results/v4_train_output.log")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--init_ckpt", type=str, default="checkpoints/best_model.pth")

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--k_max", type=int, default=64)
    parser.add_argument("--amp_scale", type=float, default=1e-3)

    parser.add_argument("--lambda_spectrum", type=float, default=0.3)
    parser.add_argument("--lambda_kk", type=float, default=0.02)
    parser.add_argument("--lambda_pos", type=float, default=0.02)
    parser.add_argument("--lambda_sum", type=float, default=0.10)
    parser.add_argument("--lambda_smooth", type=float, default=0.005)
    parser.add_argument("--lambda_stab", type=float, default=0.02)

    parser.add_argument("--w_kk", type=float, default=1.0)
    parser.add_argument("--w_pos", type=float, default=1.0)
    parser.add_argument("--w_sum", type=float, default=1.0)
    parser.add_argument("--w_smooth", type=float, default=0.25)
    parser.add_argument("--w_stab", type=float, default=1.0)

    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_path = args.log_file
    if not os.path.isabs(log_path):
        log_path = os.path.join(repo_root, log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fp = open(log_path, "w", encoding="utf-8")
    atexit.register(log_fp.close)

    def log(msg=""):
        print(msg)
        log_fp.write(f"{msg}\n")
        log_fp.flush()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    log(f"Writing log to: {log_path}")

    dataset = SpectrumDataset(args.data_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No dataset found in {args.data_dir}")

    train_idx, val_idx = make_train_val_indices(len(dataset), args.val_ratio, args.seed)
    if len(dataset) >= 2 and args.val_ratio > 0.0 and train_idx != val_idx:
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        log(f"Train/val split: train={len(train_idx)} val={len(val_idx)} (val_ratio={args.val_ratio:.2f})")
    else:
        train_dataset = dataset
        val_dataset = dataset
        log("Using full dataset for both train and val (split disabled or dataset too small).")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = SpectralEquivariantGNN(
        node_features_in=5,
        hidden_dim=args.hidden_dim,
        K_max=args.k_max,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        amp_scale=args.amp_scale,
    ).to(device)

    init_ckpt = args.init_ckpt
    if init_ckpt and not os.path.isabs(init_ckpt):
        init_ckpt = os.path.join(repo_root, init_ckpt)
    loaded_count, total_count = load_partial_state_dict(model, init_ckpt, device)
    if loaded_count > 0:
        log(f"Warm-started V4 from {init_ckpt} ({loaded_count}/{total_count} tensors loaded)")
    else:
        log("No warm-start checkpoint loaded; training V4 from random init.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    loss_kwargs = {
        "lambda_spectrum": args.lambda_spectrum,
        "lambda_kk": args.lambda_kk,
        "lambda_pos": args.lambda_pos,
        "lambda_sum": args.lambda_sum,
        "lambda_smooth": args.lambda_smooth,
        "lambda_stab": args.lambda_stab,
        "verifier_weights": {
            "kk": args.w_kk,
            "pos": args.w_pos,
            "sum": args.w_sum,
            "smooth": args.w_smooth,
            "stab": args.w_stab,
        },
    }

    best_val = float("inf")
    best_path = os.path.join(args.save_dir, "best_model_v4.pth")

    log(f"Starting V4 training for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        log(f"\nEpoch {epoch}/{args.epochs}")

        train_m = train_epoch(model, train_loader, optimizer, device, loss_kwargs, grad_clip=args.grad_clip)
        val_m = evaluate_epoch(model, val_loader, device, loss_kwargs)

        scheduler.step(val_m["total"])

        log(
            "Train total={:.4f} bip={:.4f} spec={:.4f} phys={:.4f} trust={:.4f}".format(
                train_m["total"],
                train_m["loss_bip"],
                train_m["loss_spectrum"],
                train_m["loss_physics"],
                train_m["trust_mean"],
            )
        )
        log(
            "Val   total={:.4f} bip={:.4f} spec={:.4f} phys={:.4f} trust={:.4f}".format(
                val_m["total"],
                val_m["loss_bip"],
                val_m["loss_spectrum"],
                val_m["loss_physics"],
                val_m["trust_mean"],
            )
        )
        log(
            "Val verifier: kk={:.4f} pos={:.4f} sum={:.4f} smooth={:.4f} stab={:.4f}".format(
                val_m["s_kk"],
                val_m["s_pos"],
                val_m["s_sum"],
                val_m["s_smooth"],
                val_m["s_stab"],
            )
        )

        if val_m["total"] < best_val:
            best_val = val_m["total"]
            torch.save(model.state_dict(), best_path)
            log(f"New best V4 checkpoint saved: {best_path} (val_total={best_val:.4f})")

    torch.save(model.state_dict(), os.path.join(args.save_dir, "last_model_v4.pth"))
    log("V4 training complete.")


if __name__ == "__main__":
    main()
