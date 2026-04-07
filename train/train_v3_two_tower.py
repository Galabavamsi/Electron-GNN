import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mace_net import SpectralEquivariantGNN
from models.mace_net_v1 import SpectralEquivariantGNNV1
try:
    from train.dataset import SpectrumDataset
    from train.losses import bipartite_matching_loss, auto_differential_spectrum_loss
except ModuleNotFoundError:
    # Support direct execution: python train/train_v3_two_tower.py
    from dataset import SpectrumDataset
    from losses import bipartite_matching_loss, auto_differential_spectrum_loss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_partial_state_dict(model, ckpt_path, device):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return 0, 0

    loaded = torch.load(ckpt_path, map_location=device, weights_only=True)
    current = model.state_dict()

    compatible = {
        k: v for k, v in loaded.items() if k in current and current[k].shape == v.shape
    }
    current.update(compatible)
    model.load_state_dict(current)

    return len(compatible), len(current)


def set_backbone_trainable(model, trainable):
    """
    Toggle trainability of the legacy V1 backbone while keeping output heads trainable.
    """
    for name, param in model.named_parameters():
        if name.startswith("head_freq") or name.startswith("head_prob"):
            param.requires_grad = True
        else:
            param.requires_grad = bool(trainable)


def build_optimizer(params, lr):
    trainable = [p for p in params if p.requires_grad]
    return optim.AdamW(trainable, lr=lr, weight_decay=1e-4)


def infer_v1_kmax_from_checkpoint(ckpt_path):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "head_freq.2.bias" in state:
        return int(state["head_freq.2.bias"].numel())
    if "head_freq.2.weight" in state:
        return int(state["head_freq.2.weight"].shape[0])
    return None


def split_targets_by_graph(data_batch, batch_size, device):
    if hasattr(data_batch, "num_peaks"):
        counts = data_batch.num_peaks.reshape(-1).tolist()
    else:
        counts = [int(data_batch.y_freq.shape[0])] + [0] * (batch_size - 1)

    targets = []
    offset = 0
    for batch_idx in range(batch_size):
        num_true = int(counts[batch_idx]) if batch_idx < len(counts) else 0
        w_t = data_batch.y_freq[offset : offset + num_true].to(device)
        b_t = data_batch.y_amp[offset : offset + num_true].to(device)
        offset += num_true
        targets.append((w_t, b_t))

    return targets


def frequency_tower_loss(pred_dict, data_batch):
    p_pred = pred_dict["prob"]
    w_pred = pred_dict["freq"]

    batch_size = p_pred.shape[0]
    k_max = p_pred.shape[1]
    device = p_pred.device

    targets = split_targets_by_graph(data_batch, batch_size, device)
    total_loss = p_pred.new_tensor(0.0)

    for batch_idx in range(batch_size):
        w_t, _ = targets[batch_idx]
        num_true = int(w_t.numel())

        w_p = w_pred[batch_idx]
        p_p = p_pred[batch_idx]
        p_target = torch.zeros_like(p_p)

        if num_true > 0:
            cost_matrix = torch.cdist(w_p.unsqueeze(-1), w_t.unsqueeze(-1), p=1)
            pred_indices, true_indices = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

            pred_idx_t = torch.tensor(pred_indices, dtype=torch.long, device=device)
            true_idx_t = torch.tensor(true_indices, dtype=torch.long, device=device)
            p_target[pred_idx_t] = 1.0

            loss_w = F.smooth_l1_loss(w_p[pred_idx_t], w_t[true_idx_t], beta=0.02)
            unmatched_mask = torch.ones(k_max, dtype=torch.bool, device=device)
            unmatched_mask[pred_idx_t] = False
            # Keep unmatched high-frequency spikes suppressed.
            loss_unmatched = (w_p[unmatched_mask] ** 2).mean() if unmatched_mask.any() else w_p.new_tensor(0.0)
        else:
            loss_w = w_p.new_tensor(0.0)
            loss_unmatched = (w_p**2).mean()

        loss_prob = F.binary_cross_entropy(p_p, p_target)
        target_count = torch.tensor(float(num_true), device=device)
        loss_count = F.smooth_l1_loss(p_p.sum(), target_count, beta=1.0)

        total_loss += 12.0 * loss_w + 1.0 * loss_prob + 0.3 * loss_count + 0.02 * loss_unmatched

    return total_loss / batch_size


def train_freq_epoch(
    model,
    dataloader,
    optimizer,
    device,
    grad_clip=1.0,
    teacher_model=None,
    teacher_lambda=0.0,
    teacher_slots=0,
):
    model.train()
    epoch_loss = 0.0
    epoch_base = 0.0
    epoch_teacher = 0.0

    for batch in tqdm(dataloader, desc="Train Freq Tower"):
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        base_loss = frequency_tower_loss(pred, batch)

        teacher_loss = base_loss.new_tensor(0.0)
        if teacher_model is not None and teacher_lambda > 0.0 and teacher_slots > 0:
            with torch.no_grad():
                teacher_pred = teacher_model(batch)

            slots = min(teacher_slots, pred["freq"].shape[1], teacher_pred["freq"].shape[1])
            if slots > 0:
                teacher_loss = (
                    F.smooth_l1_loss(
                        pred["freq"][:, :slots],
                        teacher_pred["freq"][:, :slots],
                        beta=0.02,
                    )
                    + 0.25 * F.mse_loss(pred["prob"][:, :slots], teacher_pred["prob"][:, :slots])
                )

        loss = base_loss + teacher_lambda * teacher_loss

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_base += base_loss.item()
        epoch_teacher += teacher_loss.item()

    n = len(dataloader)
    return epoch_loss / n, epoch_base / n, epoch_teacher / n


def eval_freq_epoch(model, dataloader, device):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval Freq Tower"):
            batch = batch.to(device)
            pred = model(batch)
            loss = frequency_tower_loss(pred, batch)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train_amp_epoch(model, dataloader, optimizer, device, lambda_spectrum=0.3, grad_clip=1.0):
    model.train()
    epoch_bip = 0.0
    epoch_spec = 0.0

    for batch in tqdm(dataloader, desc="Train Amp Tower"):
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        loss_bip, _, _ = bipartite_matching_loss(pred, batch)
        loss_spec = auto_differential_spectrum_loss(pred, batch)
        total = loss_bip + lambda_spectrum * loss_spec

        total.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        epoch_bip += loss_bip.item()
        epoch_spec += loss_spec.item()

    return epoch_bip / len(dataloader), epoch_spec / len(dataloader)


def eval_amp_epoch(model, dataloader, device):
    model.eval()
    epoch_bip = 0.0
    epoch_spec = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval Amp Tower"):
            batch = batch.to(device)
            pred = model(batch)
            loss_bip, _, _ = bipartite_matching_loss(pred, batch)
            loss_spec = auto_differential_spectrum_loss(pred, batch)

            epoch_bip += loss_bip.item()
            epoch_spec += loss_spec.item()

    return epoch_bip / len(dataloader), epoch_spec / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train Version 3 two-tower hybrid models")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--epochs_freq", type=int, default=0)
    parser.add_argument("--lr_freq", type=float, default=3e-4)
    parser.add_argument("--k_max_freq", type=int, default=64)
    parser.add_argument("--hidden_dim_freq", type=int, default=64)
    parser.add_argument("--init_freq_ckpt", type=str, default="checkpoints/best_model_v1.pth")
    parser.add_argument("--freq_warmup_epochs", type=int, default=5)
    parser.add_argument("--freq_freeze_backbone", type=int, default=1)
    parser.add_argument("--freq_early_stop_patience", type=int, default=12)
    parser.add_argument("--freq_min_delta", type=float, default=1e-3)
    parser.add_argument("--freq_teacher_lambda", type=float, default=0.5)
    parser.add_argument("--freq_teacher_ckpt", type=str, default="checkpoints/best_model_v1.pth")

    parser.add_argument("--epochs_amp", type=int, default=60)
    parser.add_argument("--lr_amp", type=float, default=2e-4)
    parser.add_argument("--k_max_amp", type=int, default=64)
    parser.add_argument("--hidden_dim_amp", type=int, default=128)
    parser.add_argument("--num_layers_amp", type=int, default=4)
    parser.add_argument("--num_heads_amp", type=int, default=4)
    parser.add_argument("--dropout_amp", type=float, default=0.0)
    parser.add_argument("--amp_scale", type=float, default=1e-3)
    parser.add_argument("--lambda_spectrum", type=float, default=0.3)
    parser.add_argument("--init_amp_ckpt", type=str, default="checkpoints/best_model.pth")

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SpectrumDataset(args.data_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No dataset found in {args.data_dir}")

    # With current tiny dataset we use same split for train/val.
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    freq_ckpt_out = os.path.join(args.save_dir, "v3_freq_tower.pth")
    amp_ckpt_out = os.path.join(args.save_dir, "v3_amp_tower.pth")

    if args.epochs_freq > 0:
        print("\n=== Training Frequency Tower (V3) ===")
        freq_model = SpectralEquivariantGNNV1(
            node_features_in=5,
            hidden_dim=args.hidden_dim_freq,
            K_max=args.k_max_freq,
        ).to(device)
        loaded, total = load_partial_state_dict(freq_model, args.init_freq_ckpt, device)
        if loaded > 0:
            print(f"Loaded {loaded}/{total} compatible params into freq tower from {args.init_freq_ckpt}")

        teacher_model = None
        teacher_slots = 0
        teacher_kmax = infer_v1_kmax_from_checkpoint(args.freq_teacher_ckpt)
        if teacher_kmax is not None and os.path.exists(args.freq_teacher_ckpt):
            teacher_model = SpectralEquivariantGNNV1(
                node_features_in=5,
                hidden_dim=args.hidden_dim_freq,
                K_max=teacher_kmax,
            ).to(device)
            load_partial_state_dict(teacher_model, args.freq_teacher_ckpt, device)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            teacher_slots = min(args.k_max_freq, teacher_kmax)
            print(
                f"Using teacher regularization from {args.freq_teacher_ckpt} "
                f"for first {teacher_slots} slots (lambda={args.freq_teacher_lambda})."
            )

        warmup_epochs = min(args.freq_warmup_epochs, args.epochs_freq)
        freeze_backbone = bool(args.freq_freeze_backbone)
        unfrozen = False
        if freeze_backbone and warmup_epochs > 0:
            set_backbone_trainable(freq_model, trainable=False)
            print(f"Frequency tower warmup: freezing backbone for {warmup_epochs} epochs.")

        optimizer_f = build_optimizer(freq_model.parameters(), lr=args.lr_freq)
        scheduler_f = optim.lr_scheduler.ReduceLROnPlateau(optimizer_f, mode="min", factor=0.5, patience=8)

        best_freq = float("inf")
        no_improve = 0
        for epoch in range(1, args.epochs_freq + 1):
            print(f"\n[Freq Tower] Epoch {epoch}/{args.epochs_freq}")

            if freeze_backbone and (not unfrozen) and epoch == warmup_epochs + 1:
                set_backbone_trainable(freq_model, trainable=True)
                optimizer_f = build_optimizer(freq_model.parameters(), lr=args.lr_freq)
                scheduler_f = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_f, mode="min", factor=0.5, patience=8
                )
                unfrozen = True
                print("Unfroze frequency tower backbone after warmup.")

            train_total, train_base, train_teacher = train_freq_epoch(
                freq_model,
                train_loader,
                optimizer_f,
                device,
                grad_clip=args.grad_clip,
                teacher_model=teacher_model,
                teacher_lambda=args.freq_teacher_lambda,
                teacher_slots=teacher_slots,
            )
            val_loss = eval_freq_epoch(freq_model, val_loader, device)
            scheduler_f.step(val_loss)

            print(
                f"Freq tower losses - train(total/base/teacher): "
                f"{train_total:.4f}/{train_base:.4f}/{train_teacher:.4f} "
                f"val: {val_loss:.4f}"
            )

            if val_loss < (best_freq - args.freq_min_delta):
                best_freq = val_loss
                torch.save(freq_model.state_dict(), freq_ckpt_out)
                print(f"Saved best freq tower: {freq_ckpt_out} (val={best_freq:.4f})")
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= args.freq_early_stop_patience:
                print(
                    f"Early stopping frequency tower after {no_improve} non-improving epochs "
                    f"(patience={args.freq_early_stop_patience})."
                )
                break

    if args.epochs_amp > 0:
        print("\n=== Training Amplitude Tower (V3) ===")
        amp_model = SpectralEquivariantGNN(
            node_features_in=5,
            hidden_dim=args.hidden_dim_amp,
            K_max=args.k_max_amp,
            num_layers=args.num_layers_amp,
            num_heads=args.num_heads_amp,
            dropout=args.dropout_amp,
            amp_scale=args.amp_scale,
        ).to(device)

        if args.init_amp_ckpt and os.path.exists(args.init_amp_ckpt):
            amp_state = torch.load(args.init_amp_ckpt, map_location=device, weights_only=True)
            amp_model.load_state_dict(amp_state, strict=False)
            print(f"Loaded amp tower init checkpoint: {args.init_amp_ckpt}")

        optimizer_a = optim.AdamW(amp_model.parameters(), lr=args.lr_amp, weight_decay=1e-4)
        scheduler_a = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_a, mode="min", factor=0.5, patience=5
        )

        best_amp = float("inf")
        for epoch in range(1, args.epochs_amp + 1):
            print(f"\n[Amp Tower] Epoch {epoch}/{args.epochs_amp}")
            train_bip, train_spec = train_amp_epoch(
                amp_model,
                train_loader,
                optimizer_a,
                device,
                lambda_spectrum=args.lambda_spectrum,
                grad_clip=args.grad_clip,
            )
            val_bip, val_spec = eval_amp_epoch(amp_model, val_loader, device)

            val_total = val_bip + args.lambda_spectrum * val_spec
            scheduler_a.step(val_total)

            print(
                f"Amp tower - train(bip/spec): {train_bip:.4f}/{train_spec:.4f} "
                f"val(bip/spec): {val_bip:.4f}/{val_spec:.4f} total={val_total:.4f}"
            )

            if val_total < best_amp:
                best_amp = val_total
                torch.save(amp_model.state_dict(), amp_ckpt_out)
                print(f"Saved best amp tower: {amp_ckpt_out} (val_total={best_amp:.4f})")

    print("\nV3 two-tower training complete.")


if __name__ == "__main__":
    main()
