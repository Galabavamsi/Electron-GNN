import argparse
import atexit
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mace_net import SpectralEquivariantGNN
from models.mace_net_v1 import SpectralEquivariantGNNV1
from utils.hybrid_inference import decode_peak_set
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


def make_train_val_indices(num_samples, val_ratio, seed):
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


def _lorentzian_spectrum_np(freqs, amps, omega, gamma=0.015):
    spec = np.zeros_like(omega)
    if len(freqs) == 0:
        return spec
    for w_k, b_k in zip(freqs, amps):
        spec += b_k * (gamma / ((omega - w_k) ** 2 + gamma**2))
    return spec


def _spectral_overlap_np(pred_w, pred_b, true_w, true_b):
    omega = np.linspace(0.01, 5.0, 512)
    spec_p = _lorentzian_spectrum_np(pred_w, pred_b, omega)
    spec_t = _lorentzian_spectrum_np(true_w, true_b, omega)
    denom = np.linalg.norm(spec_p) * np.linalg.norm(spec_t)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(spec_p, spec_t) / denom)


def eval_amp_decode_quality(model, dataloader, device, prob_threshold=0.65, fallback_topk=8):
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            pred = model(batch)
            batch_size = pred["prob"].shape[0]
            targets = split_targets_by_graph(batch, batch_size, device)

            for batch_idx in range(batch_size):
                dec = decode_peak_set(
                    pred,
                    batch_idx=batch_idx,
                    prob_threshold=prob_threshold,
                    fallback_top_k=fallback_topk,
                    use_count_head=True,
                )
                pred_w = np.asarray(dec["freq"], dtype=np.float64)
                pred_b = np.asarray(dec["amp"], dtype=np.float64)

                true_w_t, true_b_t = targets[batch_idx]
                true_w = true_w_t.detach().cpu().numpy().astype(np.float64)
                true_b = np.abs(true_b_t.detach().cpu().numpy().astype(np.float64))

                overlap = _spectral_overlap_np(pred_w, pred_b, true_w, true_b)
                count_penalty = abs(len(pred_w) - len(true_w)) / max(1, len(true_w))
                scores.append(overlap - 0.25 * count_penalty)

    if not scores:
        return -1e9
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description="Train Version 3 two-tower hybrid models")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_file", type=str, default="results/v3_train_output.log")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.5)
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
    parser.add_argument("--amp_early_stop_patience", type=int, default=12)
    parser.add_argument("--amp_score_min_delta", type=float, default=1e-4)

    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_file = None
    if args.log_file:
        log_path = args.log_file
        if not os.path.isabs(log_path):
            log_path = os.path.join(repo_root, log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        atexit.register(log_file.close)

    def log(msg=""):
        print(msg)
        if log_file is not None:
            log_file.write(f"{msg}\n")
            log_file.flush()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    if log_file is not None:
        log(f"Writing log to: {log_file.name}")

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

    freq_ckpt_out = os.path.join(args.save_dir, "v3_freq_tower.pth")
    amp_ckpt_out = os.path.join(args.save_dir, "v3_amp_tower.pth")

    if args.epochs_freq > 0:
        log("\n=== Training Frequency Tower (V3) ===")
        freq_model = SpectralEquivariantGNNV1(
            node_features_in=5,
            hidden_dim=args.hidden_dim_freq,
            K_max=args.k_max_freq,
        ).to(device)
        loaded, total = load_partial_state_dict(freq_model, args.init_freq_ckpt, device)
        if loaded > 0:
            log(f"Loaded {loaded}/{total} compatible params into freq tower from {args.init_freq_ckpt}")

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
            log(
                f"Using teacher regularization from {args.freq_teacher_ckpt} "
                f"for first {teacher_slots} slots (lambda={args.freq_teacher_lambda})."
            )

        warmup_epochs = min(args.freq_warmup_epochs, args.epochs_freq)
        freeze_backbone = bool(args.freq_freeze_backbone)
        unfrozen = False
        if freeze_backbone and warmup_epochs > 0:
            set_backbone_trainable(freq_model, trainable=False)
            log(f"Frequency tower warmup: freezing backbone for {warmup_epochs} epochs.")

        optimizer_f = build_optimizer(freq_model.parameters(), lr=args.lr_freq)
        scheduler_f = optim.lr_scheduler.ReduceLROnPlateau(optimizer_f, mode="min", factor=0.5, patience=8)

        best_freq = float("inf")
        no_improve = 0
        for epoch in range(1, args.epochs_freq + 1):
            log(f"\n[Freq Tower] Epoch {epoch}/{args.epochs_freq}")

            if freeze_backbone and (not unfrozen) and epoch == warmup_epochs + 1:
                set_backbone_trainable(freq_model, trainable=True)
                optimizer_f = build_optimizer(freq_model.parameters(), lr=args.lr_freq)
                scheduler_f = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_f, mode="min", factor=0.5, patience=8
                )
                unfrozen = True
                log("Unfroze frequency tower backbone after warmup.")

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

            log(
                f"Freq tower losses - train(total/base/teacher): "
                f"{train_total:.4f}/{train_base:.4f}/{train_teacher:.4f} "
                f"val: {val_loss:.4f}"
            )

            if val_loss < (best_freq - args.freq_min_delta):
                best_freq = val_loss
                torch.save(freq_model.state_dict(), freq_ckpt_out)
                log(f"Saved best freq tower: {freq_ckpt_out} (val={best_freq:.4f})")
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= args.freq_early_stop_patience:
                log(
                    f"Early stopping frequency tower after {no_improve} non-improving epochs "
                    f"(patience={args.freq_early_stop_patience})."
                )
                break

    if args.epochs_amp > 0:
        log("\n=== Training Amplitude Tower (V3) ===")
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
            log(f"Loaded amp tower init checkpoint: {args.init_amp_ckpt}")

        optimizer_a = optim.AdamW(amp_model.parameters(), lr=args.lr_amp, weight_decay=1e-4)
        scheduler_a = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_a, mode="min", factor=0.5, patience=5
        )

        best_amp = float("inf")
        best_amp_score = -1e9
        no_improve_amp = 0
        for epoch in range(1, args.epochs_amp + 1):
            log(f"\n[Amp Tower] Epoch {epoch}/{args.epochs_amp}")
            train_bip, train_spec = train_amp_epoch(
                amp_model,
                train_loader,
                optimizer_a,
                device,
                lambda_spectrum=args.lambda_spectrum,
                grad_clip=args.grad_clip,
            )
            val_bip, val_spec = eval_amp_epoch(amp_model, val_loader, device)
            val_quality = eval_amp_decode_quality(
                amp_model,
                val_loader,
                device,
                prob_threshold=0.65,
                fallback_topk=8,
            )

            val_total = val_bip + args.lambda_spectrum * val_spec
            scheduler_a.step(val_total)

            log(
                f"Amp tower - train(bip/spec): {train_bip:.4f}/{train_spec:.4f} "
                f"val(bip/spec): {val_bip:.4f}/{val_spec:.4f} total={val_total:.4f} "
                f"quality={val_quality:.4f}"
            )

            improved_quality = val_quality > (best_amp_score + args.amp_score_min_delta)
            improved_total = val_total < best_amp
            if improved_quality or (abs(val_quality - best_amp_score) <= args.amp_score_min_delta and improved_total):
                best_amp = val_total
                best_amp_score = val_quality
                torch.save(amp_model.state_dict(), amp_ckpt_out)
                log(
                    f"Saved best amp tower: {amp_ckpt_out} "
                    f"(val_total={best_amp:.4f}, quality={best_amp_score:.4f})"
                )
                no_improve_amp = 0
            else:
                no_improve_amp += 1

            if no_improve_amp >= args.amp_early_stop_patience:
                log(
                    f"Early stopping amp tower after {no_improve_amp} non-improving epochs "
                    f"(patience={args.amp_early_stop_patience})."
                )
                break

    log("\nV3 two-tower training complete.")


if __name__ == "__main__":
    main()
