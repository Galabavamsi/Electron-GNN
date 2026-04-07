import argparse
import os
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mace_net import SpectralEquivariantGNN
from models.mace_net_v1 import SpectralEquivariantGNNV1
from train.dataset import SpectrumDataset
from utils.hybrid_inference import decode_peak_set, combine_two_tower_predictions


def prepare_graph(graph_data, device):
    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
    graph_data.y_freq_batch = torch.zeros(graph_data.y_freq.shape[0], dtype=torch.long)
    return graph_data.to(device)


def lorentzian_spectrum(freqs, amps, omega_grid, gamma=0.015):
    spec = np.zeros_like(omega_grid)
    for w_k, b_k in zip(freqs, amps):
        spec += b_k * (gamma / ((omega_grid - w_k) ** 2 + gamma**2))
    return spec


def spectral_overlap(pred_w, pred_b, true_w, true_b):
    omega = np.linspace(0.01, 5.0, 1024)
    spec_p = lorentzian_spectrum(pred_w, pred_b, omega)
    spec_t = lorentzian_spectrum(true_w, true_b, omega)
    denom = np.linalg.norm(spec_p) * np.linalg.norm(spec_t)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(spec_p, spec_t) / denom)


def matched_mae(pred_w, pred_b, true_w, true_b):
    if len(pred_w) == 0 or len(true_w) == 0:
        return float("nan"), float("nan")

    cost = 10.0 * np.abs(pred_w[:, None] - true_w[None, :]) + np.abs(pred_b[:, None] - true_b[None, :])
    pred_idx, true_idx = linear_sum_assignment(cost)

    f_mae = float(np.mean(np.abs(pred_w[pred_idx] - true_w[true_idx])))
    a_mae = float(np.mean(np.abs(pred_b[pred_idx] - true_b[true_idx])))
    return f_mae, a_mae


def load_v1_model(ckpt_path, device, k_max):
    model = SpectralEquivariantGNNV1(node_features_in=5, K_max=k_max)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_v2_model(ckpt_path, device, k_max):
    model = SpectralEquivariantGNN(node_features_in=5, K_max=k_max)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def metric_row(name, pred_w, pred_b, true_w, true_b):
    f_mae, a_mae = matched_mae(pred_w, pred_b, true_w, true_b)
    overlap = spectral_overlap(pred_w, pred_b, true_w, true_b)
    return {
        "model": name,
        "freq_mae": f_mae,
        "amp_mae": a_mae,
        "overlap": overlap,
        "num_pred": int(len(pred_w)),
        "num_true": int(len(true_w)),
    }


def format_row(row):
    return (
        f"{row['model']:<18} "
        f"freq_mae={row['freq_mae']:.5f} "
        f"amp_mae={row['amp_mae']:.6e} "
        f"overlap={row['overlap']:.4f} "
        f"pred={row['num_pred']:>3d} true={row['num_true']:>3d}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate V1, V2, and hybrid two-tower inference.")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--v1_ckpt", type=str, default="checkpoints/best_model_v1.pth")
    parser.add_argument("--v2_ckpt", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--v1_kmax", type=int, default=50)
    parser.add_argument("--v2_kmax", type=int, default=64)
    parser.add_argument("--prob_threshold", type=float, default=0.65)
    parser.add_argument("--fallback_topk", type=int, default=5)
    parser.add_argument("--hybrid_min_freq_separation", type=float, default=0.005)
    parser.add_argument("--disable_hybrid_amp_overflow", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.v1_ckpt):
        raise FileNotFoundError(f"V1 checkpoint not found: {args.v1_ckpt}")
    if not os.path.exists(args.v2_ckpt):
        raise FileNotFoundError(f"V2 checkpoint not found: {args.v2_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SpectrumDataset(args.data_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No processed dataset files found in {args.data_dir}")

    v1_model = load_v1_model(args.v1_ckpt, device, args.v1_kmax)
    v2_model = load_v2_model(args.v2_ckpt, device, args.v2_kmax)

    all_rows = {"V1": [], "V2": [], "Hybrid": []}

    for i, file_path in enumerate(dataset.data_files):
        sample_name = os.path.basename(file_path).replace("_targets.pt", "")
        graph_data = prepare_graph(dataset[i], device)
        true_w = graph_data.y_freq.detach().cpu().numpy()
        true_b = np.abs(graph_data.y_amp.detach().cpu().numpy())

        with torch.no_grad():
            pred_v1 = v1_model(graph_data)
            pred_v2 = v2_model(graph_data)

        v1_dec = decode_peak_set(
            pred_v1,
            prob_threshold=args.prob_threshold,
            fallback_top_k=args.fallback_topk,
        )
        v2_dec = decode_peak_set(
            pred_v2,
            prob_threshold=args.prob_threshold,
            fallback_top_k=args.fallback_topk,
        )
        hyb_w, hyb_b, _ = combine_two_tower_predictions(
            pred_v1,
            pred_v2,
            prob_threshold=args.prob_threshold,
            fallback_top_k=args.fallback_topk,
            allow_amp_overflow=not args.disable_hybrid_amp_overflow,
            min_freq_separation=args.hybrid_min_freq_separation,
        )

        rows = [
            metric_row("V1", v1_dec["freq"], v1_dec["amp"], true_w, true_b),
            metric_row("V2", v2_dec["freq"], v2_dec["amp"], true_w, true_b),
            metric_row("Hybrid", hyb_w, hyb_b, true_w, true_b),
        ]

        print(f"\nSample: {sample_name}")
        for row in rows:
            print(format_row(row))
            all_rows[row["model"]].append(row)

    print("\nAverages")
    for model_name in ["V1", "V2", "Hybrid"]:
        rows = all_rows[model_name]
        avg_freq = np.nanmean([r["freq_mae"] for r in rows])
        avg_amp = np.nanmean([r["amp_mae"] for r in rows])
        avg_overlap = np.nanmean([r["overlap"] for r in rows])
        print(
            f"{model_name:<18} "
            f"freq_mae={avg_freq:.5f} "
            f"amp_mae={avg_amp:.6e} "
            f"overlap={avg_overlap:.4f}"
        )


if __name__ == "__main__":
    main()
