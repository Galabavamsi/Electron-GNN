import argparse
import os
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mace_net import SpectralEquivariantGNN
from train.dataset import SpectrumDataset
from utils.hybrid_inference import decode_peak_set
from utils.physics_verifiers import lorentzian_spectrum, peak_set_verifier_scores
from utils.rational_refiner import refine_peak_set


def prepare_graph(graph_data, device):
    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
    graph_data.y_freq_batch = torch.zeros(graph_data.y_freq.shape[0], dtype=torch.long)
    return graph_data.to(device)


def lorentzian_spectrum_np(freqs, amps, omega_grid, gamma=0.015):
    spec = np.zeros_like(omega_grid)
    for w_k, b_k in zip(freqs, amps):
        spec += b_k * (gamma / ((omega_grid - w_k) ** 2 + gamma**2))
    return spec


def spectral_overlap(pred_w, pred_b, true_w, true_b):
    omega = np.linspace(0.01, 5.0, 1024)
    spec_p = lorentzian_spectrum_np(pred_w, pred_b, omega)
    spec_t = lorentzian_spectrum_np(true_w, true_b, omega)
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


def trust_from_scores(scores, weights):
    return (
        weights["kk"] * scores["s_kk"]
        + weights["pos"] * scores["s_pos"]
        + weights["sum"] * scores["s_sum"]
        + weights["smooth"] * scores["s_smooth"]
    )


def metric_row(name, pred_w, pred_b, true_w, true_b, trust):
    f_mae, a_mae = matched_mae(pred_w, pred_b, true_w, true_b)
    overlap = spectral_overlap(pred_w, pred_b, true_w, true_b)
    return {
        "mode": name,
        "freq_mae": f_mae,
        "amp_mae": a_mae,
        "overlap": overlap,
        "trust": float(trust),
        "num_pred": int(len(pred_w)),
        "num_true": int(len(true_w)),
    }


def format_row(row):
    return (
        f"{row['mode']:<10} "
        f"freq_mae={row['freq_mae']:.5f} "
        f"amp_mae={row['amp_mae']:.6e} "
        f"overlap={row['overlap']:.4f} "
        f"trust={row['trust']:.4f} "
        f"pred={row['num_pred']:>3d} true={row['num_true']:>3d}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate V4 trust-gated refinement")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model_v4.pth")

    parser.add_argument("--k_max", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--amp_scale", type=float, default=1e-3)

    parser.add_argument("--prob_threshold", type=float, default=0.65)
    parser.add_argument("--fallback_topk", type=int, default=5)
    parser.add_argument("--ignore_count_head", action="store_true")
    parser.add_argument("--trust_threshold", type=float, default=0.5)

    parser.add_argument("--w_kk", type=float, default=1.0)
    parser.add_argument("--w_pos", type=float, default=1.0)
    parser.add_argument("--w_sum", type=float, default=1.0)
    parser.add_argument("--w_smooth", type=float, default=0.25)

    parser.add_argument("--refine_steps", type=int, default=200)
    parser.add_argument("--refine_lr", type=float, default=0.05)
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SpectralEquivariantGNN(
        node_features_in=5,
        hidden_dim=args.hidden_dim,
        K_max=args.k_max,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        amp_scale=args.amp_scale,
    ).to(device)
    state_dict = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dataset = SpectrumDataset(args.data_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No processed dataset files found in {args.data_dir}")

    weights = {
        "kk": args.w_kk,
        "pos": args.w_pos,
        "sum": args.w_sum,
        "smooth": args.w_smooth,
    }

    all_raw = []
    all_refined = []
    refined_count = 0

    for i, file_path in enumerate(dataset.data_files):
        sample_name = os.path.basename(file_path).replace("_targets.pt", "")
        graph_data = prepare_graph(dataset[i], device)
        true_w = graph_data.y_freq.detach().cpu().numpy()
        true_b = np.abs(graph_data.y_amp.detach().cpu().numpy())

        omega = torch.linspace(0.01, 5.0, 512, device=device)
        true_spec = lorentzian_spectrum(
            torch.as_tensor(true_w, dtype=torch.float32, device=device),
            torch.as_tensor(true_b, dtype=torch.float32, device=device),
            omega,
        )
        area_ref = torch.trapz(true_spec, omega)

        with torch.no_grad():
            pred_dict = model(graph_data)

        decoded = decode_peak_set(
            pred_dict,
            prob_threshold=args.prob_threshold,
            fallback_top_k=args.fallback_topk,
            use_count_head=not args.ignore_count_head,
        )
        raw_w = np.asarray(decoded["freq"], dtype=np.float64)
        raw_b = np.asarray(decoded["amp"], dtype=np.float64)

        raw_scores = peak_set_verifier_scores(
            raw_w,
            raw_b,
            omega_grid=omega,
            expected_area=area_ref,
        )
        raw_trust = trust_from_scores(raw_scores, weights)

        refined_w = raw_w
        refined_b = raw_b
        refined_scores = raw_scores

        if raw_trust > args.trust_threshold and raw_w.size > 0:
            refined_count += 1
            ref = refine_peak_set(
                torch.as_tensor(raw_w, dtype=torch.float32, device=device),
                torch.as_tensor(raw_b, dtype=torch.float32, device=device),
                omega_grid=omega,
                steps=args.refine_steps,
                lr=args.refine_lr,
            )
            refined_w = np.asarray(ref["refined_freq"], dtype=np.float64)
            refined_b = np.asarray(ref["refined_amp"], dtype=np.float64)
            refined_scores = ref["refined_scores"]

        refined_trust = trust_from_scores(refined_scores, weights)

        raw_row = metric_row("raw", raw_w, raw_b, true_w, true_b, raw_trust)
        ref_row = metric_row("refined", refined_w, refined_b, true_w, true_b, refined_trust)
        all_raw.append(raw_row)
        all_refined.append(ref_row)

        print(f"\nSample: {sample_name}")
        print(format_row(raw_row))
        print(format_row(ref_row))

    def summarize(rows, key):
        return np.nanmean([r[key] for r in rows])

    print("\nAverages")
    print(
        "raw      "
        f"freq_mae={summarize(all_raw, 'freq_mae'):.5f} "
        f"amp_mae={summarize(all_raw, 'amp_mae'):.6e} "
        f"overlap={summarize(all_raw, 'overlap'):.4f} "
        f"trust={summarize(all_raw, 'trust'):.4f}"
    )
    print(
        "refined  "
        f"freq_mae={summarize(all_refined, 'freq_mae'):.5f} "
        f"amp_mae={summarize(all_refined, 'amp_mae'):.6e} "
        f"overlap={summarize(all_refined, 'overlap'):.4f} "
        f"trust={summarize(all_refined, 'trust'):.4f}"
    )
    print(f"Refiner triggered on {refined_count}/{len(dataset)} samples")


if __name__ == "__main__":
    main()
