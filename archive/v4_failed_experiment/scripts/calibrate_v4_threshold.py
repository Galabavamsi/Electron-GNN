import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mace_net import SpectralEquivariantGNN
from train.dataset import SpectrumDataset
from utils.hybrid_inference import decode_peak_set
from utils.physics_verifiers import lorentzian_spectrum, peak_set_verifier_scores
from utils.rational_refiner import refine_peak_set


def _spectrum_from_peaks(freqs, amps, omega_grid, gamma=0.015):
    spec = np.zeros_like(omega_grid)
    for w_k, b_k in zip(freqs, amps):
        spec += b_k * (gamma / ((omega_grid - w_k) ** 2 + gamma**2))
    return spec


def matched_mae(pred_w, pred_b, true_w, true_b):
    if len(pred_w) == 0 or len(true_w) == 0:
        return float("nan"), float("nan")

    cost = 10.0 * np.abs(pred_w[:, None] - true_w[None, :]) + np.abs(pred_b[:, None] - true_b[None, :])
    pred_idx, true_idx = linear_sum_assignment(cost)

    f_mae = float(np.mean(np.abs(pred_w[pred_idx] - true_w[true_idx])))
    a_mae = float(np.mean(np.abs(pred_b[pred_idx] - true_b[true_idx])))
    return f_mae, a_mae


def spectral_overlap(pred_w, pred_b, true_w, true_b):
    omega = np.linspace(0.01, 5.0, 1024)
    spec_p = _spectrum_from_peaks(pred_w, pred_b, omega)
    spec_t = _spectrum_from_peaks(true_w, true_b, omega)
    denom = np.linalg.norm(spec_p) * np.linalg.norm(spec_t)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(spec_p, spec_t) / denom)


def parse_thresholds(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def build_v4_model(ckpt, device):
    state = torch.load(ckpt, map_location=device, weights_only=True)
    q = state.get("query_embed")
    k_max = int(q.shape[0]) if q is not None and q.ndim == 2 else 64
    hidden_dim = int(q.shape[1]) if q is not None and q.ndim == 2 else 128

    conv_ids = []
    for key in state.keys():
        if key.startswith("convs."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                conv_ids.append(int(parts[1]))
    num_layers = max(conv_ids) + 1 if conv_ids else 4
    num_heads = 4 if hidden_dim % 4 == 0 else 1

    model = SpectralEquivariantGNN(
        node_features_in=5,
        hidden_dim=hidden_dim,
        K_max=k_max,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def prepare_samples(model, data_dir, device, prob_threshold=0.65, fallback_topk=5, use_count_head=True):
    dataset = SpectrumDataset(data_dir)
    samples = []

    for idx, fpath in enumerate(dataset.data_files):
        graph = dataset[idx]
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
        graph.y_freq_batch = torch.zeros(graph.y_freq.shape[0], dtype=torch.long)
        graph = graph.to(device)

        true_w = graph.y_freq.detach().cpu().numpy()
        true_b = np.abs(graph.y_amp.detach().cpu().numpy())

        with torch.no_grad():
            pred = model(graph)
        dec = decode_peak_set(
            pred,
            prob_threshold=prob_threshold,
            fallback_top_k=fallback_topk,
            use_count_head=use_count_head,
        )
        raw_w = np.asarray(dec["freq"], dtype=np.float64)
        raw_b = np.asarray(dec["amp"], dtype=np.float64)

        omega_t = torch.linspace(0.01, 5.0, 512, device=device)
        true_spec_t = lorentzian_spectrum(
            torch.as_tensor(true_w, dtype=torch.float32, device=device),
            torch.as_tensor(true_b, dtype=torch.float32, device=device),
            omega_t,
        )
        area_ref = torch.trapz(true_spec_t, omega_t)

        raw_scores = peak_set_verifier_scores(raw_w, raw_b, omega_grid=omega_t, expected_area=area_ref)
        trust_raw = raw_scores["s_kk"] + raw_scores["s_pos"] + raw_scores["s_sum"] + 0.25 * raw_scores["s_smooth"]

        samples.append(
            {
                "name": os.path.basename(fpath).replace("_targets.pt", ""),
                "true_w": true_w,
                "true_b": true_b,
                "raw_w": raw_w,
                "raw_b": raw_b,
                "omega_t": omega_t,
                "trust_raw": trust_raw,
            }
        )

    return samples


def main():
    parser = argparse.ArgumentParser(description="Sweep V4 trust thresholds and report calibration metrics")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model_v4.pth")
    parser.add_argument("--thresholds", type=str, default="0.2,0.35,0.5,0.75,1.0")
    parser.add_argument("--prob_threshold", type=float, default=0.65)
    parser.add_argument("--fallback_topk", type=int, default=5)
    parser.add_argument("--refine_steps", type=int, default=120)
    parser.add_argument("--ignore_count_head", action="store_true")
    parser.add_argument("--out_csv", type=str, default="results/v4_plots/v4_threshold_sweep.csv")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_v4_model(args.ckpt, device)
    thresholds = parse_thresholds(args.thresholds)

    samples = prepare_samples(
        model,
        args.data_dir,
        device,
        prob_threshold=args.prob_threshold,
        fallback_topk=args.fallback_topk,
        use_count_head=not args.ignore_count_head,
    )
    if not samples:
        raise RuntimeError(f"No samples found in {args.data_dir}")

    rows = []
    for th in thresholds:
        all_freq = []
        all_amp = []
        all_overlap = []
        refine_count = 0

        for s in samples:
            pred_w = s["raw_w"]
            pred_b = s["raw_b"]

            if s["trust_raw"] > th and pred_w.size > 0:
                refine_count += 1
                out = refine_peak_set(
                    torch.as_tensor(pred_w, dtype=torch.float32, device=device),
                    torch.as_tensor(pred_b, dtype=torch.float32, device=device),
                    omega_grid=s["omega_t"],
                    steps=args.refine_steps,
                    lr=0.05,
                )
                pred_w = np.asarray(out["refined_freq"], dtype=np.float64)
                pred_b = np.asarray(out["refined_amp"], dtype=np.float64)

            fm, am = matched_mae(pred_w, pred_b, s["true_w"], s["true_b"])
            ov = spectral_overlap(pred_w, pred_b, s["true_w"], s["true_b"])
            all_freq.append(fm)
            all_amp.append(am)
            all_overlap.append(ov)

        rows.append(
            {
                "threshold": th,
                "trigger_rate": refine_count / len(samples),
                "freq_mae": float(np.nanmean(all_freq)),
                "amp_mae": float(np.nanmean(all_amp)),
                "overlap": float(np.nanmean(all_overlap)),
            }
        )

    df = pd.DataFrame(rows).sort_values("threshold")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(df.to_string(index=False))
    best_idx = int(df["overlap"].idxmax())
    best = df.loc[best_idx]
    print(
        "\nBest overlap threshold: "
        f"{best['threshold']:.2f} "
        f"(overlap={best['overlap']:.4f}, trigger_rate={best['trigger_rate']:.2f})"
    )


if __name__ == "__main__":
    main()
