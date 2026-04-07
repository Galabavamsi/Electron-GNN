import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mace_net import SpectralEquivariantGNN
from models.mace_net_v1 import SpectralEquivariantGNNV1
from train.dataset import SpectrumDataset
from utils.hybrid_inference import combine_two_tower_predictions, decode_peak_set
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


def metric_row(name, pred_w, pred_b, true_w, true_b):
    f_mae, a_mae = matched_mae(pred_w, pred_b, true_w, true_b)
    overlap = spectral_overlap(pred_w, pred_b, true_w, true_b)
    num_pred = int(len(pred_w))
    num_true = int(len(true_w))
    count_error = float(abs(num_pred - num_true) / max(1, num_true))
    return {
        "mode": name,
        "freq_mae": f_mae,
        "amp_mae": a_mae,
        "overlap": overlap,
        "num_pred": num_pred,
        "num_true": num_true,
        "count_error": count_error,
    }


def summarize(rows):
    if not rows:
        return {
            "freq_mae": float("nan"),
            "amp_mae": float("nan"),
            "overlap": float("nan"),
            "count_error": float("nan"),
        }
    return {
        "freq_mae": float(np.nanmean([r["freq_mae"] for r in rows])),
        "amp_mae": float(np.nanmean([r["amp_mae"] for r in rows])),
        "overlap": float(np.nanmean([r["overlap"] for r in rows])),
        "count_error": float(np.nanmean([r["count_error"] for r in rows])),
    }


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


def load_v4_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
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


def run_v4_pipeline(
    pred_dict,
    true_w,
    true_b,
    device,
    trust_threshold,
    refine_steps,
    prob_threshold,
    fallback_topk,
    use_count_head,
):
    decoded = decode_peak_set(
        pred_dict,
        prob_threshold=prob_threshold,
        fallback_top_k=fallback_topk,
        use_count_head=use_count_head,
    )

    pred_w = np.asarray(decoded["freq"], dtype=np.float64)
    pred_b = np.asarray(decoded["amp"], dtype=np.float64)

    omega_t = torch.linspace(0.01, 5.0, 512, device=device)
    true_spec = lorentzian_spectrum(
        torch.as_tensor(true_w, dtype=torch.float32, device=device),
        torch.as_tensor(true_b, dtype=torch.float32, device=device),
        omega_t,
    )
    area_ref = torch.trapz(true_spec, omega_t)

    raw_scores = peak_set_verifier_scores(pred_w, pred_b, omega_grid=omega_t, expected_area=area_ref)
    trust_raw = raw_scores["s_kk"] + raw_scores["s_pos"] + raw_scores["s_sum"] + 0.25 * raw_scores["s_smooth"]

    refined = False
    if trust_raw > trust_threshold and pred_w.size > 0:
        out = refine_peak_set(
            torch.as_tensor(pred_w, dtype=torch.float32, device=device),
            torch.as_tensor(pred_b, dtype=torch.float32, device=device),
            omega_grid=omega_t,
            steps=refine_steps,
            lr=0.05,
        )
        pred_w = np.asarray(out["refined_freq"], dtype=np.float64)
        pred_b = np.asarray(out["refined_amp"], dtype=np.float64)
        refined = True

    return pred_w, pred_b, float(trust_raw), refined


def select_mode(hybrid_avg, v4_avg, count_weight, switch_margin):
    hybrid_score = hybrid_avg["overlap"] - count_weight * hybrid_avg["count_error"]
    v4_score = v4_avg["overlap"] - count_weight * v4_avg["count_error"]

    if np.isnan(v4_score):
        return "V3 hybrid two-tower", "V4 score unavailable; keeping V3 hybrid"

    if v4_score >= hybrid_score + switch_margin:
        return "V4 verifier+refiner", "V4 score exceeded V3 by switch margin"

    return "V3 hybrid two-tower", "V3 remains more reliable under overlap/count policy"


def main():
    parser = argparse.ArgumentParser(description="Compare V3 hybrid and V4, then write dashboard mode recommendation")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--v1_ckpt", type=str, default="checkpoints/best_model_v1.pth")
    parser.add_argument("--v2_ckpt", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--v4_ckpt", type=str, default="checkpoints/best_model_v4.pth")
    parser.add_argument("--v1_kmax", type=int, default=50)
    parser.add_argument("--v2_kmax", type=int, default=64)

    parser.add_argument("--v4_prob_threshold", type=float, default=0.70)
    parser.add_argument("--v4_fallback_topk", type=int, default=32)
    parser.add_argument("--v4_ignore_count_head", action="store_true")
    parser.add_argument("--v4_trust_threshold", type=float, default=0.20)
    parser.add_argument("--v4_refine_steps", type=int, default=120)

    parser.add_argument("--count_weight", type=float, default=0.10)
    parser.add_argument("--switch_margin", type=float, default=0.01)
    parser.add_argument("--out_json", type=str, default="results/model_selection.json")
    args = parser.parse_args()

    required = [args.v1_ckpt, args.v2_ckpt]
    for ckpt in required:
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SpectrumDataset(args.data_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No processed dataset files found in {args.data_dir}")

    v1_model = load_v1_model(args.v1_ckpt, device, args.v1_kmax)
    v2_model = load_v2_model(args.v2_ckpt, device, args.v2_kmax)

    v4_model = None
    if os.path.exists(args.v4_ckpt):
        v4_model = load_v4_model(args.v4_ckpt, device)
    else:
        print(f"Warning: V4 checkpoint not found ({args.v4_ckpt}); recommendation will favor V3.")

    rows_hybrid = []
    rows_v4 = []
    sample_rows = []

    use_count_head = not args.v4_ignore_count_head
    for i, file_path in enumerate(dataset.data_files):
        sample_name = os.path.basename(file_path).replace("_targets.pt", "")
        graph_data = prepare_graph(dataset[i], device)
        true_w = graph_data.y_freq.detach().cpu().numpy()
        true_b = np.abs(graph_data.y_amp.detach().cpu().numpy())

        with torch.no_grad():
            pred_v1 = v1_model(graph_data)
            pred_v2 = v2_model(graph_data)

        hyb_w, hyb_b, _ = combine_two_tower_predictions(
            pred_v1,
            pred_v2,
            prob_threshold=0.65,
            fallback_top_k=5,
        )
        row_hyb = metric_row("V3 hybrid two-tower", np.asarray(hyb_w), np.asarray(hyb_b), true_w, true_b)
        rows_hybrid.append(row_hyb)

        row_v4 = None
        trust_raw = None
        did_refine = False
        if v4_model is not None:
            with torch.no_grad():
                pred_v4 = v4_model(graph_data)
            v4_w, v4_b, trust_raw, did_refine = run_v4_pipeline(
                pred_v4,
                true_w,
                true_b,
                device,
                trust_threshold=args.v4_trust_threshold,
                refine_steps=args.v4_refine_steps,
                prob_threshold=args.v4_prob_threshold,
                fallback_topk=args.v4_fallback_topk,
                use_count_head=use_count_head,
            )
            row_v4 = metric_row("V4 verifier+refiner", v4_w, v4_b, true_w, true_b)
            rows_v4.append(row_v4)

        sample_rows.append(
            {
                "sample": sample_name,
                "hybrid": row_hyb,
                "v4": row_v4,
                "v4_trust_raw": trust_raw,
                "v4_refined": bool(did_refine),
            }
        )

    avg_hybrid = summarize(rows_hybrid)
    avg_v4 = summarize(rows_v4)

    recommended_mode, reason = select_mode(
        avg_hybrid,
        avg_v4,
        count_weight=args.count_weight,
        switch_margin=args.switch_margin,
    )

    print("\nAverages")
    print(
        "V3 hybrid two-tower "
        f"freq_mae={avg_hybrid['freq_mae']:.5f} "
        f"amp_mae={avg_hybrid['amp_mae']:.6e} "
        f"overlap={avg_hybrid['overlap']:.4f} "
        f"count_err={avg_hybrid['count_error']:.4f}"
    )
    if rows_v4:
        print(
            "V4 verifier+refiner "
            f"freq_mae={avg_v4['freq_mae']:.5f} "
            f"amp_mae={avg_v4['amp_mae']:.6e} "
            f"overlap={avg_v4['overlap']:.4f} "
            f"count_err={avg_v4['count_error']:.4f}"
        )
    print(f"Recommended dashboard mode: {recommended_mode}")
    print(f"Reason: {reason}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": args.data_dir,
        "recommended_mode": recommended_mode,
        "reason": reason,
        "selection_policy": {
            "count_weight": args.count_weight,
            "switch_margin": args.switch_margin,
        },
        "v4_decode": {
            "trust_threshold": args.v4_trust_threshold,
            "prob_threshold": args.v4_prob_threshold,
            "fallback_topk": args.v4_fallback_topk,
            "use_count_head": use_count_head,
            "refine_steps": args.v4_refine_steps,
        },
        "averages": {
            "V3 hybrid two-tower": avg_hybrid,
            "V4 verifier+refiner": avg_v4,
        },
        "samples": sample_rows,
    }

    out_json = args.out_json
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote recommendation: {out_json}")


if __name__ == "__main__":
    main()
