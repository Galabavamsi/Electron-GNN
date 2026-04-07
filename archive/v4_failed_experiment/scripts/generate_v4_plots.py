import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mace_net import SpectralEquivariantGNN
from train.dataset import SpectrumDataset
from utils.hybrid_inference import decode_peak_set
from utils.physics_verifiers import lorentzian_spectrum, peak_set_verifier_scores
from utils.rational_refiner import refine_peak_set


def parse_v4_log(log_path):
    rows = {}
    current_epoch = None

    if not os.path.exists(log_path):
        return {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m_ep = re.search(r"Epoch (\d+)/(\d+)", line)
            if m_ep:
                current_epoch = int(m_ep.group(1))
                rows.setdefault(current_epoch, {})
                continue

            if current_epoch is None:
                continue

            m_tr = re.search(
                r"Train total=([\d.]+) bip=([\d.]+) spec=([\d.]+) phys=([\d.]+) trust=([\d.]+)",
                line,
            )
            if m_tr:
                rows[current_epoch].update(
                    {
                        "train_total": float(m_tr.group(1)),
                        "train_bip": float(m_tr.group(2)),
                        "train_spec": float(m_tr.group(3)),
                        "train_phys": float(m_tr.group(4)),
                        "train_trust": float(m_tr.group(5)),
                    }
                )
                continue

            m_va = re.search(
                r"Val\s+total=([\d.]+) bip=([\d.]+) spec=([\d.]+) phys=([\d.]+) trust=([\d.]+)",
                line,
            )
            if m_va:
                rows[current_epoch].update(
                    {
                        "val_total": float(m_va.group(1)),
                        "val_bip": float(m_va.group(2)),
                        "val_spec": float(m_va.group(3)),
                        "val_phys": float(m_va.group(4)),
                        "val_trust": float(m_va.group(5)),
                    }
                )
                continue

            m_vv = re.search(
                r"Val verifier: kk=([\d.]+) pos=([\d.]+) sum=([\d.]+) smooth=([\d.]+) stab=([\d.]+)",
                line,
            )
            if m_vv:
                rows[current_epoch].update(
                    {
                        "val_kk": float(m_vv.group(1)),
                        "val_pos": float(m_vv.group(2)),
                        "val_sum": float(m_vv.group(3)),
                        "val_smooth": float(m_vv.group(4)),
                        "val_stab": float(m_vv.group(5)),
                    }
                )

    return rows


def plot_v4_training(log_path, out_dir):
    rows = parse_v4_log(log_path)
    if not rows:
        print(f"No V4 log data found at {log_path}")
        return None

    epochs = sorted(rows.keys())
    data = {
        "epoch": epochs,
        "train_total": [rows[e].get("train_total", np.nan) for e in epochs],
        "val_total": [rows[e].get("val_total", np.nan) for e in epochs],
        "val_bip": [rows[e].get("val_bip", np.nan) for e in epochs],
        "val_phys": [rows[e].get("val_phys", np.nan) for e in epochs],
        "val_trust": [rows[e].get("val_trust", np.nan) for e in epochs],
        "val_kk": [rows[e].get("val_kk", np.nan) for e in epochs],
        "val_pos": [rows[e].get("val_pos", np.nan) for e in epochs],
        "val_sum": [rows[e].get("val_sum", np.nan) for e in epochs],
        "val_smooth": [rows[e].get("val_smooth", np.nan) for e in epochs],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    axes[0].plot(data["epoch"], data["train_total"], label="train total", color="#c9973a")
    axes[0].plot(data["epoch"], data["val_total"], label="val total", color="#f0c674", linestyle="--")
    axes[0].plot(data["epoch"], data["val_bip"], label="val bip", color="#9dbad5")
    axes[0].plot(data["epoch"], data["val_phys"], label="val physics", color="#d88b8b")
    axes[0].set_title("V4 Training Losses")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.25)

    axes[1].plot(data["epoch"], data["val_trust"], label="val trust", color="#c9973a")
    axes[1].plot(data["epoch"], data["val_kk"], label="kk", color="#9dbad5")
    axes[1].plot(data["epoch"], data["val_pos"], label="pos", color="#6b8fb3")
    axes[1].plot(data["epoch"], data["val_sum"], label="sum", color="#d88b8b")
    axes[1].plot(data["epoch"], data["val_smooth"], label="smooth", color="#7fa678")
    axes[1].set_title("V4 Verifier Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.25)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "v4_training_curves.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")

    csv_path = os.path.join(out_dir, "v4_training_metrics.csv")
    pd.DataFrame(data).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    return data


def _prepare_graph(dataset, idx, device):
    graph_data = dataset[idx]
    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
    graph_data.y_freq_batch = torch.zeros(graph_data.y_freq.shape[0], dtype=torch.long)
    true_w = graph_data.y_freq.detach().cpu().numpy()
    true_b = np.abs(graph_data.y_amp.detach().cpu().numpy())
    return graph_data.to(device), true_w, true_b


def _spectrum_from_peaks(freqs, amps, omega_grid, gamma=0.015):
    spec = np.zeros_like(omega_grid)
    for w_k, b_k in zip(freqs, amps):
        spec += b_k * (gamma / ((omega_grid - w_k) ** 2 + gamma**2))
    return spec


def _trust(scores):
    return scores["s_kk"] + scores["s_pos"] + scores["s_sum"] + 0.25 * scores["s_smooth"]


def plot_v4_predictions(
    data_dir,
    ckpt,
    out_dir,
    trust_threshold=0.5,
    refine_steps=200,
    use_count_head=True,
    prob_threshold=0.65,
    fallback_topk=5,
):
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    dataset = SpectrumDataset(data_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for idx, fpath in enumerate(dataset.data_files):
        sample = os.path.basename(fpath).replace("_targets.pt", "")
        graph_data, true_w, true_b = _prepare_graph(dataset, idx, device)

        with torch.no_grad():
            pred = model(graph_data)
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
        trust_raw = _trust(raw_scores)

        ref_w = raw_w
        ref_b = raw_b
        ref_scores = raw_scores
        did_refine = False

        if trust_raw > trust_threshold and raw_w.size > 0:
            out = refine_peak_set(
                torch.as_tensor(raw_w, dtype=torch.float32, device=device),
                torch.as_tensor(raw_b, dtype=torch.float32, device=device),
                omega_grid=omega_t,
                steps=refine_steps,
                lr=0.05,
            )
            ref_w = np.asarray(out["refined_freq"], dtype=np.float64)
            ref_b = np.asarray(out["refined_amp"], dtype=np.float64)
            ref_scores = out["refined_scores"]
            did_refine = True

        trust_ref = _trust(ref_scores)

        omega = np.linspace(0.01, 5.0, 1200)
        spec_true = _spectrum_from_peaks(true_w, true_b, omega)
        spec_raw = _spectrum_from_peaks(raw_w, raw_b, omega)
        spec_ref = _spectrum_from_peaks(ref_w, ref_b, omega)

        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(omega, spec_true, color="black", linewidth=2.0, label="true")
        ax.plot(omega, spec_raw, color="#c46f3f", linestyle="--", linewidth=1.8, label="raw")
        ax.plot(omega, spec_ref, color="#4e8f70", linestyle="-.", linewidth=1.8, label="refined")
        ax.set_title(f"V4 spectrum: {sample}")
        ax.set_xlabel("Frequency (a.u.)")
        ax.set_ylabel("Intensity")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend()
        plot_path = os.path.join(out_dir, f"v4_spectrum_{sample}.png")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

        rows.append(
            {
                "sample": sample,
                "raw_peaks": int(len(raw_w)),
                "refined_peaks": int(len(ref_w)),
                "true_peaks": int(len(true_w)),
                "trust_raw": float(trust_raw),
                "trust_refined": float(trust_ref),
                "did_refine": int(did_refine),
                "raw_sum": float(raw_scores["s_sum"]),
                "ref_sum": float(ref_scores["s_sum"]),
            }
        )
        print(f"Saved {plot_path}")

    df = pd.DataFrame(rows)
    summary_csv = os.path.join(out_dir, "v4_prediction_summary.csv")
    df.to_csv(summary_csv, index=False)
    print(f"Saved {summary_csv}")

    if not df.empty:
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        x = np.arange(len(df))
        ax.bar(x - 0.15, df["trust_raw"], width=0.3, label="raw trust", color="#c46f3f")
        ax.bar(x + 0.15, df["trust_refined"], width=0.3, label="refined trust", color="#4e8f70")
        ax.set_xticks(x)
        ax.set_xticklabels(df["sample"].tolist())
        ax.set_ylabel("Trust score")
        ax.set_title("V4 trust before vs after refinement")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.legend()
        trust_path = os.path.join(out_dir, "v4_trust_comparison.png")
        fig.tight_layout()
        fig.savefig(trust_path, dpi=200)
        plt.close(fig)
        print(f"Saved {trust_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate V4 training and inference plots")
    parser.add_argument("--log", type=str, default="results/v4_train_output.log")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model_v4.pth")
    parser.add_argument("--out_dir", type=str, default="results/v4_plots")
    parser.add_argument("--trust_threshold", type=float, default=0.5)
    parser.add_argument("--prob_threshold", type=float, default=0.65)
    parser.add_argument("--fallback_topk", type=int, default=5)
    parser.add_argument("--refine_steps", type=int, default=200)
    parser.add_argument("--ignore_count_head", action="store_true")
    args = parser.parse_args()

    plot_v4_training(args.log, args.out_dir)
    plot_v4_predictions(
        args.data_dir,
        args.ckpt,
        args.out_dir,
        trust_threshold=args.trust_threshold,
        refine_steps=args.refine_steps,
        use_count_head=not args.ignore_count_head,
        prob_threshold=args.prob_threshold,
        fallback_topk=args.fallback_topk,
    )


if __name__ == "__main__":
    main()
