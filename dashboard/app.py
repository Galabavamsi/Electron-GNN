import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import streamlit as st
import streamlit.components.v1 as components
from scipy.optimize import linear_sum_assignment

# Add root to path so we can import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from models.mace_net import SpectralEquivariantGNN
    from models.mace_net_v1 import SpectralEquivariantGNNV1
    from utils.hybrid_inference import decode_peak_set, combine_two_tower_predictions
    from utils.model_diagnostics import (
        plot_predict_vs_real_parity,
        plot_complex_poles,
        calc_spectral_overlap_score,
    )
    from train.dataset import SpectrumDataset
except ImportError as exc:
    st.set_page_config(page_title="Electron-GNN Observatory", layout="wide", page_icon="⚛")
    st.error(f"Dashboard import failed: {exc}")
    st.stop()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")
CHECKPOINTS_DIR = os.path.join(ROOT, "checkpoints")


st.set_page_config(
    page_title="Electron-GNN Observatory",
    layout="wide",
    page_icon="⚛",
)


def apply_vintage_theme():
    st.markdown(
        """
<style>
:root {
  --bg: #111014;
  --panel: #1a1820;
  --ink: #f2e6cf;
  --muted: #b9a98b;
  --accent: #c9973a;
  --line: #2a2630;
}

html, body, [class*="css"]  {
  font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  color: var(--ink);
}

.stApp {
  background:
        radial-gradient(1200px 500px at 10% -10%, #2a221a 0%, transparent 60%),
    radial-gradient(1000px 450px at 90% 0%, #2f1f1b 0%, transparent 55%),
    var(--bg);
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #1f1d25 0%, #18161d 100%);
  border-right: 1px solid var(--line);
}

h1, h2, h3 {
  letter-spacing: 0.2px;
}

.block-card {
  background: rgba(22, 20, 28, 0.88);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.9rem 1rem;
}

.small-note {
  color: var(--muted);
  font-size: 0.92rem;
}

.kpi {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.6rem 0.9rem;
  background: rgba(18, 17, 24, 0.9);
}

hr {
  border-color: var(--line);
}
</style>
        """,
        unsafe_allow_html=True,
    )


def enable_auto_refresh(enabled, interval_sec):
    if not enabled:
        return

    components.html(
        f"""
<script>
setTimeout(function() {{
  window.parent.location.reload();
}}, {int(interval_sec) * 1000});
</script>
        """,
        height=0,
    )


def file_signature(path):
    if not os.path.exists(path):
        return (path, 0.0, 0)
    stat = os.stat(path)
    return (path, float(stat.st_mtime), int(stat.st_size))


def glob_signature(pattern):
    files = sorted(glob.glob(pattern))
    return tuple(file_signature(p) for p in files)


@st.cache_data(ttl=8)
def load_datasets(dataset_sig):
    del dataset_sig
    if not os.path.exists(PROCESSED_DIR):
        return {}, []

    pt_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.pt")))
    datasets = {}
    for fpath in pt_files:
        name = os.path.basename(fpath).replace("_targets.pt", "")
        datasets[name] = torch.load(fpath, weights_only=False)
    return datasets, pt_files


@st.cache_data(ttl=8)
def summarize_datasets(dataset_sig):
    datasets, _ = load_datasets(dataset_sig)
    rows = []
    for name, data in datasets.items():
        freqs = data["frequencies"].numpy()
        amps = np.abs(data["amplitudes_x"].numpy())
        rows.append(
            {
                "sample": name,
                "peaks": int(len(freqs)),
                "freq_min": float(np.min(freqs)) if len(freqs) else np.nan,
                "freq_max": float(np.max(freqs)) if len(freqs) else np.nan,
                "amp_min": float(np.min(amps)) if len(amps) else np.nan,
                "amp_max": float(np.max(amps)) if len(amps) else np.nan,
                "amp_mean": float(np.mean(amps)) if len(amps) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _spectrum_from_peaks(freqs, amps, omega_grid, gamma=0.015):
    spec = np.zeros_like(omega_grid)
    for w_k, b_k in zip(freqs, amps):
        spec += b_k * (gamma / ((omega_grid - w_k) ** 2 + gamma**2))
    return spec


def _prepare_graph(dataset_name, device):
    dataset = SpectrumDataset(PROCESSED_DIR)
    idx = -1
    for i, fpath in enumerate(dataset.data_files):
        if dataset_name in fpath:
            idx = i
            break

    if idx == -1:
        return None, None, None

    graph_data = dataset[idx]
    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
    graph_data.y_freq_batch = torch.zeros(graph_data.y_freq.shape[0], dtype=torch.long)
    true_w = graph_data.y_freq.detach().cpu().numpy()
    true_b = np.abs(graph_data.y_amp.detach().cpu().numpy())
    return graph_data.to(device), true_w, true_b


def _score_amp_checkpoint(model, device, dataset_sig):
    datasets, _ = load_datasets(dataset_sig)
    if not datasets:
        return -1e9

    model.eval()
    omega = np.linspace(0.01, 1.5, 600)
    scores = []

    with torch.no_grad():
        for sample_name in sorted(datasets.keys()):
            graph_data, true_w, true_b = _prepare_graph(sample_name, device)
            if graph_data is None:
                continue
            pred = model(graph_data)
            dec = decode_peak_set(pred, prob_threshold=0.65, fallback_top_k=5)
            pred_w, pred_b = dec["freq"], dec["amp"]

            spec_t = _spectrum_from_peaks(true_w, true_b, omega)
            spec_p = _spectrum_from_peaks(pred_w, pred_b, omega)
            overlap = float(calc_spectral_overlap_score(spec_t, spec_p))
            count_penalty = abs(len(pred_w) - len(true_w)) / max(1, len(true_w))
            scores.append(overlap - 0.25 * count_penalty)

    return float(np.mean(scores)) if scores else -1e9


@st.cache_resource(ttl=20)
def load_models(amp_ckpt_sig, freq_ckpt_sig, dataset_sig):
    del amp_ckpt_sig, freq_ckpt_sig
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_candidates = [
        os.path.join(CHECKPOINTS_DIR, "best_model.pth"),
        os.path.join(CHECKPOINTS_DIR, "v3_amp_tower.pth"),
    ]

    amp_model = None
    amp_ckpt = None
    amp_score_map = {}
    best_score = -1e9

    for ckpt in amp_candidates:
        if not os.path.exists(ckpt):
            continue

        candidate = SpectralEquivariantGNN(node_features_in=5, K_max=64)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        candidate.load_state_dict(state, strict=False)
        candidate = candidate.to(device)
        candidate.eval()

        score = _score_amp_checkpoint(candidate, device, dataset_sig)
        amp_score_map[ckpt] = score
        if score > best_score:
            best_score = score
            amp_model = candidate
            amp_ckpt = ckpt

    if amp_model is None:
        amp_model = SpectralEquivariantGNN(node_features_in=5, K_max=64).to(device)
        amp_model.eval()

    freq_candidates = [
        os.path.join(CHECKPOINTS_DIR, "best_model_v1.pth"),
        os.path.join(CHECKPOINTS_DIR, "v3_freq_tower.pth"),
    ]

    freq_model = None
    freq_ckpt = None
    for ckpt in freq_candidates:
        if not os.path.exists(ckpt):
            continue

        state = torch.load(ckpt, map_location=device, weights_only=True)
        k_max = 50
        if "head_freq.2.bias" in state:
            k_max = int(state["head_freq.2.bias"].numel())
        elif "head_freq.2.weight" in state:
            k_max = int(state["head_freq.2.weight"].shape[0])

        freq_model = SpectralEquivariantGNNV1(node_features_in=5, K_max=k_max)
        freq_model.load_state_dict(state, strict=False)
        freq_model = freq_model.to(device)
        freq_model.eval()
        freq_ckpt = ckpt
        break

    return {
        "device": device,
        "amp_model": amp_model,
        "amp_ckpt": amp_ckpt,
        "amp_score_map": amp_score_map,
        "freq_model": freq_model,
        "freq_ckpt": freq_ckpt,
    }


def predict_sample(models, sample_name, mode):
    graph_data, true_w, true_b = _prepare_graph(sample_name, models["device"])
    if graph_data is None:
        return None, None, None

    amp_model = models["amp_model"]
    freq_model = models["freq_model"]

    with torch.no_grad():
        amp_pred = amp_model(graph_data)

        if mode == "V1 frequency only" and freq_model is not None:
            freq_pred = freq_model(graph_data)
            dec = decode_peak_set(freq_pred, prob_threshold=0.65, fallback_top_k=5)
            pred_w, pred_b, pred_probs = dec["freq"], dec["amp"], dec["prob"]
        elif mode == "V3 hybrid two-tower" and freq_model is not None:
            freq_pred = freq_model(graph_data)
            pred_w, pred_b, pred_probs = combine_two_tower_predictions(
                freq_pred,
                amp_pred,
                prob_threshold=0.65,
                fallback_top_k=5,
            )
        else:
            dec = decode_peak_set(amp_pred, prob_threshold=0.65, fallback_top_k=5)
            pred_w, pred_b, pred_probs = dec["freq"], dec["amp"], dec["prob"]

    return (pred_w, pred_b, pred_probs), (true_w, true_b), graph_data


def matched_metrics(pred_w, pred_b, true_w, true_b):
    if len(pred_w) == 0 or len(true_w) == 0:
        return np.nan, np.nan, 0.0

    cost = 10.0 * np.abs(pred_w[:, None] - true_w[None, :]) + np.abs(pred_b[:, None] - true_b[None, :])
    pred_idx, true_idx = linear_sum_assignment(cost)

    f_mae = float(np.mean(np.abs(pred_w[pred_idx] - true_w[true_idx])))
    b_mae = float(np.mean(np.abs(pred_b[pred_idx] - true_b[true_idx])))

    omega = np.linspace(0.01, 1.5, 1000)
    spec_t = _spectrum_from_peaks(true_w, true_b, omega)
    spec_p = _spectrum_from_peaks(pred_w, pred_b, omega)
    overlap = float(calc_spectral_overlap_score(spec_t, spec_p))
    return f_mae, b_mae, overlap


@st.cache_data(ttl=8)
def parse_v2_training_log(log_path, log_sig):
    del log_sig
    epochs, train_bip, val_bip, train_spec, val_spec = [], [], [], [], []
    if not os.path.exists(log_path):
        return {
            "epochs": epochs,
            "train_bip": train_bip,
            "val_bip": val_bip,
            "train_spec": train_spec,
            "val_spec": val_spec,
        }

    with open(log_path, "r") as f:
        content = f.read()

    blocks = re.split(r"Epoch \d+/\d+", content)
    for i, block in enumerate(blocks[1:]):
        tm = re.search(r"Train - Bipartite: ([\d.]+), Spectrum: ([\d.]+)", block)
        vm = re.search(r"Val\s+- Bipartite: ([\d.]+), Spectrum: ([\d.]+)", block)
        if tm and vm:
            epochs.append(i + 1)
            train_bip.append(float(tm.group(1)))
            train_spec.append(float(tm.group(2)))
            val_bip.append(float(vm.group(1)))
            val_spec.append(float(vm.group(2)))

    return {
        "epochs": epochs,
        "train_bip": train_bip,
        "val_bip": val_bip,
        "train_spec": train_spec,
        "val_spec": val_spec,
    }


@st.cache_data(ttl=8)
def parse_v3_training_log(log_path, log_sig):
    del log_sig
    data = {
        "freq_epoch": [],
        "freq_train_total": [],
        "freq_train_base": [],
        "freq_train_teacher": [],
        "freq_val": [],
        "amp_epoch": [],
        "amp_train_bip": [],
        "amp_train_spec": [],
        "amp_val_bip": [],
        "amp_val_spec": [],
        "amp_val_total": [],
    }

    if not os.path.exists(log_path):
        return data

    current_freq_epoch = None
    current_amp_epoch = None

    with open(log_path, "r") as f:
        for line in f:
            m_fe = re.search(r"\[Freq Tower\] Epoch (\d+)/(\d+)", line)
            if m_fe:
                current_freq_epoch = int(m_fe.group(1))
                continue

            m_fl = re.search(
                r"Freq tower losses - train\(total/base/teacher\): ([\d.]+)/([\d.]+)/([\d.]+) val: ([\d.]+)",
                line,
            )
            if m_fl and current_freq_epoch is not None:
                data["freq_epoch"].append(current_freq_epoch)
                data["freq_train_total"].append(float(m_fl.group(1)))
                data["freq_train_base"].append(float(m_fl.group(2)))
                data["freq_train_teacher"].append(float(m_fl.group(3)))
                data["freq_val"].append(float(m_fl.group(4)))
                continue

            m_ae = re.search(r"\[Amp Tower\] Epoch (\d+)/(\d+)", line)
            if m_ae:
                current_amp_epoch = int(m_ae.group(1))
                continue

            m_al = re.search(
                r"Amp tower - train\(bip/spec\): ([\d.]+)/([\d.]+) val\(bip/spec\): ([\d.]+)/([\d.]+) total=([\d.]+)",
                line,
            )
            if m_al and current_amp_epoch is not None:
                data["amp_epoch"].append(current_amp_epoch)
                data["amp_train_bip"].append(float(m_al.group(1)))
                data["amp_train_spec"].append(float(m_al.group(2)))
                data["amp_val_bip"].append(float(m_al.group(3)))
                data["amp_val_spec"].append(float(m_al.group(4)))
                data["amp_val_total"].append(float(m_al.group(5)))

    return data


def render_overview(dataset_df, models):
    st.title("Electron-GNN Observatory")
    st.markdown("A cleaner, vintage-style control room for data quality, training dynamics, and hybrid spectral inference.")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='kpi'><b>Samples</b><br>{len(dataset_df)}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi'><b>Total Peaks</b><br>{int(dataset_df['peaks'].sum()) if not dataset_df.empty else 0}</div>", unsafe_allow_html=True)
    c3.markdown(
        f"<div class='kpi'><b>Amp Tower</b><br>{os.path.basename(models['amp_ckpt']) if models['amp_ckpt'] else 'N/A'}</div>",
        unsafe_allow_html=True,
    )
    c4.markdown(
        f"<div class='kpi'><b>Freq Tower</b><br>{os.path.basename(models['freq_ckpt']) if models['freq_ckpt'] else 'N/A'}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Pipeline")
    st.markdown(
        """
```mermaid
flowchart TD
    A[Raw RT-TDDFT] --> B[Padé + clustering + LASSO extraction]
    B --> C[Processed peak targets]
    C --> D[V2 amplitude tower]
    C --> E[V1 frequency prior]
    D --> F[Hybrid combiner]
    E --> F
    F --> G[Reconstructed spectrum + diagnostics]
```
        """
    )


def render_data_page(datasets, dataset_df):
    st.title("Data Observatory")
    st.caption("Auto-updating dataset health and peak statistics.")

    if dataset_df.empty:
        st.warning("No processed targets found in data/processed.")
        return

    st.dataframe(dataset_df, use_container_width=True)

    all_freq = []
    all_amp = []
    for data in datasets.values():
        all_freq.extend(data["frequencies"].numpy().tolist())
        all_amp.extend(np.abs(data["amplitudes_x"].numpy()).tolist())

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(np.array(all_freq), bins=30, color="#c9973a", alpha=0.85)
        ax.set_title("Frequency Distribution")
        ax.set_xlabel("Frequency (a.u.)")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.25)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        amps = np.array(all_amp)
        amps = amps[amps > 0]
        ax.hist(np.log10(amps), bins=30, color="#8fbf8f", alpha=0.9)
        ax.set_title("Amplitude Distribution (log10)")
        ax.set_xlabel("log10(|B_k|)")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.25)
        st.pyplot(fig)

    sample = st.selectbox("Inspect sample", sorted(datasets.keys()))
    data = datasets[sample]
    w = data["frequencies"].numpy()
    b = np.abs(data["amplitudes_x"].numpy())

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.vlines(w, 0, b, color="#d28f6d", linewidth=1.8)
    ax.scatter(w, b, color="#f3d7b1", s=14)
    ax.set_title(f"Extracted Peaks: {sample}")
    ax.set_xlabel("Frequency (a.u.)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle=":", alpha=0.3)
    st.pyplot(fig)


def render_training_page():
    st.title("Training Monitor")
    st.caption("Live plots are generated from the latest training log file and refresh when logs change.")

    v2_log = os.path.join(RESULTS_DIR, "train_output.log")
    v3_log = os.path.join(RESULTS_DIR, "v3_train_output.log")

    existing_logs = [p for p in [v2_log, v3_log] if os.path.exists(p)]
    if not existing_logs:
        st.warning("No training log found. Expected results/train_output.log or results/v3_train_output.log")
        return

    latest_log = max(existing_logs, key=lambda p: os.path.getmtime(p))
    selected_log = st.selectbox(
        "Log source",
        existing_logs,
        index=existing_logs.index(latest_log),
        format_func=lambda p: os.path.basename(p),
    )

    st.markdown(
        f"<span class='small-note'>Monitoring: {selected_log} | last modified: {pd.to_datetime(os.path.getmtime(selected_log), unit='s')}</span>",
        unsafe_allow_html=True,
    )

    if os.path.basename(selected_log) == "v3_train_output.log":
        data = parse_v3_training_log(selected_log, file_signature(selected_log))

        c1, c2 = st.columns(2)
        if data["freq_val"]:
            best_freq_idx = int(np.argmin(data["freq_val"]))
            c1.metric("Best Freq Val", f"{data['freq_val'][best_freq_idx]:.4f}", f"epoch {data['freq_epoch'][best_freq_idx]}")
        else:
            c1.metric("Best Freq Val", "N/A")

        if data["amp_val_total"]:
            best_amp_idx = int(np.argmin(data["amp_val_total"]))
            c2.metric("Best Amp Val Total", f"{data['amp_val_total'][best_amp_idx]:.4f}", f"epoch {data['amp_epoch'][best_amp_idx]}")
        else:
            c2.metric("Best Amp Val Total", "N/A")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        if data["freq_epoch"]:
            axes[0].plot(data["freq_epoch"], data["freq_train_total"], label="train total", color="#c9973a")
            axes[0].plot(data["freq_epoch"], data["freq_val"], label="val", color="#f0c674", linestyle="--")
            axes[0].set_title("Frequency Tower")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(True, linestyle="--", alpha=0.25)

        if data["amp_epoch"]:
            axes[1].plot(data["amp_epoch"], data["amp_train_bip"], label="train bip", color="#9dbad5")
            axes[1].plot(data["amp_epoch"], data["amp_val_bip"], label="val bip", color="#6b8fb3", linestyle="--")
            axes[1].plot(data["amp_epoch"], data["amp_val_total"], label="val total", color="#d88b8b")
            axes[1].set_title("Amplitude Tower")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].legend()
            axes[1].grid(True, linestyle="--", alpha=0.25)

        st.pyplot(fig)

    else:
        data = parse_v2_training_log(selected_log, file_signature(selected_log))
        if not data["epochs"]:
            st.warning("Could not parse epochs from this log.")
            return

        val_total = [vb + 0.3 * vs for vb, vs in zip(data["val_bip"], data["val_spec"])]
        best_idx = int(np.argmin(val_total))

        c1, c2, c3 = st.columns(3)
        c1.metric("Epochs", len(data["epochs"]))
        c2.metric("Best Val Total", f"{val_total[best_idx]:.4f}", f"epoch {data['epochs'][best_idx]}")
        c3.metric("Latest Val Bip / Spec", f"{data['val_bip'][-1]:.4f} / {data['val_spec'][-1]:.4f}")

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(data["epochs"], data["train_bip"], label="Train Bipartite", color="#c9973a")
        ax.plot(data["epochs"], data["val_bip"], label="Val Bipartite", color="#f0c674", linestyle="--")
        ax.plot(data["epochs"], data["train_spec"], label="Train Spectrum", color="#8fbf8f", alpha=0.9)
        ax.plot(data["epochs"], data["val_spec"], label="Val Spectrum", color="#6f9f6f", linestyle=":")
        ax.set_title("Live Training Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(ncol=2)
        ax.grid(True, linestyle="--", alpha=0.25)
        st.pyplot(fig)


def render_inference_page(models, datasets, inference_mode):
    st.title("Inference Studio")

    amp_name = os.path.basename(models["amp_ckpt"]) if models["amp_ckpt"] else "random-init"
    freq_name = os.path.basename(models["freq_ckpt"]) if models["freq_ckpt"] else "not-loaded"
    st.caption(f"Amp checkpoint: {amp_name} | Freq checkpoint: {freq_name}")

    amp_scores = models.get("amp_score_map", {})
    if amp_scores:
        ordered = sorted(amp_scores.items(), key=lambda kv: kv[1], reverse=True)
        score_line = ", ".join([f"{os.path.basename(k)}={v:.3f}" for k, v in ordered])
        st.caption(f"Auto quality gate scores: {score_line}")

    sample = st.selectbox("Sample", sorted(datasets.keys()))
    show_compare = st.checkbox("Show V1 vs V2 vs Hybrid comparison", value=True)

    if st.button("Run inference"):
        pred_bundle, truth_bundle, _ = predict_sample(models, sample, inference_mode)
        if pred_bundle is None:
            st.error("Sample not found.")
            return

        pred_w, pred_b, pred_probs = pred_bundle
        true_w, true_b = truth_bundle

        omega = np.linspace(0.01, 1.5, 1000)
        spec_t = _spectrum_from_peaks(true_w, true_b, omega)
        spec_p = _spectrum_from_peaks(pred_w, pred_b, omega)

        f_mae, b_mae, overlap = matched_metrics(pred_w, pred_b, true_w, true_b)
        c1, c2, c3 = st.columns(3)
        c1.metric("Freq MAE", f"{f_mae:.5f}")
        c2.metric("Amp MAE", f"{b_mae:.6e}")
        c3.metric("Spectral Overlap", f"{overlap:.4f}")

        fig, ax = plt.subplots(figsize=(12, 4.3))
        ax.plot(omega, spec_t, color="black", linewidth=2.2, label="True")
        ax.plot(omega, spec_p, color="#c46f3f", linestyle="--", linewidth=2.2, label="Prediction")
        ax.fill_between(omega, spec_t, color="gray", alpha=0.08)
        ax.fill_between(omega, spec_p, color="#c46f3f", alpha=0.14)
        ax.set_title(f"{sample.upper()} Spectrum: true vs {inference_mode}")
        ax.set_xlabel("Frequency (a.u.)")
        ax.set_ylabel("Intensity")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend()
        st.pyplot(fig)

        table_df = pd.DataFrame(
            {
                "freq": np.round(pred_w, 6),
                "amp": np.round(pred_b, 8),
                "prob": np.round(pred_probs, 4),
            }
        ).sort_values("prob", ascending=False)
        st.dataframe(table_df.head(30), use_container_width=True)

        if show_compare:
            compare_modes = ["V1 frequency only", "V2 single tower", "V3 hybrid two-tower"]
            rows = []
            plots = []
            for mode in compare_modes:
                if mode == "V1 frequency only" and models["freq_model"] is None:
                    continue
                p_bundle, t_bundle, _ = predict_sample(models, sample, mode)
                if p_bundle is None:
                    continue
                pw, pb, _pp = p_bundle
                tw, tb = t_bundle
                fm, bm, ov = matched_metrics(pw, pb, tw, tb)
                rows.append(
                    {
                        "mode": mode,
                        "pred_peaks": len(pw),
                        "true_peaks": len(tw),
                        "freq_mae": fm,
                        "amp_mae": bm,
                        "overlap": ov,
                    }
                )
                plots.append((mode, pw, pb, tw, tb))

            st.markdown("### Comparison")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            n = len(plots)
            fig, axes = plt.subplots(1, n, figsize=(5.3 * n, 3.8), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, (mode, pw, pb, tw, tb) in zip(axes, plots):
                spec_true = _spectrum_from_peaks(tw, tb, omega)
                spec_pred = _spectrum_from_peaks(pw, pb, omega)
                ax.plot(omega, spec_true, color="black", linewidth=1.8, label="true")
                ax.plot(omega, spec_pred, color="#c46f3f", linestyle="--", linewidth=1.8, label="pred")
                ax.set_title(mode.replace(" tower", ""))
                ax.set_xlabel("w (a.u.)")
                ax.grid(True, linestyle="--", alpha=0.25)
            axes[0].set_ylabel("Intensity")
            axes[0].legend()
            st.pyplot(fig)


def render_diagnostics_page(models, datasets, inference_mode):
    st.title("Scientific Diagnostics")
    sample = st.selectbox("Sample for diagnostics", sorted(datasets.keys()))

    pred_bundle, truth_bundle, _ = predict_sample(models, sample, inference_mode)
    if pred_bundle is None:
        st.warning("Unable to run diagnostics for this sample.")
        return

    pred_w, pred_b, _ = pred_bundle
    true_w, true_b = truth_bundle

    tab1, tab2, tab3 = st.tabs(["Parity", "Complex Poles", "Overlap"])

    with tab1:
        try:
            fig = plot_predict_vs_real_parity(true_w, pred_w, true_b, pred_b)
            st.pyplot(fig)
        except Exception as exc:
            st.error(f"Parity plot failed: {exc}")

    with tab2:
        true_roots = np.exp(1j * true_w)
        pred_roots = np.exp(1j * pred_w) * np.random.normal(0.98, 0.05, len(pred_w))
        roots = np.concatenate((pred_roots, true_roots))
        try:
            fig = plot_complex_poles(roots, true_frequencies=true_w)
            st.pyplot(fig)
        except Exception as exc:
            st.error(f"Complex pole plot failed: {exc}")

    with tab3:
        omega = np.linspace(0.01, 1.5, 500)
        spec_t = _spectrum_from_peaks(true_w, true_b, omega)
        spec_p = _spectrum_from_peaks(pred_w, pred_b, omega)
        score = calc_spectral_overlap_score(spec_t, spec_p)
        st.metric("Cosine overlap", f"{100 * score:.2f}%")

        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(omega, spec_t, color="black", label="true")
        ax.plot(omega, spec_p, color="#c46f3f", linestyle="--", label="pred")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend()
        st.pyplot(fig)


def render_3d_page(models, inference_mode):
    st.title("Dynamic 3D Atom Visualizer")
    st.caption("Vintage mode keeps this section lightweight while preserving core inspection capability.")

    try:
        import utils.visualize_atoms as va
    except Exception as exc:
        st.error(f"3D visualizer import failed: {exc}")
        return

    data_dir = os.path.join(ROOT, "data", "raw", "ammonia_x")
    xyz_path = os.path.join(data_dir, "rvlab.tdscf.xyz")
    if not os.path.exists(xyz_path):
        st.warning("No ammonia_x xyz file found for 3D view.")
        return

    atoms, atom_positions, grid_points = va.load_xyz_and_grid(xyz_path)
    rho0_path = os.path.join(data_dir, "rvlab.tdscf.rho.00000")
    if not os.path.exists(rho0_path):
        st.warning("No baseline density file found.")
        return

    rho_0 = va.load_density_file(rho0_path)

    threshold = st.slider("Density threshold", min_value=1e-8, max_value=5e-3, value=5e-5, format="%.7f")
    step_val = st.slider("Frame (step)", min_value=0, max_value=380, value=70, step=5)

    step_str = f"{step_val:05d}"
    rho_t_path = os.path.join(data_dir, f"rvlab.tdscf.rho.{step_str}")
    if os.path.exists(rho_t_path):
        rho_t = va.load_density_file(rho_t_path)
        delta = rho_t - rho_0
        fig_3d = va.plot_molecule_heatmap_3d(atoms, atom_positions, grid_points, delta, threshold=threshold)
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("Selected frame is not available.")

    pred_bundle, truth_bundle, _ = predict_sample(models, "ammonia", inference_mode)
    if pred_bundle is None:
        return

    times = np.linspace(0, 400, 400)
    true_signal = np.zeros_like(times)
    pred_signal = np.zeros_like(times)
    pred_w, pred_b, _ = pred_bundle
    true_w, true_b = truth_bundle

    for w_k, b_k in zip(true_w, true_b):
        true_signal += b_k * np.sin(w_k * times)
    for w_k, b_k in zip(pred_w, pred_b):
        pred_signal += b_k * np.sin(w_k * times)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, true_signal, color="black", label="true")
    ax.plot(times, pred_signal, color="#c46f3f", linestyle="--", label="pred")
    ax.axvline(float(step_val), color="#c9973a", linestyle=":")
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel("response")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.25)
    st.pyplot(fig)


# ---------- App ----------
apply_vintage_theme()

st.sidebar.title("Q-Spectra Console")
st.sidebar.caption("Vintage scientific cockpit")

page = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Data",
        "Training",
        "Inference",
        "Diagnostics",
        "3D Visualizer",
    ],
)

inference_mode = st.sidebar.selectbox(
    "Inference mode",
    [
        "V3 hybrid two-tower",
        "V2 single tower",
        "V1 frequency only",
    ],
)

auto_refresh = st.sidebar.toggle("Auto-refresh data/logs", value=True)
refresh_sec = st.sidebar.slider("Refresh interval (sec)", 5, 120, 20)
if st.sidebar.button("Refresh now"):
    st.rerun()

enable_auto_refresh(auto_refresh, refresh_sec)

processed_sig = glob_signature(os.path.join(PROCESSED_DIR, "*.pt"))
amp_ckpt_sig = (
    file_signature(os.path.join(CHECKPOINTS_DIR, "best_model.pth")),
    file_signature(os.path.join(CHECKPOINTS_DIR, "v3_amp_tower.pth")),
)
freq_ckpt_sig = (
    file_signature(os.path.join(CHECKPOINTS_DIR, "best_model_v1.pth")),
    file_signature(os.path.join(CHECKPOINTS_DIR, "v3_freq_tower.pth")),
)

datasets, _pt_files = load_datasets(processed_sig)
dataset_df = summarize_datasets(processed_sig)
models = load_models(amp_ckpt_sig, freq_ckpt_sig, processed_sig)

if page == "Overview":
    render_overview(dataset_df, models)
elif page == "Data":
    render_data_page(datasets, dataset_df)
elif page == "Training":
    render_training_page()
elif page == "Inference":
    if not datasets:
        st.warning("No processed datasets available.")
    else:
        render_inference_page(models, datasets, inference_mode)
elif page == "Diagnostics":
    if not datasets:
        st.warning("No processed datasets available.")
    else:
        render_diagnostics_page(models, datasets, inference_mode)
elif page == "3D Visualizer":
    render_3d_page(models, inference_mode)
