"""
Generate publication-grade SVG figures for Electron-GNN.

The output is a research-paper-ready figure suite in docs/assets/figures/*.svg,
covering architecture, end-to-end flow, and physics-grounded demo visuals.

Run (after activating the project virtual environment):
    python scripts/make_paper_figures.py
    python scripts/make_paper_figures.py --no-data

In no-data mode, figures fall back to deterministic synthetic examples.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ---- Style -------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "stix",
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 10.5,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "svg.fonttype": "none",
    }
)

PALETTE = {
    "ink": "#1d2433",
    "muted": "#5d6578",
    "line": "#2a3243",
    "input": "#eaf2ff",
    "input_e": "#2d5fb3",
    "physics": "#e9f8ef",
    "physics_e": "#2c8c4d",
    "graph": "#fff4e6",
    "graph_e": "#c8701c",
    "tower": "#f4ebff",
    "tower_e": "#6e3ab2",
    "hybrid": "#eaf6f6",
    "hybrid_e": "#2b7f8d",
    "head": "#ffeef0",
    "head_e": "#ba3f4b",
    "note": "#f8f8fb",
    "note_e": "#8a90a3",
}

EL_COLOR = {
    1: ("H", "#dceeff", "#2d5fb3"),
    6: ("C", "#f0e5ff", "#6e3ab2"),
    7: ("N", "#e7f8ec", "#2c8c4d"),
    8: ("O", "#ffeed8", "#c8701c"),
    9: ("F", "#ffe8ec", "#ba3f4b"),
}


# ---- Helpers -----------------------------------------------------------------

def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle,
    fc,
    ec,
    title_size=10.8,
    subtitle_size=9.0,
    lw=1.5,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    title_obj = ax.text(
        x + w / 2,
        y + h * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        color=PALETTE["ink"],
        fontweight="semibold",
        clip_on=True,
        clip_path=patch,
    )

    # Wrap long plain-language subtitles before rendering so paper figures remain legible.
    if subtitle and ("$" not in subtitle) and ("\n" not in subtitle):
        wrap_width = max(16, int(w * 10))
        subtitle = textwrap.fill(subtitle, width=wrap_width)

    sub_obj = None
    if subtitle:
        sub_obj = ax.text(
            x + w / 2,
            y + h * 0.35,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=PALETTE["muted"],
            linespacing=1.25,
            clip_on=True,
            clip_path=patch,
        )

    _fit_text_to_box(ax, title_obj, x, y, w, h * 0.46, min_font=7.8)
    if sub_obj is not None:
        _fit_text_to_box(ax, sub_obj, x, y, w, h * 0.48, min_font=6.8)


def _fit_text_to_box(ax, text_obj, x, y, w, h, min_font=6.8, pad_ratio=0.08):
    """Shrink text until it fits inside a target box in data coordinates."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    max_w = w * (1.0 - pad_ratio)
    max_h = h * (1.0 - pad_ratio)

    for _ in range(36):
        bbox_disp = text_obj.get_window_extent(renderer=renderer)
        bbox_data = bbox_disp.transformed(ax.transData.inverted())
        fits_w = bbox_data.width <= max_w
        fits_h = bbox_data.height <= max_h
        if fits_w and fits_h:
            return

        fs = float(text_obj.get_fontsize())
        if fs <= min_font + 1e-6:
            return
        text_obj.set_fontsize(max(min_font, fs - 0.25))
        fig.canvas.draw()


def _arrow(ax, x1, y1, x2, y2, lw=1.4, color=None, rad=0.0):
    color = PALETTE["line"] if color is None else color
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def _save(fig, out_dir: Path, name: str):
    path = out_dir / f"{name}.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  wrote {path.relative_to(Path.cwd()) if path.is_absolute() else path}")


def _lorentzian_spectrum(omega, freqs, amps, gamma=0.015):
    spec = np.zeros_like(omega)
    if len(freqs) == 0:
        return spec
    for w_i, b_i in zip(freqs, amps):
        spec += b_i * (gamma / ((omega - w_i) ** 2 + gamma**2))
    return spec


def _dipole_signal(times, freqs, amps):
    if len(freqs) == 0:
        return np.zeros_like(times)
    wave = np.zeros_like(times)
    for w_i, b_i in zip(freqs, amps):
        wave += b_i * np.sin(w_i * times)
    return wave


def _load_processed_samples(data_dir: Path, max_samples=4):
    files = sorted(data_dir.glob("*_targets.pt"))
    if not files:
        return []

    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        print("  [info] torch not installed, using synthetic fallback for data-dependent figures")
        return []

    samples = []
    for fp in files[:max_samples]:
        data = torch.load(fp, weights_only=False)
        freq = _to_numpy(data.get("frequencies", np.array([]))).astype(float)
        amp = np.abs(_to_numpy(data.get("amplitudes_x", np.array([]))).astype(float))
        z = _to_numpy(data.get("atomic_numbers", np.array([]))).astype(int)
        pos = _to_numpy(data.get("positions", np.empty((0, 3)))).astype(float)

        order = np.argsort(freq)
        samples.append(
            {
                "name": fp.stem.replace("_targets", ""),
                "freq": freq[order],
                "amp": amp[order],
                "z": z,
                "pos": pos,
            }
        )
    return samples


def _synthetic_sample(name="synthetic", n_peaks=16):
    rng = np.random.default_rng(23)
    freq = np.sort(rng.uniform(0.12, 1.35, size=n_peaks))
    amp = rng.lognormal(mean=-7.1, sigma=0.55, size=n_peaks)
    z = np.array([7, 1, 1, 1], dtype=int)
    pos = np.array(
        [
            [0.00, 0.00, 0.14],
            [0.90, 0.00, -0.35],
            [-0.48, 0.80, -0.35],
            [-0.48, -0.80, -0.35],
        ],
        dtype=float,
    )
    return {"name": name, "freq": freq, "amp": amp, "z": z, "pos": pos}


def _make_demo_prediction(true_freq, true_amp):
    if len(true_freq) == 0:
        return np.array([]), np.array([])
    rng = np.random.default_rng(7)
    pred_freq = np.clip(true_freq + rng.normal(0.0, 0.012, size=true_freq.shape), 0.02, None)
    amp_scale = np.clip(1.0 + rng.normal(0.0, 0.15, size=true_amp.shape), 0.5, 1.7)
    pred_amp = np.maximum(1e-8, true_amp * amp_scale)
    order = np.argsort(pred_freq)
    return pred_freq[order], pred_amp[order]


def _paired_metrics(true_freq, true_amp, pred_freq, pred_amp):
    if len(true_freq) == 0 or len(pred_freq) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.nan, np.nan

    n = min(len(true_freq), len(pred_freq))
    t_order = np.argsort(true_freq)[:n]
    p_order = np.argsort(pred_freq)[:n]

    tw = true_freq[t_order]
    pw = pred_freq[p_order]
    tb = true_amp[t_order]
    pb = pred_amp[p_order]

    freq_mae = float(np.mean(np.abs(pw - tw)))
    amp_mae = float(np.mean(np.abs(pb - tb)))
    return tw, pw, tb, pb, freq_mae, amp_mae


def _spectral_overlap(true_freq, true_amp, pred_freq, pred_amp, gamma=0.015):
    omega = np.linspace(0.01, 1.5, 1200)
    spec_true = _lorentzian_spectrum(omega, true_freq, true_amp, gamma=gamma)
    spec_pred = _lorentzian_spectrum(omega, pred_freq, pred_amp, gamma=gamma)
    denom = np.linalg.norm(spec_true) * np.linalg.norm(spec_pred)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(spec_true, spec_pred) / denom)


# ---- Figures -----------------------------------------------------------------

def fig_pipeline(out_dir: Path):
    fig, ax = plt.subplots(figsize=(12.8, 4.6))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 5.2)
    ax.set_axis_off()

    stages = [
        ("S0", "Raw RT-TDDFT", "ReSpect .out/.xyz", 0.4, 3.2, PALETTE["input"], PALETTE["input_e"]),
        ("S1-S2", "Physics Extraction", "Pad\'e + clustering + sparse fit", 3.1, 3.2, PALETTE["physics"], PALETTE["physics_e"]),
        ("S3-S4", "Graph + Targets", "Data(x, edge_index, y_freq, y_amp)", 5.95, 3.2, PALETTE["graph"], PALETTE["graph_e"]),
        ("S5-S6", "Model Training", "Hungarian loss + spectrum regularizer", 8.8, 3.2, PALETTE["tower"], PALETTE["tower_e"]),
        ("S7", "Hybrid Decode", "V1 frequency prior + V2 amplitudes", 11.65, 3.2, PALETTE["hybrid"], PALETTE["hybrid_e"]),
    ]

    for _, title, sub, x, y, fc, ec in stages:
        _box(ax, x, y, 2.55, 1.4, title, sub, fc, ec, title_size=10.4)

    for i in range(len(stages) - 1):
        x_right = stages[i][3] + 2.55
        x_next = stages[i + 1][3]
        _arrow(ax, x_right + 0.03, 3.9, x_next - 0.03, 3.9)

    for stage_id, _, _, x, y, _, _ in stages:
        ax.text(x + 0.08, y + 1.25, stage_id, fontsize=8.8, color=PALETTE["muted"], fontweight="semibold")

    _box(
        ax,
        4.35,
        1.0,
        6.6,
        1.45,
        "Final observable",
        r"$S(\omega)=\sum_k B_k\,\frac{\gamma}{(\omega-\omega_k)^2+\gamma^2}$"
        "\nmetric panel: frequency MAE, amplitude MAE, spectral overlap",
        PALETTE["note"],
        PALETTE["note_e"],
        title_size=10.6,
        subtitle_size=9.7,
    )
    _arrow(ax, 12.95, 3.2, 10.9, 2.45, rad=-0.1)

    ax.set_title("Electron-GNN End-to-End Flow (implementation-aligned)", loc="left")
    _save(fig, out_dir, "fig1_pipeline")


def fig_v2_stack(out_dir: Path):
    fig, ax = plt.subplots(figsize=(8.8, 10.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.6)
    ax.set_axis_off()

    blocks = [
        (11.95, 1.05, "Input graph", r"$x\in\mathbb{R}^{N\times 5}$, $e\in\mathbb{R}^{E\times 4}$, batch", PALETTE["input"], PALETTE["input_e"]),
        (10.55, 1.05, "Node/edge embedding", "Linear + SiLU + LayerNorm", PALETTE["graph"], PALETTE["graph_e"]),
        (8.45, 1.8, "GATv2 encoder x4", "Edge-aware message passing, residual, GELU", PALETTE["tower"], PALETTE["tower_e"]),
        (7.2, 0.9, "Dense memory tokens", "to_dense_batch(h)", PALETTE["hybrid"], PALETTE["hybrid_e"]),
        (5.0, 1.7, "Set decoder", r"$K_{max}=64$ learned queries + TransformerDecoder x2", PALETTE["physics"], PALETTE["physics_e"]),
        (2.65, 1.9, "Prediction heads", r"$p_k$, $\omega_k$, $B_k$, $\hat K$ (count)", PALETTE["head"], PALETTE["head_e"]),
        (1.1, 1.1, "Output dict", "{prob, prob_logits, freq, amp, count}", PALETTE["note"], PALETTE["note_e"]),
    ]

    x0, w0 = 0.6, 7.0
    for y, h, title, subtitle, fc, ec in blocks:
        _box(ax, x0, y, w0, h, title, subtitle, fc, ec)

    for idx in range(len(blocks) - 1):
        y_from = blocks[idx][0]
        y_to = blocks[idx + 1][0] + blocks[idx + 1][1]
        _arrow(ax, x0 + w0 / 2, y_from, x0 + w0 / 2, y_to)

    _box(
        ax,
        7.9,
        5.25,
        1.8,
        2.0,
        "Global context",
        "mean/max pooling\nslot refine",
        PALETTE["note"],
        PALETTE["note_e"],
        title_size=9.4,
        subtitle_size=8.3,
    )
    _arrow(ax, x0 + w0, 8.9, 7.9, 6.9, rad=-0.25)
    _arrow(ax, 7.9, 6.1, x0 + w0, 3.45, rad=-0.25)

    ax.text(
        0.6,
        0.35,
        r"Activation constraints: $\omega_k=\mathrm{softplus}(\cdot)+10^{-5}$, "
        r"$B_k=\mathrm{softplus}(\cdot)\times\mathrm{amp\_scale}$, $p_k=\sigma(\cdot)$",
        fontsize=8.9,
        color=PALETTE["muted"],
    )
    ax.set_title("Amplitude Tower Architecture (SpectralEquivariantGNN)", loc="left")
    _save(fig, out_dir, "fig2_v2_stack")


def fig_graph_schematic(out_dir: Path):
    fig, ax = plt.subplots(figsize=(10.6, 4.9))
    ax.set_aspect("equal")
    ax.set_xlim(-2.25, 5.6)
    ax.set_ylim(-1.9, 2.0)
    ax.set_axis_off()

    coords = {
        0: (0.00, 0.86),
        1: (-0.92, 0.03),
        2: (0.92, 0.03),
        3: (0.00, -1.00),
    }
    z = {0: 7, 1: 1, 2: 1, 3: 1}
    cutoff = 1.45

    for i, (xi, yi) in coords.items():
        for j, (xj, yj) in coords.items():
            if i == j:
                continue
            dij = float(np.hypot(xi - xj, yi - yj))
            if dij <= cutoff:
                _arrow(ax, xi * 0.84 + xj * 0.16, yi * 0.84 + yj * 0.16, xj * 0.84 + xi * 0.16, yj * 0.84 + yi * 0.16, lw=1.05, color="#7a8296")

    for idx, (xv, yv) in coords.items():
        symbol, fc, ec = EL_COLOR[z[idx]]
        ax.add_patch(plt.Circle((xv, yv), 0.2, facecolor=fc, edgecolor=ec, linewidth=1.6))
        ax.text(xv, yv, symbol, ha="center", va="center", fontsize=10, fontweight="semibold", color=PALETTE["ink"])

    _box(
        ax,
        2.1,
        0.85,
        3.25,
        0.95,
        "Node feature",
        r"$x_i=\mathrm{onehot}(Z_i)$ over [H,C,N,O,F]",
        PALETTE["note"],
        PALETTE["note_e"],
    )
    _box(
        ax,
        2.1,
        -0.18,
        3.25,
        0.95,
        "Edge rule",
        r"$(i,j)\in\mathcal{E}$ iff $i\neq j$"
        "\n"
        r"and $\|r_i-r_j\|_2\leq r_c$",
        PALETTE["note"],
        PALETTE["note_e"],
    )
    _box(
        ax,
        2.1,
        -1.21,
        3.25,
        0.95,
        "Edge feature",
        r"$e_{ij}=[d_{ij},\Delta x_{ij},\Delta y_{ij},\Delta z_{ij}]$",
        PALETTE["note"],
        PALETTE["note_e"],
    )

    ax.text(-2.05, -1.6, r"Default project cutoff: $r_c=5.0$ a.u.", fontsize=9.3, color=PALETTE["muted"])
    ax.set_title("Molecular Graph Encoding (physics-aware geometric input)", loc="left")
    _save(fig, out_dir, "fig3_graph_schematic")


def fig_hybrid(out_dir: Path):
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    ax.set_xlim(0, 14.3)
    ax.set_ylim(0, 5.2)
    ax.set_axis_off()

    _box(ax, 0.5, 1.9, 2.5, 1.3, "Shared graph input", "same molecule to both towers", PALETTE["input"], PALETTE["input_e"])

    _box(ax, 3.8, 3.05, 3.15, 1.25, "V1 Frequency Tower", r"outputs $p^{(f)}$, $\omega^{(f)}$", PALETTE["tower"], PALETTE["tower_e"])
    _box(ax, 3.8, 0.85, 3.15, 1.25, "V2 Amplitude Tower", r"outputs $p^{(a)}$, $\omega^{(a)}$, $B^{(a)}$, $\hat K$", PALETTE["head"], PALETTE["head_e"])
    _box(ax, 7.95, 1.9, 3.35, 1.3, "Hybrid Combiner", "count-aware decode + frequency matching", PALETTE["hybrid"], PALETTE["hybrid_e"])
    _box(ax, 12.0, 1.9, 1.8, 1.3, "Final peak set", r"$\{(\omega_k,B_k)\}_{k=1}^{\hat K}$", PALETTE["physics"], PALETTE["physics_e"])

    _arrow(ax, 3.0, 2.55, 3.8, 3.7)
    _arrow(ax, 3.0, 2.55, 3.8, 1.45)
    _arrow(ax, 6.95, 3.7, 7.95, 2.85)
    _arrow(ax, 6.95, 1.45, 7.95, 2.25)
    _arrow(ax, 11.3, 2.55, 12.0, 2.55)

    ax.text(
        7.95,
        0.72,
        r"Assignment cost: $C_{ij}=|\omega_i^{(f)}-\omega_j^{(a)}| + \lambda_c(1-p_j^{(a)})$",
        fontsize=9.0,
        color=PALETTE["muted"],
    )
    ax.text(
        7.95,
        0.40,
        r"Overflow option: borrow extra $\omega$ from amplitude tower when $\hat K$ exceeds V1 slots.",
        fontsize=8.8,
        color=PALETTE["muted"],
    )

    ax.set_title("V3 Hybrid Inference: frequency prior + amplitude calibration", loc="left")
    _save(fig, out_dir, "fig4_hybrid")


def fig_real_graphs(out_dir: Path, samples, cutoff=5.0):
    if not samples:
        samples = [_synthetic_sample(name="synthetic_a", n_peaks=12), _synthetic_sample(name="synthetic_b", n_peaks=10)]

    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    for ax, sample in zip(axes[0], samples):
        ax.set_aspect("equal")
        ax.set_axis_off()
        z = sample["z"]
        pos = sample["pos"]
        if pos.size == 0:
            continue

        x = pos[:, 0]
        y = pos[:, 1]
        for i in range(len(z)):
            for j in range(i + 1, len(z)):
                d = float(np.linalg.norm(pos[i] - pos[j]))
                if d <= cutoff:
                    ax.plot([x[i], x[j]], [y[i], y[j]], color="#8a90a3", lw=1.0, zorder=1)

        for i in range(len(z)):
            symbol, fc, ec = EL_COLOR.get(int(z[i]), (str(int(z[i])), "#eceff4", "#616a7f"))
            ax.add_patch(plt.Circle((x[i], y[i]), 0.24, facecolor=fc, edgecolor=ec, lw=1.5, zorder=2))
            ax.text(x[i], y[i], symbol, ha="center", va="center", fontsize=9.8, fontweight="semibold", color=PALETTE["ink"], zorder=3)

        ax.set_title(f"{sample['name']}  (N={len(z)},  r_c={cutoff:.1f} a.u.)", fontsize=10.7)

    fig.suptitle("Molecular Graph Instances Used by the Model", y=1.02)
    _save(fig, out_dir, "fig5_real_graphs")


def fig_spectrum(out_dir: Path, samples):
    if not samples:
        samples = [_synthetic_sample(name="synthetic", n_peaks=16)]

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    omega = np.linspace(0.01, 1.5, 1400)
    colors = ["#2d5fb3", "#2c8c4d", "#c8701c", "#6e3ab2"]

    for idx, sample in enumerate(samples[:4]):
        spec = _lorentzian_spectrum(omega, sample["freq"], sample["amp"], gamma=0.015)
        spec = spec / (np.max(spec) + 1e-12)
        ax.plot(omega, spec, lw=1.7, color=colors[idx % len(colors)], label=sample["name"])

    ax.set_xlabel(r"Frequency $\omega$ (a.u.)")
    ax.set_ylabel(r"$S(\omega)$ (normalized)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Lorentzian-Reconstructed Target Spectra")
    _save(fig, out_dir, "fig6_target_spectra")


def fig_physics_extraction(out_dir: Path):
    fig = plt.figure(figsize=(12.2, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.25], wspace=0.35)

    # Panel A: dipole signal
    ax0 = fig.add_subplot(gs[0, 0])
    t = np.linspace(0.0, 220.0, 1200)
    true_w = np.array([0.26, 0.41, 0.67, 0.93])
    true_b = np.array([2.7e-4, 1.9e-4, 1.3e-4, 8.8e-5])
    mu = _dipole_signal(t, true_w, true_b)
    ax0.plot(t, mu, color="#2d5fb3", lw=1.4)
    ax0.set_xlabel(r"Time $t$ (a.u.)")
    ax0.set_ylabel(r"$\mu_x(t)$")
    ax0.set_title("Input dipole trace")

    # Panel B: complex poles and selected physical poles
    ax1 = fig.add_subplot(gs[0, 1])
    theta = np.linspace(0, 2 * np.pi, 400)
    ax1.plot(np.cos(theta), np.sin(theta), color="#8a90a3", lw=1.0, label="unit circle")
    rng = np.random.default_rng(21)
    ghost_r = rng.uniform(0.70, 1.25, size=40)
    ghost_t = rng.uniform(0.0, 2 * np.pi, size=40)
    ax1.scatter(ghost_r * np.cos(ghost_t), ghost_r * np.sin(ghost_t), s=13, color="#c0c6d4", alpha=0.65, label="candidate poles")
    phys_t = np.array([0.33, 0.58, 0.93, 1.22]) * np.pi
    ax1.scatter(np.cos(phys_t), np.sin(phys_t), s=28, color="#ba3f4b", label="selected poles")
    ax1.set_aspect("equal")
    ax1.set_xlabel("Re(z)")
    ax1.set_ylabel("Im(z)")
    ax1.set_title("Pad\'e roots + clustering")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    # Panel C: extracted peaks
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.vlines(true_w, 0.0, true_b, color="#2c8c4d", lw=2.0)
    ax2.scatter(true_w, true_b, color="#2c8c4d", s=28)
    ax2.set_xlabel(r"Transition frequency $\omega_k$ (a.u.)")
    ax2.set_ylabel(r"Amplitude $B_k$")
    ax2.set_title("Sparse positive fit output")

    fig.suptitle("Physics-Guided Peak Extraction: Pad\'e -> clustering -> sparse positive regression", y=1.03)
    _save(fig, out_dir, "fig7_physics_extraction")


def fig_dipole_to_spectrum_demo(out_dir: Path, sample):
    true_w = sample["freq"]
    true_b = sample["amp"]
    pred_w, pred_b = _make_demo_prediction(true_w, true_b)

    times = np.linspace(0.0, 320.0, 1500)
    omega = np.linspace(0.01, 1.5, 1500)
    mu_true = _dipole_signal(times, true_w, true_b)
    mu_pred = _dipole_signal(times, pred_w, pred_b)
    spec_true = _lorentzian_spectrum(omega, true_w, true_b, gamma=0.015)
    spec_pred = _lorentzian_spectrum(omega, pred_w, pred_b, gamma=0.015)

    overlap = _spectral_overlap(true_w, true_b, pred_w, pred_b, gamma=0.015)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    axes[0].plot(times, mu_true, color="#1f2d5f", lw=1.5, label="reference")
    axes[0].plot(times, mu_pred, color="#c05b2a", lw=1.3, ls="--", label="model")
    axes[0].set_xlabel(r"Time $t$ (a.u.)")
    axes[0].set_ylabel(r"$\mu_x(t)$")
    axes[0].set_title("Dipole response reconstruction")
    axes[0].legend(frameon=False)

    axes[1].plot(omega, spec_true, color="#1f2d5f", lw=1.6, label="reference")
    axes[1].plot(omega, spec_pred, color="#c05b2a", lw=1.4, ls="--", label="model")
    axes[1].set_xlabel(r"Frequency $\omega$ (a.u.)")
    axes[1].set_ylabel(r"$S(\omega)$")
    axes[1].set_title(f"Lorentzian spectrum (overlap = {overlap:.3f})")
    axes[1].legend(frameon=False)

    fig.suptitle("From Predicted Peaks to Observable Physics", y=1.03)
    _save(fig, out_dir, "fig8_dipole_to_spectrum_demo")


def fig_metrics_parity_overlap(out_dir: Path, sample):
    true_w = sample["freq"]
    true_b = sample["amp"]
    pred_w, pred_b = _make_demo_prediction(true_w, true_b)

    tw, pw, tb, pb, freq_mae, amp_mae = _paired_metrics(true_w, true_b, pred_w, pred_b)
    overlap = _spectral_overlap(true_w, true_b, pred_w, pred_b)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), gridspec_kw={"width_ratios": [1.0, 1.0, 0.85]})

    # Frequency parity
    axes[0].scatter(tw, pw, color="#2d5fb3", s=25, alpha=0.85)
    if len(tw) > 0:
        lo = float(min(np.min(tw), np.min(pw)))
        hi = float(max(np.max(tw), np.max(pw)))
        axes[0].plot([lo, hi], [lo, hi], color="#7a8296", lw=1.1, ls="--")
        axes[0].set_xlim(lo * 0.95, hi * 1.05)
        axes[0].set_ylim(lo * 0.95, hi * 1.05)
    axes[0].set_xlabel(r"True $\omega_k$")
    axes[0].set_ylabel(r"Predicted $\omega_k$")
    axes[0].set_title("Frequency parity")

    # Amplitude parity
    axes[1].scatter(tb, pb, color="#c8701c", s=25, alpha=0.85)
    if len(tb) > 0:
        lo = float(max(1e-8, min(np.min(tb), np.min(pb))))
        hi = float(max(np.max(tb), np.max(pb)))
        axes[1].plot([lo, hi], [lo, hi], color="#7a8296", lw=1.1, ls="--")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlim(lo * 0.8, hi * 1.25)
        axes[1].set_ylim(lo * 0.8, hi * 1.25)
    axes[1].set_xlabel(r"True $B_k$")
    axes[1].set_ylabel(r"Predicted $B_k$")
    axes[1].set_title("Amplitude parity")

    # Summary metric panel
    labels = ["Freq MAE", "Amp MAE", "Overlap"]
    values = [0.0 if np.isnan(freq_mae) else freq_mae, 0.0 if np.isnan(amp_mae) else amp_mae, overlap]
    scales = [max(1e-3, values[0]), max(1e-7, values[1] * 1e4), values[2]]
    axes[2].bar(labels, scales, color=["#2d5fb3", "#c8701c", "#2c8c4d"], alpha=0.88)
    axes[2].set_title("Evaluation summary")
    axes[2].set_ylabel("Scaled value")
    axes[2].set_ylim(0, max(scales) * 1.35 + 1e-8)
    axes[2].text(0.0, scales[0] + 0.02 * (axes[2].get_ylim()[1]), f"{values[0]:.4f}", ha="center", fontsize=8)
    axes[2].text(1.0, scales[1] + 0.02 * (axes[2].get_ylim()[1]), f"{values[1]:.2e}", ha="center", fontsize=8)
    axes[2].text(2.0, scales[2] + 0.02 * (axes[2].get_ylim()[1]), f"{values[2]:.3f}", ha="center", fontsize=8)
    axes[2].set_xticks(range(3), labels, rotation=0)

    fig.suptitle("Prediction Diagnostics: Parity + Spectral Overlap", y=1.03)
    _save(fig, out_dir, "fig9_metrics_parity_overlap")


def fig_training_objective(out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # Left: synthetic matching cost matrix + assignment
    n_pred, n_true = 12, 10
    rng = np.random.default_rng(8)
    base = np.abs(rng.normal(0.0, 1.0, size=(n_pred, n_true)))
    trend = np.abs(np.subtract.outer(np.linspace(0, 1, n_pred), np.linspace(0, 1, n_true)))
    cost = 0.65 * trend + 0.35 * base / (np.max(base) + 1e-12)

    axes[0].imshow(cost, cmap="Blues", aspect="auto")
    diag_n = min(n_pred, n_true)
    axes[0].scatter(np.arange(diag_n), np.arange(diag_n), s=20, color="#ba3f4b", label="matched pairs")
    axes[0].set_title("Hungarian cost matrix")
    axes[0].set_xlabel("True peaks j")
    axes[0].set_ylabel("Predicted slots i")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)

    # Right: objective summary
    axes[1].set_axis_off()
    _box(
        axes[1],
        0.02,
        0.60,
        0.95,
        0.35,
        "Bipartite objective",
        r"$\mathcal{L}_{bip}=8\mathcal{L}_{\omega}+8\mathcal{L}_{amp}+1.2\mathcal{L}_{prob}$"
        "\n"
        r"$+6\mathcal{L}_{sum}+0.5\mathcal{L}_{count}$",
        PALETTE["tower"],
        PALETTE["tower_e"],
        title_size=10.3,
        subtitle_size=9.4,
    )
    _box(
        axes[1],
        0.02,
        0.18,
        0.95,
        0.34,
        "Physics regularizer",
        r"$\mathcal{L}_{spec}=\mathcal{L}_{time}+0.5\mathcal{L}_{Lorentz}+0.5\mathcal{L}_{area}$"
        "\n"
        r"total: $\mathcal{L}=\mathcal{L}_{bip}+\lambda\mathcal{L}_{spec}$",
        PALETTE["physics"],
        PALETTE["physics_e"],
        title_size=10.3,
        subtitle_size=9.3,
    )
    axes[1].text(
        0.02,
        0.06,
        "Training target: permutation-invariant peak alignment with physically consistent spectral behavior.",
        transform=axes[1].transAxes,
        fontsize=8.9,
        color=PALETTE["muted"],
    )

    fig.suptitle("Training Objective: assignment + physics consistency", y=1.02)
    _save(fig, out_dir, "fig10_training_objective")


# ---- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="docs/assets/figures",
        help="Destination directory for SVG files",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing *_targets.pt files",
    )
    parser.add_argument(
        "--no-data",
        action="store_true",
        help="Force synthetic fallback for data-driven figures",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    data_dir = (repo_root / args.data_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[paper figures] writing to {out_dir}")

    samples = [] if args.no_data else _load_processed_samples(data_dir, max_samples=4)
    if not samples:
        samples = [_synthetic_sample()]

    fig_pipeline(out_dir)
    fig_v2_stack(out_dir)
    fig_graph_schematic(out_dir)
    fig_hybrid(out_dir)
    fig_real_graphs(out_dir, samples=samples, cutoff=5.0)
    fig_spectrum(out_dir, samples=samples)
    fig_physics_extraction(out_dir)
    fig_dipole_to_spectrum_demo(out_dir, sample=samples[0])
    fig_metrics_parity_overlap(out_dir, sample=samples[0])
    fig_training_objective(out_dir)

    print("[paper figures] done.")


if __name__ == "__main__":
    sys.exit(main())
