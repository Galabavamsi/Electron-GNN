"""
Generate publication-quality SVG figures for Electron-GNN.

Outputs to docs/assets/figures/*.svg using matplotlib (+ networkx if available).
All figures are vector SVG; embed in slides, LaTeX, or HTML directly.

Run (after `source .venv/bin/activate` — see README Environment setup):
    python scripts/make_paper_figures.py
    python scripts/make_paper_figures.py --no-data    # skip real-data figures

Do not use the system interpreter if it lacks matplotlib; use `.venv/bin/python` instead.

Optional dependencies:
    matplotlib  (required)
    numpy       (required)
    networkx    (optional; graph_schematic falls back to manual layout)
    torch       (optional; only needed for figures based on processed .pt files)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False


# ---- Style: paper-friendly ---------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "svg.fonttype": "none",  # text stays editable in Inkscape / Illustrator
})

PALETTE = {
    "input": "#e3f2fd",      "input_e": "#1565c0",
    "embed": "#e8f5e9",      "embed_e": "#2e7d32",
    "mp":    "#fff3e0",      "mp_e":    "#ef6c00",
    "dense": "#eceff1",      "dense_e": "#455a64",
    "dec":   "#f3e5f5",      "dec_e":   "#6a1b9a",
    "head":  "#ffebee",      "head_e":  "#c62828",
    "extra": "#fff8e1",      "extra_e": "#f57c00",
}


# ---- Helpers -----------------------------------------------------------------

def _box(ax, x, y, w, h, text, fc, ec, sub=None, fontsize=11):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    cy = y + h / 2 + (0.06 if sub else 0)
    ax.text(x + w / 2, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.12, sub, ha="center", va="center",
                fontsize=fontsize - 2, color="#444")


def _arrow(ax, x1, y1, x2, y2, color="#222", lw=1.6):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12,
        color=color, lw=lw, shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arr)


def _save(fig, out_dir: Path, name: str):
    path = out_dir / f"{name}.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    print(f"  wrote {path.relative_to(Path.cwd()) if path.is_absolute() else path}")


# ---- Figure 1: Pipeline ------------------------------------------------------

def fig_pipeline(out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 2.4); ax.set_axis_off()

    boxes = [
        ("Raw RT-TDDFT",   "ReSpect .out / .xyz",  PALETTE["input"], PALETTE["input_e"]),
        ("Parse + peaks",  "Pade + LASSO",         PALETTE["embed"], PALETTE["embed_e"]),
        ("Graph + targets","data/processed/*.pt",  PALETTE["mp"],    PALETTE["mp_e"]),
        ("GNN one-shot",   "V2 or V3 hybrid",      PALETTE["dec"],   PALETTE["dec_e"]),
        ("Decode",         "slots → peaks",        PALETTE["head"],  PALETTE["head_e"]),
        ("Spectrum + metrics", "Lorentzian sum",   PALETTE["dense"], PALETTE["dense_e"]),
    ]
    bw, bh, gap = 1.95, 1.15, 0.25
    x = 0.2
    centers = []
    for label, sub, fc, ec in boxes:
        _box(ax, x, 0.6, bw, bh, label, fc, ec, sub=sub, fontsize=10)
        centers.append((x + bw / 2, 0.6 + bh / 2, x + bw))
        x += bw + gap

    for i in range(len(centers) - 1):
        x_end = centers[i][2]
        x_next = x_end + gap
        _arrow(ax, x_end + 0.02, centers[i][1], x_next - 0.02, centers[i + 1][1])

    ax.set_title("Electron-GNN pipeline (offline data → one-shot inference)",
                 loc="left", fontsize=12, pad=10)
    _save(fig, out_dir, "fig1_pipeline")


# ---- Figure 2: V2 stack ------------------------------------------------------

def fig_v2_stack(out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 9.0))
    ax.set_xlim(0, 8); ax.set_ylim(0, 11.5); ax.set_axis_off()

    blocks = [
        # (y, h, title, sub, fc, ec)
        (10.1, 0.9, "Inputs (PyG Data)",
         "x ∈ R^{N×5}  ·  edge_index ∈ Z^{2×E}  ·  edge_attr ∈ R^{E×4}  ·  batch",
         PALETTE["input"], PALETTE["input_e"]),
        (8.95, 0.85, "Embeddings",
         "node_emb 5→128  ·  edge_emb 4→128 (SiLU + LayerNorm)",
         PALETTE["embed"], PALETTE["embed_e"]),
        (6.6, 2.05, "GATv2 message passing × 4",
         "GATv2Conv (4 heads, edge-aware)  →  LayerNorm  →  GELU  →  + residual\n"
         "h ∈ R^{N×128}; same edges reused every layer",
         PALETTE["mp"], PALETTE["mp_e"]),
        (5.45, 0.85, "to_dense_batch",
         "padded memory tokens  M ∈ R^{B×N_max×128}  + key padding mask",
         PALETTE["dense"], PALETTE["dense_e"]),
        (3.55, 1.6, "TransformerDecoder × 2 (DETR-style)",
         "K_max=64 learnable queries cross-attend to atoms\n"
         "+ global pool concat  +  slot_refine MLP",
         PALETTE["dec"], PALETTE["dec_e"]),
        (1.55, 1.5, "Heads",
         "count (B,)  ·  prob (B,K)  ·  freq (B,K)  ·  amp (B,K)\n"
         "softplus on freq/amp/count, sigmoid on prob",
         PALETTE["head"], PALETTE["head_e"]),
        (0.15, 1.0, "Output: pred_dict",
         "{prob, prob_logits, freq, amp, count}",
         PALETTE["extra"], PALETTE["extra_e"]),
    ]
    for y, h, title, sub, fc, ec in blocks:
        _box(ax, 0.4, y, 7.2, h, title, fc, ec, sub=sub, fontsize=11)

    ys = [b[0] + b[1] for b in blocks]
    starts_ends = list(zip(ys[1:], [b[0] for b in blocks[:-1]]))
    for y_top, y_bot in zip(ys[1:], [b[0] for b in blocks[:-1]]):
        _arrow(ax, 4.0, y_top, 4.0, y_bot)

    ax.set_title("SpectralEquivariantGNN  (one forward, fixed K_max slots)",
                 loc="center", fontsize=13, pad=10, fontweight="bold")
    _save(fig, out_dir, "fig2_v2_stack")


# ---- Figure 3: Cutoff graph schematic ---------------------------------------

def _schematic_graph_data():
    """A small mock molecule (NH3-like layout) for the schematic."""
    pos = {
        0: (0.0, 0.85, "N"),
        1: (-0.85, 0.0, "H"),
        2: (0.85, 0.0, "H"),
        3: (0.0, -0.95, "H"),
    }
    cutoff = 1.4
    return pos, cutoff


def fig_graph_schematic(out_dir: Path):
    pos, cutoff = _schematic_graph_data()
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 4.7); ax.set_ylim(-1.6, 1.9); ax.set_axis_off()

    color_map = {"H": ("#e3f2fd", "#1565c0"),
                 "N": ("#e8f5e9", "#2e7d32"),
                 "O": ("#fff3e0", "#ef6c00"),
                 "C": ("#f3e5f5", "#6a1b9a"),
                 "F": ("#ffebee", "#c62828")}

    nodes = list(pos.keys())
    coords = {i: (pos[i][0], pos[i][1]) for i in nodes}
    elems  = {i: pos[i][2] for i in nodes}

    edges = []
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            d = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            if d <= cutoff:
                edges.append((i, j, d))

    for i, j, _ in edges:
        x1, y1 = coords[i]; x2, y2 = coords[j]
        dx, dy = x2 - x1, y2 - y1
        L = np.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        nx_, ny_ = -uy, ux
        off = 0.04
        sx = x1 + off * nx_ + 0.18 * ux
        sy = y1 + off * ny_ + 0.18 * uy
        ex = x2 + off * nx_ - 0.18 * ux
        ey = y2 + off * ny_ - 0.18 * uy
        _arrow(ax, sx, sy, ex, ey, color="#666", lw=1.0)

    for i in nodes:
        x, y = coords[i]
        fc, ec = color_map.get(elems[i], ("#eee", "#555"))
        ax.add_patch(plt.Circle((x, y), 0.18, facecolor=fc, edgecolor=ec, linewidth=1.6))
        ax.text(x, y, elems[i], ha="center", va="center", fontsize=11, fontweight="bold")

    legend_x = 2.3
    _box(
        ax, legend_x, 0.7, 2.2, 0.95, "Node feature",
        "#fafafa", "#bbb",
        sub="one-hot in R^5 over H,C,N,O,F",
        fontsize=10,
    )
    _box(
        ax, legend_x, -0.35, 2.2, 0.95, "Edge feature",
        "#fafafa", "#bbb",
        sub="[ d, Delta r_x, Delta r_y, Delta r_z ] in R^4",
        fontsize=10,
    )
    _box(
        ax, legend_x, -1.4, 2.2, 0.95, "Edge rule",
        "#fafafa", "#bbb",
        sub="add i to j if dist( r_i, r_j ) <= r_c  (r_c=5.0 a.u.)",
        fontsize=10,
    )

    ax.set_title("Cutoff molecular graph  (directed edges within r_c)",
                 fontsize=12, fontweight="bold", loc="left")
    _save(fig, out_dir, "fig3_graph_schematic")


# ---- Figure 4: Hybrid two-tower ---------------------------------------------

def fig_hybrid(out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 4.4); ax.set_axis_off()

    _box(
        ax, 0.3, 1.6, 2.4, 1.2, "Same PyG graph",
        PALETTE["input"], PALETTE["input_e"],
        sub="x, edge_index, pos",
        fontsize=11,
    )

    _box(
        ax, 4.2, 2.85, 3.2, 1.2, "V1 frequency tower",
        PALETTE["extra"], PALETTE["extra_e"],
        sub="SpectralEquivariantGNNV1",
        fontsize=11,
    )
    _box(
        ax, 4.2, 0.35, 3.2, 1.2, "V2 amplitude tower",
        PALETTE["dec"], PALETTE["dec_e"],
        sub="SpectralEquivariantGNN",
        fontsize=11,
    )

    _box(
        ax, 8.4, 1.6, 3.0, 1.2, "Hybrid combiner",
        PALETTE["dense"], PALETTE["dense_e"],
        sub="freq prior + amp match\nutils/hybrid_inference.py",
        fontsize=11,
    )

    _box(
        ax, 12.0, 1.6, 1.8, 1.2, "Peaks",
        PALETTE["head"], PALETTE["head_e"],
        sub="omega, B",
        fontsize=11,
    )

    _arrow(ax, 2.7, 2.2, 4.2, 3.45)
    _arrow(ax, 2.7, 2.2, 4.2, 0.95)
    _arrow(ax, 7.4, 3.45, 8.4, 2.4)
    _arrow(ax, 7.4, 0.95, 8.4, 1.95)
    _arrow(ax, 11.4, 2.2, 12.0, 2.2)

    ax.set_title("V3 hybrid inference  (two full-graph forwards, deterministic merge)",
                 fontsize=12, fontweight="bold", loc="left")
    _save(fig, out_dir, "fig4_hybrid")


# ---- Figure 5: Real ammonia / water graphs from data/processed --------------

def _build_real_graph(pt_path: Path):
    import torch
    d = torch.load(pt_path, weights_only=False)
    Z = d["atomic_numbers"].numpy()
    R = d["positions"].numpy()
    return Z, R


def fig_real_graphs(out_dir: Path, data_dir: Path, cutoff: float = 5.0):
    files = sorted(data_dir.glob("*_targets.pt"))
    if not files:
        print(f"  [skip] no .pt files in {data_dir}")
        return

    try:
        import torch  # noqa: F401
    except ImportError:
        print("  [skip] torch not installed")
        return

    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    z_to_color = {1: ("#e3f2fd", "#1565c0", "H"),
                  6: ("#f3e5f5", "#6a1b9a", "C"),
                  7: ("#e8f5e9", "#2e7d32", "N"),
                  8: ("#fff3e0", "#ef6c00", "O"),
                  9: ("#ffebee", "#c62828", "F")}

    for ax, fp in zip(axes[0], files):
        ax.set_aspect("equal"); ax.set_axis_off()
        Z, R = _build_real_graph(fp)
        # project to xy plane (atomic units)
        x, y = R[:, 0], R[:, 1]
        for i in range(len(Z)):
            for j in range(len(Z)):
                if i == j:
                    continue
                d = np.linalg.norm(R[i] - R[j])
                if d <= cutoff:
                    ax.plot([x[i], x[j]], [y[i], y[j]],
                            color="#888", lw=0.8, zorder=1)
        for i in range(len(Z)):
            fc, ec, sym = z_to_color.get(int(Z[i]), ("#ddd", "#555", str(int(Z[i]))))
            ax.add_patch(plt.Circle((x[i], y[i]), 0.35,
                                    facecolor=fc, edgecolor=ec, lw=1.6, zorder=2))
            ax.text(x[i], y[i], sym, ha="center", va="center",
                    fontsize=10, fontweight="bold", zorder=3)

        name = fp.stem.replace("_targets", "")
        ax.set_title(f"{name}   (N={len(Z)}, r_c={cutoff} a.u.)",
                     fontsize=11, fontweight="bold")

    fig.suptitle("Cutoff graphs from data/processed (xy projection, atomic units)",
                 fontsize=12, fontweight="bold", y=1.02)
    _save(fig, out_dir, "fig5_real_graphs")


# ---- Figure 6: Spectrum (Lorentzian) ----------------------------------------

def _lorentzian(omega, freqs, amps, gamma=0.015):
    s = np.zeros_like(omega)
    for w, b in zip(freqs, amps):
        s += b * (gamma / ((omega - w) ** 2 + gamma ** 2))
    return s


def fig_spectrum(out_dir: Path, data_dir: Path):
    files = sorted(data_dir.glob("*_targets.pt"))
    try:
        import torch  # noqa: F401
    except ImportError:
        files = []

    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    omega = np.linspace(0.05, 1.5, 2048)

    if not files:
        rng = np.random.default_rng(0)
        freqs = np.sort(rng.uniform(0.2, 1.2, size=12))
        amps = rng.uniform(0.05, 1.0, size=12)
        ax.plot(omega, _lorentzian(omega, freqs, amps),
                label="synthetic", color="#1565c0", lw=1.5)
        for w, b in zip(freqs, amps):
            ax.vlines(w, 0, b * 0.6, color="#c62828", alpha=0.6, lw=0.8)
        ax.set_title("Synthetic Lorentzian reconstruction (no data found)",
                     fontsize=12, loc="left")
    else:
        import torch
        colors = ["#1565c0", "#2e7d32", "#ef6c00", "#6a1b9a"]
        for k, fp in enumerate(files[:4]):
            d = torch.load(fp, weights_only=False)
            w = d["frequencies"].numpy()
            b = np.abs(d["amplitudes_x"].numpy())
            mask = (w >= omega.min()) & (w <= omega.max())
            w, b = w[mask], b[mask]
            if w.size == 0:
                continue
            spec = _lorentzian(omega, w, b)
            spec = spec / (spec.max() + 1e-12)
            label = fp.stem.replace("_targets", "")
            ax.plot(omega, spec, label=label, color=colors[k % len(colors)], lw=1.4)
        ax.set_title("Lorentzian-reconstructed target spectra (data/processed)",
                     fontsize=12, loc="left")

    ax.set_xlabel("Frequency ω  (a.u.)")
    ax.set_ylabel("S(ω)   (normalized)")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, out_dir, "fig6_target_spectra")


# ---- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=str,
                        default="docs/assets/figures",
                        help="Where to write SVG files")
    parser.add_argument("--data_dir", type=str,
                        default="data/processed",
                        help="Processed .pt directory for real-data figures")
    parser.add_argument("--no-data", action="store_true",
                        help="Skip figures that need data/processed")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_dir).resolve()
    data_dir = (repo_root / args.data_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[paper figures] writing to {out_dir}")
    fig_pipeline(out_dir)
    fig_v2_stack(out_dir)
    fig_graph_schematic(out_dir)
    fig_hybrid(out_dir)

    if not args.no_data:
        fig_real_graphs(out_dir, data_dir)
        fig_spectrum(out_dir, data_dir)

    print("[paper figures] done.")


if __name__ == "__main__":
    sys.exit(main())
