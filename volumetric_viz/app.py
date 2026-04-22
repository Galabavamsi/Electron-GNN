"""
app.py — Standalone Streamlit dashboard for isosurface & contour visualization.

Run with:
    streamlit run volumetric_viz/app.py

The app reads directly from data/raw/{molecule}/ directories and provides
an interactive UI for all four visualization modes:
  1. Isosurface (±Δρ lobes)
  2. Volume rendering (semi-transparent cloud)
  3. 2D Contour cross-section
  4. Combined 3D (isosurface + contour planes)
  5. Animated time-evolution
"""

from __future__ import annotations

import os
import sys
import time

# Make sure the project root is on the path regardless of how the script is called.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import numpy as np
import streamlit as st

from volumetric_viz.loader import RespectLoader
from volumetric_viz.interpolate import scatter_to_grid, normalise_symmetric
from volumetric_viz.isosurface import build_volume_trace
from volumetric_viz.render import (
    make_isosurface_figure,
    make_contour_figure,
    make_combined_figure,
    make_animated_isosurface,
    make_three_panel_contour,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Electron Density Visualizer",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — dark, premium, minimalist
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0d1117;
    color: #c9d1d9;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] .css-1d391kg {
    padding-top: 1rem;
}

/* Cards */
.viz-card {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.viz-card h4 {
    color: #58a6ff;
    font-weight: 600;
    margin: 0 0 0.4rem 0;
    font-size: 1.05rem;
}
.viz-card p {
    color: #8b949e;
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.5;
}

/* Metric columns */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}
.metric-box {
    flex: 1;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-box .val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #58a6ff;
}
.metric-box .lbl {
    font-size: 0.8rem;
    color: #8b949e;
    margin-top: 0.2rem;
}

/* Section headers */
.section-header {
    border-left: 3px solid #58a6ff;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}
.section-header h2 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 600;
    color: #e6edf3;
}
.section-header p {
    margin: 0.2rem 0 0 0;
    font-size: 0.85rem;
    color: #8b949e;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 1.5rem 0;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.2em 0.7em;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.3rem;
}
.badge-blue  { background: #1f4068; color: #58a6ff; }
.badge-green { background: #0d4429; color: #3fb950; }
.badge-red   { background: #4a1c1c; color: #f85149; }
.badge-grey  { background: #21262d; color: #8b949e;  }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚛ Electron Density")
    st.markdown("#### Isosurface & Contour Visualizer")
    st.markdown("<hr style='border-color:#21262d;margin:0.5rem 0 1rem 0'>", unsafe_allow_html=True)

    # -- Molecule selector --
    raw_root = os.path.join(_ROOT, "data", "raw")
    available = sorted([
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
    ]) if os.path.isdir(raw_root) else []

    if not available:
        st.error("No raw data directories found under `data/raw/`.")
        st.stop()

    molecule = st.selectbox("🔬 Molecule", available, index=0)
    run_dir = os.path.join(raw_root, molecule)

    st.markdown("<hr style='border-color:#21262d;margin:0.8rem 0'>", unsafe_allow_html=True)

    # -- Visualization mode --
    st.markdown("**Visualization Mode**")
    viz_mode = st.radio(
        "", 
        [
            "🔵 Isosurface (±lobes)",
            "🌫 Volume Rendering",
            "🗺 Contour Slices (2D)",
            "🔮 Combined 3D",
            "🎬 Animated Time Evolution",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#21262d;margin:0.8rem 0'>", unsafe_allow_html=True)
    st.markdown("**Parameters**")

    # -- Snapshot selector --
    snapshot_idx = st.slider(
        "Snapshot index (t)",
        min_value=1,
        max_value=40,
        value=8,
        help="Index into the sorted .rho.* files (1 = first non-baseline).",
    )

    resolution = st.select_slider(
        "Grid resolution",
        options=[20, 28, 36, 44, 52],
        value=36,
        help="Points per axis for interpolation. Higher = sharper but slower.",
    )

    if "Isosurface" in viz_mode or "Combined" in viz_mode:
        isovalue = st.slider(
            "Isovalue (%max)",
            min_value=5, max_value=70, value=25, step=5,
        ) / 100.0
        iso_opacity = st.slider(
            "Surface opacity",
            min_value=10, max_value=90, value=55, step=5,
        ) / 100.0
    else:
        isovalue = 0.25
        iso_opacity = 0.55

    if "Contour" in viz_mode or "Combined" in viz_mode:
        slice_axis = st.selectbox("Cut plane axis", ["z", "y", "x"], index=0)
        n_contours = st.slider("Contour levels", 10, 40, 22, step=2)
    else:
        slice_axis = "z"
        n_contours = 22

    if "Animated" in viz_mode:
        anim_stride = st.slider("Frame stride (snapshots)", 1, 10, 3)
        max_frames = st.slider("Max frames", 5, 30, 15)
        frame_ms = st.slider("Frame duration (ms)", 50, 500, 200, step=50)

    st.markdown("<hr style='border-color:#21262d;margin:0.8rem 0'>", unsafe_allow_html=True)
    show_raw_stats = st.checkbox("Show raw density statistics", value=False)


# ---------------------------------------------------------------------------
# Data loading (cached per molecule + snapshot)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_loader(run_dir: str) -> RespectLoader:
    loader = RespectLoader(run_dir)
    loader.load()
    return loader


@st.cache_data(show_spinner=False)
def get_volume(
    run_dir: str,
    snapshot_idx: int,
    resolution: int,
) -> tuple:
    """Returns (X, Y, Z, vol_raw, atoms, positions, steps)."""
    loader = get_loader(run_dir)
    actual_idx = min(snapshot_idx, loader.n_snapshots - 1)
    delta_rho = loader.load_delta_rho(actual_idx)
    X, Y, Z, vol = scatter_to_grid(
        loader.grid_points, delta_rho, resolution=resolution
    )
    return X, Y, Z, vol, loader.atoms, loader.atom_positions, loader.snapshot_steps


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div style="padding:1.5rem 0 0.5rem 0">
    <h1 style="margin:0;font-size:2rem;font-weight:700;color:#e6edf3">
        ⚛ Electron Density Visualizer
    </h1>
    <p style="margin:0.3rem 0 0 0;color:#8b949e;font-size:1rem">
        RT-TDDFT isosurface & contour surface visualization &mdash; Δρ(r) = ρ(t) − ρ(t₀)
    </p>
</div>
<hr style="border:none;border-top:1px solid #21262d;margin:0.8rem 0 1.2rem 0">
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

progress_placeholder = st.empty()
with st.spinner(f"Loading & interpolating density for **{molecule}** (snapshot {snapshot_idx})…"):
    t0 = time.time()
    try:
        X, Y, Z, vol_raw, atoms, positions, steps = get_volume(run_dir, snapshot_idx, resolution)
    except Exception as e:
        st.error(f"**Data loading failed:** {e}")
        st.info("Make sure .xyz and .rho.* files exist in the selected run directory.")
        st.stop()
    elapsed = time.time() - t0

n_atoms = len(atoms)
n_grid  = vol_raw.size
delta_max = float(np.max(np.abs(vol_raw)))
step_label = steps[min(snapshot_idx, len(steps) - 1)] if steps else snapshot_idx


# ---------------------------------------------------------------------------
# Info row
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Molecule", molecule.replace("_x", "").capitalize())
with c2:
    st.metric("Atoms", n_atoms)
with c3:
    st.metric("Δρ max (a.u.)", f"{delta_max:.5f}")
with c4:
    st.metric("Step", step_label)

st.markdown("<hr style='border-color:#21262d;margin:0.4rem 0 1rem 0'>", unsafe_allow_html=True)

# Optional raw stats
if show_raw_stats:
    with st.expander("📊 Raw density statistics", expanded=False):
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Grid pts", f"{X.size:,}")
        col_b.metric("Vol min", f"{float(vol_raw.min()):.5e}")
        col_c.metric("Vol max", f"{float(vol_raw.max()):.5e}")
        col_d.metric("Load time", f"{elapsed:.2f} s")
        pos_frac = float(np.sum(vol_raw > 0)) / vol_raw.size
        neg_frac = float(np.sum(vol_raw < 0)) / vol_raw.size
        colr1, colr2 = st.columns(2)
        colr1.metric("Positive voxels (%)", f"{pos_frac*100:.1f}")
        colr2.metric("Negative voxels (%)", f"{neg_frac*100:.1f}")


# ---------------------------------------------------------------------------
# Render selected visualization
# ---------------------------------------------------------------------------

def _mode_header(title: str, desc: str):
    st.markdown(f"""
    <div class="section-header">
        <h2>{title}</h2>
        <p>{desc}</p>
    </div>
    """, unsafe_allow_html=True)


# -- 1. Isosurface --
if "Isosurface" in viz_mode:
    _mode_header(
        "🔵 Isosurface — ±Δρ Lobes",
        "Blue: electron accumulation (Δρ > 0). Red: electron depletion (Δρ < 0).",
    )
    with st.spinner("Rendering isosurface…"):
        fig = make_isosurface_figure(
            X, Y, Z, vol_raw, atoms, positions,
            isovalue=isovalue,
            opacity=iso_opacity,
            title=f"Δρ(r) Isosurface — {molecule} (step {step_label})",
        )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="viz-card">
        <h4>Reading Guide</h4>
        <p>
            <span class="badge badge-blue">Blue lobe</span> regions where electron density <em>increased</em>
            relative to the unperturbed ground state ρ(t₀).
            <span class="badge badge-red">Red lobe</span> regions where electron density <em>decreased</em>.
            These directly reveal the direction of charge polarisation induced by the laser pulse.
        </p>
    </div>""", unsafe_allow_html=True)


# -- 2. Volume rendering --
elif "Volume" in viz_mode:
    _mode_header(
        "🌫 Volume Rendering — Δρ Cloud",
        "Semi-transparent volumetric rendering showing the full 3D density distribution.",
    )
    import plotly.graph_objects as go
    from volumetric_viz.render import _atom_traces, _scene_layout

    vol_n = normalise_symmetric(vol_raw)
    with st.spinner("Rendering volume…"):
        vol_trace = build_volume_trace(X, Y, Z, vol_n, opacity_scale=0.07, surface_count=18)
        mol_traces = _atom_traces(atoms, positions)
        layout = _scene_layout(positions, f"Δρ(r) Volume — {molecule} (step {step_label})")
        layout["height"] = 620
        fig = go.Figure(data=mol_traces + [vol_trace])
        fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# -- 3. Contour slices 2D --
elif "Contour Slices" in viz_mode:
    _mode_header(
        "🗺 Contour Cross-Sections",
        "2D contour maps of Δρ through orthogonal cutting planes.",
    )
    tab1, tab2 = st.tabs(["Single slice", "Three-panel view"])

    with tab1:
        with st.spinner("Rendering contour…"):
            fig = make_contour_figure(
                X, Y, Z, vol_raw,
                axis=slice_axis,
                n_contours=n_contours,
                title=f"Δρ(r) Contour ({slice_axis}=0) — {molecule} (step {step_label})",
                atom_positions=positions,
                atoms=atoms,
            )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        with st.spinner("Rendering three panels…"):
            fig3 = make_three_panel_contour(
                X, Y, Z, vol_raw,
                title=f"Δρ(r) Three Cross-Sections — {molecule} (step {step_label})",
            )
        st.plotly_chart(fig3, use_container_width=True)


# -- 4. Combined 3D --
elif "Combined" in viz_mode:
    _mode_header(
        "🔮 Combined — Isosurface + Contour Planes",
        "Isosurface lobes with coloured contour planes at x=0, y=0, z=0.",
    )
    with st.spinner("Rendering combined scene…"):
        fig = make_combined_figure(
            X, Y, Z, vol_raw, atoms, positions,
            isovalue=isovalue,
            iso_opacity=iso_opacity,
            title=f"Δρ(r) Combined — {molecule} (step {step_label})",
        )
    st.plotly_chart(fig, use_container_width=True)


# -- 5. Animated --
elif "Animated" in viz_mode:
    _mode_header(
        "🎬 Animated Time Evolution",
        "Isosurface evolution of Δρ(r) across RT-TDDFT time steps.",
    )
    st.info(
        "⏳ Animation builds all frames upfront — may take ~20–45 seconds. "
        "Use a small stride and ≤15 frames for best performance."
    )
    if st.button("🚀 Build Animation", type="primary"):
        loader_c = get_loader(run_dir)
        frame_data = list(loader_c.iter_delta_rho(step=anim_stride, max_frames=max_frames))

        progress_bar = st.progress(0, text="Interpolating frames…")
        grid_frames = []
        for fi, (step, drho) in enumerate(frame_data):
            Xf, Yf, Zf, volf = scatter_to_grid(loader_c.grid_points, drho, resolution=resolution)
            grid_frames.append((step, volf))
            progress_bar.progress((fi + 1) / len(frame_data), text=f"Frame {fi+1}/{len(frame_data)}")

        progress_bar.empty()
        with st.spinner("Building Plotly animation…"):
            fig = make_animated_isosurface(
                Xf, Yf, Zf, atoms, positions,
                frame_data=grid_frames,
                isovalue=isovalue,
                opacity=iso_opacity,
                title=f"Δρ(r) Time Evolution — {molecule}",
                frame_duration_ms=frame_ms,
            )
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"Animation complete — {len(grid_frames)} frames rendered.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("<hr style='border:none;border-top:1px solid #21262d;margin:2rem 0 0.8rem 0'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#484f58;font-size:0.82rem">
    Electron-GNN · RT-TDDFT Volumetric Visualization · 
    Δρ(r) = ρ(t) − ρ(t₀) · 
    Grid interpolated via LinearNDInterpolator · 
    Isosurfaces via Plotly Marching-Cubes
</div>
""", unsafe_allow_html=True)
