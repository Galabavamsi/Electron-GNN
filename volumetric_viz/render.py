"""
render.py — High-level, publication-quality Plotly figure builders.

Every figure function inserts the molecular skeleton (atoms + bonds) into a
shared 3D scene and layers the requested volumetric visual on top.

Colour conventions (ACS/IUPAC standard CPK):
    H  → silver (#CCCCCC)    N → blue   (#3050F8)
    C  → dark grey (#333333)  O → red    (#FF0D0D)
    F  → light green (#90E050)
    Unknown → green (#00FF00)
"""

from __future__ import annotations

from typing import Optional, Literal, List, Tuple

import numpy as np
import plotly.graph_objects as go

from .isosurface import build_isosurface_trace, build_volume_trace
from .contour import build_contour_slice, make_three_plane_figure
from .interpolate import normalise_symmetric


# ---------------------------------------------------------------------------
# CPK atom styles
# ---------------------------------------------------------------------------

_CPK_COLORS = {
    "H":  "#CCCCCC", "C":  "#333333", "N":  "#3050F8",
    "O":  "#FF0D0D", "F":  "#90E050", "S":  "#FFFF30",
    "Cl": "#1FF01F", "P":  "#FF8000", "Lu": "#00FFFF",
}
_CPK_RADII = {
    "H": 10, "C": 16, "N": 16, "O": 16, "F": 12,
    "S": 22, "Cl": 20, "P": 18,
}
_BOND_CUTOFF_AU = 2.5   # ~1.32 Å in a.u.


# ---------------------------------------------------------------------------
# Shared scene utilities
# ---------------------------------------------------------------------------

def _atom_traces(
    atoms: list[str],
    positions: np.ndarray,           # (N, 3) a.u.
    label_atoms: bool = True,
) -> list[go.Scatter3d]:
    """Return bond + atom Scatter3d traces."""
    traces: list[go.Scatter3d] = []
    n = len(atoms)

    # --- Bonds ---
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < _BOND_CUTOFF_AU:
                traces.append(go.Scatter3d(
                    x=[positions[i, 0], positions[j, 0]],
                    y=[positions[i, 1], positions[j, 1]],
                    z=[positions[i, 2], positions[j, 2]],
                    mode="lines",
                    line=dict(color="#888", width=8),
                    hoverinfo="skip",
                    showlegend=False,
                ))

    # --- Atoms ---
    for i, sym in enumerate(atoms):
        color  = _CPK_COLORS.get(sym, "#00FF00")
        radius = _CPK_RADII.get(sym, 14)
        traces.append(go.Scatter3d(
            x=[positions[i, 0]],
            y=[positions[i, 1]],
            z=[positions[i, 2]],
            mode="markers+text" if label_atoms else "markers",
            marker=dict(
                size=radius,
                color=color,
                line=dict(color="white", width=1.5),
                symbol="circle",
            ),
            text=[f"<b>{sym}</b>"] if label_atoms else None,
            textfont=dict(size=11, color="white"),
            textposition="middle center",
            name=f"Atom {sym}{i+1}",
            showlegend=False,
        ))
    return traces


def _scene_layout(
    positions: np.ndarray,
    title: str,
    bgcolor: str = "#0d1117",
    camera: dict | None = None,
) -> dict:
    pad = 3.0
    lim = max(
        np.abs(positions).max() + pad,
        6.0,
    )
    cam = camera or dict(eye=dict(x=1.4, y=1.4, z=1.4))
    axis_style = dict(
        showgrid=False,
        zeroline=False,
        showbackground=True,
        backgroundcolor=bgcolor,
        showspikes=False,
        showticklabels=True,
        tickfont=dict(size=10, color="#8b949e"),
        title_font=dict(size=11, color="#8b949e"),
        gridcolor="#21262d",
        linecolor="#30363d",
    )
    return dict(
        scene=dict(
            xaxis=dict(range=[-lim, lim], title="x (a.u.)", **axis_style),
            yaxis=dict(range=[-lim, lim], title="y (a.u.)", **axis_style),
            zaxis=dict(range=[-lim, lim], title="z (a.u.)", **axis_style),
            aspectmode="cube",
            bgcolor=bgcolor,
            camera=cam,
        ),
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=15, family="Inter, Arial, sans-serif", color="#e6edf3"),
        ),
        paper_bgcolor=bgcolor,
        legend=dict(
            font=dict(size=11, color="#c9d1d9"),
            bgcolor="rgba(22,27,34,0.8)",
            bordercolor="#30363d",
            borderwidth=1,
            x=0.01,
            y=0.99,
        ),
        margin=dict(t=50, l=0, r=0, b=0),
        font=dict(family="Inter, Arial, sans-serif"),
    )


# ---------------------------------------------------------------------------
# Figure 1 — Isosurface (± lobes)
# ---------------------------------------------------------------------------

def make_isosurface_figure(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    vol: np.ndarray,
    atoms: list[str],
    positions: np.ndarray,
    isovalue: float = 0.25,
    opacity: float = 0.55,
    title: str = "Δρ(r) Isosurface",
    surface_count: int = 1,
) -> go.Figure:
    """
    Professional 3D isosurface figure with molecular skeleton.

    Positive Δρ lobe → royal blue  (electron accumulation)
    Negative Δρ lobe → crimson     (electron depletion)

    Parameters
    ----------
    X, Y, Z  : meshgrid from scatter_to_grid
    vol      : normalised (−1…1) density difference volume
    atoms    : list of element symbols
    positions: (N, 3) atomic positions in a.u.
    isovalue : isosurface level (fraction of max, 0–1)
    opacity  : surface opacity
    title    : figure title
    surface_count : isosurfaces per lobe
    """
    vol_n = normalise_symmetric(vol)
    trace_pos, trace_neg = build_isosurface_trace(
        X, Y, Z, vol_n,
        isovalue=isovalue,
        pos_color="royalblue",
        neg_color="crimson",
        opacity=opacity,
        surface_count=surface_count,
    )
    mol_traces = _atom_traces(atoms, positions)
    layout = _scene_layout(positions, title)
    layout["height"] = 600

    fig = go.Figure(data=mol_traces + [trace_pos, trace_neg])
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Contour cross-sections (2D)
# ---------------------------------------------------------------------------

def make_contour_figure(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    vol: np.ndarray,
    axis: Literal["x", "y", "z"] = "z",
    level: float = 0.0,
    n_contours: int = 22,
    title: str = "Δρ(r) Contour Cross-Section",
    atom_positions: np.ndarray | None = None,
    atoms: list[str] | None = None,
) -> go.Figure:
    """
    2D contour plot of a planar cross-section of Δρ(r).

    Parameters
    ----------
    axis  : normal direction ("x", "y", "z")
    level : plane position in a.u.
    """
    vol_n = normalise_symmetric(vol)
    slc = build_contour_slice(X, Y, Z, vol_n, axis=axis, level=level,
                              n_contours=n_contours, label="Δρ")
    fig = go.Figure(data=[slc["trace_2d"]])

    # Optionally overlay atoms projected onto the slice plane
    if atom_positions is not None and atoms is not None:
        _add_projected_atoms(fig, atoms, atom_positions, axis)

    axis_map = dict(
        x=("y", "z"), y=("x", "z"), z=("x", "y")
    )
    ax1l = f"{axis_map[axis][0]} (a.u.)"
    ax2l = f"{axis_map[axis][1]} (a.u.)"

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sup>{slc['plane_label']}</sup>",
            x=0.5,
            xanchor="center",
            font=dict(size=15, family="Inter, Arial, sans-serif", color="#e6edf3"),
        ),
        xaxis=dict(
            title=ax1l,
            gridcolor="#30363d",
            zerolinecolor="#58a6ff",
            linecolor="#30363d",
        ),
        yaxis=dict(
            title=ax2l,
            gridcolor="#30363d",
            zerolinecolor="#58a6ff",
            linecolor="#30363d",
            scaleanchor="x",
            scaleratio=1,
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#c9d1d9"),
        height=500,
        margin=dict(t=70, l=60, r=30, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Combined 3D: isosurface + contour planes
# ---------------------------------------------------------------------------

def make_combined_figure(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    vol: np.ndarray,
    atoms: list[str],
    positions: np.ndarray,
    isovalue: float = 0.25,
    iso_opacity: float = 0.45,
    plane_axes: list[str] | None = None,
    title: str = "Δρ(r) — Isosurface + Contour Planes",
) -> go.Figure:
    """
    Combined 3D figure with isosurface lobes + embedded contour slice planes.

    Parameters
    ----------
    plane_axes : list of axes to cut planes for, e.g. ["x", "y", "z"].
                 Default: all three.
    """
    if plane_axes is None:
        plane_axes = ["x", "y", "z"]

    vol_n = normalise_symmetric(vol)

    traces: list[go.BaseTraceType] = []
    traces.extend(_atom_traces(atoms, positions))

    # Isosurface lobes
    t_pos, t_neg = build_isosurface_trace(
        X, Y, Z, vol_n,
        isovalue=isovalue,
        opacity=iso_opacity,
        surface_count=1,
    )
    traces.append(t_pos)
    traces.append(t_neg)

    # Contour planes
    for ax in plane_axes:
        slc = build_contour_slice(X, Y, Z, vol_n, axis=ax, level=0.0,
                                  show_colorbar=False, label="Δρ")
        traces.append(slc["trace_3d_surface"])

    layout = _scene_layout(positions, title)
    layout["height"] = 650
    fig = go.Figure(data=traces)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Animated time-evolution isosurface
# ---------------------------------------------------------------------------

def make_animated_isosurface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    atoms: list[str],
    positions: np.ndarray,
    frame_data: list[Tuple[int, np.ndarray]],   # [(step, vol), ...]
    isovalue: float = 0.25,
    opacity: float = 0.55,
    title: str = "Δρ(r) Time Evolution — Isosurface",
    frame_duration_ms: int = 180,
) -> go.Figure:
    """
    Animated Plotly figure showing temporal evolution of Δρ isosurfaces.

    Parameters
    ----------
    frame_data : list of (step_label, vol_raw) tuples
        Each ``vol_raw`` is Δρ at that time step (not yet normalised).
    isovalue   : isosurface level used for every frame.
    frame_duration_ms : animation speed per frame.
    """
    # Compute global vmax over all frames for consistent normalisation
    all_vals = np.concatenate([v.ravel() for _, v in frame_data])
    global_vmax = float(np.nanpercentile(np.abs(all_vals), 99.5))
    if global_vmax < 1e-30:
        global_vmax = 1.0

    mol_traces = _atom_traces(atoms, positions)
    n_mol = len(mol_traces)

    # --- First frame as initial data ---
    step0, vol0 = frame_data[0]
    vol0_n = np.clip(vol0 / global_vmax, -1.0, 1.0).astype(np.float32)
    t_pos0, t_neg0 = build_isosurface_trace(
        X, Y, Z, vol0_n, isovalue=isovalue, opacity=opacity
    )

    all_traces = mol_traces + [t_pos0, t_neg0]
    traces_pos_idx = n_mol
    traces_neg_idx = n_mol + 1

    # --- Build frames ---
    plotly_frames = []
    for step, vol_raw in frame_data:
        vol_n = np.clip(vol_raw / global_vmax, -1.0, 1.0).astype(np.float32)
        t_p, t_n = build_isosurface_trace(
            X, Y, Z, vol_n, isovalue=isovalue, opacity=opacity,
            name_prefix=f"t={step}"
        )
        plotly_frames.append(go.Frame(
            data=[t_p, t_n],
            name=str(step),
            traces=[traces_pos_idx, traces_neg_idx],
        ))

    layout = _scene_layout(positions, title)
    layout["height"] = 650
    layout["updatemenus"] = [dict(
        type="buttons",
        showactive=False,
        y=0.0, x=0.12,
        xanchor="right",
        yanchor="top",
        pad=dict(t=80, r=10),
        direction="left",
        buttons=[
            dict(
                label="▶ Play",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=frame_duration_ms, redraw=True),
                    fromcurrent=True,
                    transition=dict(duration=0),
                )],
            ),
            dict(
                label="⏸ Pause",
                method="animate",
                args=[[None], dict(
                    frame=dict(duration=0, redraw=False),
                    mode="immediate",
                    transition=dict(duration=0),
                )],
            ),
        ],
    )]
    layout["sliders"] = [dict(
        active=0,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(
            font=dict(size=13, color="#c9d1d9"),
            prefix="Step: ",
            visible=True,
            xanchor="right",
        ),
        transition=dict(duration=0),
        pad=dict(b=10, t=50),
        len=0.88,
        x=0.12,
        y=0,
        steps=[
            dict(
                args=[[str(step)], dict(
                    frame=dict(duration=0, redraw=True),
                    mode="immediate",
                    transition=dict(duration=0),
                )],
                label=str(step),
                method="animate",
            )
            for step, _ in frame_data
        ],
    )]

    fig = go.Figure(data=all_traces, frames=plotly_frames)
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Three-plane contour convenience re-export
# ---------------------------------------------------------------------------

def make_three_panel_contour(
    X, Y, Z, vol,
    title: str = "Δρ(r) — Three Orthogonal Cross-Sections",
) -> go.Figure:
    """Thin wrapper around contour.make_three_plane_figure."""
    return make_three_plane_figure(X, Y, Z, normalise_symmetric(vol), title=title)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _add_projected_atoms(
    fig: go.Figure,
    atoms: list[str],
    positions: np.ndarray,
    slice_axis: str,
):
    """Project atoms onto the 2D slice plane and draw as scatter markers."""
    ax_map = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
    i1, i2 = ax_map[slice_axis]
    for i, sym in enumerate(atoms):
        color = _CPK_COLORS.get(sym, "#00FF00")
        radius = max(8, _CPK_RADII.get(sym, 14) - 2)
        fig.add_trace(go.Scatter(
            x=[positions[i, i1]],
            y=[positions[i, i2]],
            mode="markers+text",
            marker=dict(
                size=radius,
                color=color,
                line=dict(color="white", width=1.2),
            ),
            text=[f"<b>{sym}</b>"],
            textfont=dict(size=10, color="white"),
            textposition="middle center",
            name=f"{sym}{i+1}",
            showlegend=False,
        ))
