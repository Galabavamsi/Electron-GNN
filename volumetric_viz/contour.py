"""
contour.py — 2D cross-sectional contour plots and contour-on-3D-surface overlays.

Scientific context
------------------
Contour maps of Δρ(r) on planes through the molecule (XY, XZ, YZ) are the
standard way to show electron density changes in 2D — widely used in TDDFT
and NBO papers. This module provides:

1. **Planar 2D contour** (``go.Contour``) — the classic publication figure.
2. **3D contour surface** (``go.Surface`` with z=const) — a coloured plane
   embedded in the 3D molecular scene.

Usage
-----
    slice_dict = build_contour_slice(X, Y, Z, vol, axis='z', level=0)
    fig = go.Figure([slice_dict["trace_2d"]])           # 2D standalone
    fig3d = go.Figure([slice_dict["trace_3d_surface"]]) # 3D embedded
"""

from __future__ import annotations

from typing import Literal, Dict, Any

import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Default colorscale (blue-white-red, print-safe)
# ---------------------------------------------------------------------------

_CS_DIV = [
    [0.00, "rgb(49,  54, 149)"],  # deep blue
    [0.20, "rgb(116,173,209)"],   # mid blue
    [0.40, "rgb(224,243,248)"],   # pale blue
    [0.50, "rgb(255,255,255)"],   # white (zero)
    [0.60, "rgb(254,224,182)"],   # pale orange
    [0.80, "rgb(244,109, 67)"],   # mid red
    [1.00, "rgb(165,  0, 38)"],   # deep red
]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_contour_slice(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    vol: np.ndarray,
    axis: Literal["x", "y", "z"] = "z",
    level: float = 0.0,
    n_contours: int = 25,
    colorscale: list | None = None,
    show_colorbar: bool = True,
    label: str = "Δρ",
) -> Dict[str, Any]:
    """
    Extract a planar cross-section and return 2D + 3D contour traces.

    Parameters
    ----------
    X, Y, Z  : (nx, ny, nz) ndarrays
        Coordinate meshgrid (from ``scatter_to_grid``).
    vol      : (nx, ny, nz) ndarray
        Normalised density field.
    axis     : "x" | "y" | "z"
        Normal axis of the cutting plane.
    level    : float
        Position of the cutting plane in a.u. Nearest grid point is used.
    n_contours : int
        Number of contour levels.
    colorscale : list | None
        Plotly colorscale (default: diverging blue-white-red).
    show_colorbar : bool
        Show colour bar on the 2D trace.
    label    : str
        Axis / figure annotation label.

    Returns
    -------
    dict with keys:
        "trace_2d"         : go.Contour  — standalone 2D heatmap+contour
        "trace_3d_surface" : go.Surface  — coloured plane for 3D scene
        "slice_values"     : 2D ndarray  — raw slice data
        "ax1_vals"         : 1D ndarray  — horizontal axis values
        "ax2_vals"         : 1D ndarray  — vertical axis values
        "ax1_label"        : str
        "ax2_label"        : str
        "plane_label"      : str
    """
    cs = colorscale or _CS_DIV

    # --- Extract the slice ---
    if axis == "z":
        idx = int(np.argmin(np.abs(Z[0, 0, :] - level)))
        slice_vals = vol[:, :, idx]              # (nx, ny)
        ax1_vals   = X[:, 0, idx]               # x values
        ax2_vals   = Y[0, :, idx]               # y values
        ax1_label, ax2_label = "x (a.u.)", "y (a.u.)"
        plane_label = f"z ≈ {float(Z[0, 0, idx]):.2f} a.u."

        z_const = np.full_like(slice_vals, Z[0, 0, idx])
        surf_x  = X[:, :, idx]
        surf_y  = Y[:, :, idx]
        surf_z  = z_const

    elif axis == "y":
        idx = int(np.argmin(np.abs(Y[0, :, 0] - level)))
        slice_vals = vol[:, idx, :]
        ax1_vals   = X[:, idx, 0]
        ax2_vals   = Z[0, idx, :]
        ax1_label, ax2_label = "x (a.u.)", "z (a.u.)"
        plane_label = f"y ≈ {float(Y[0, idx, 0]):.2f} a.u."

        y_const = np.full_like(slice_vals, Y[0, idx, 0])
        surf_x  = X[:, idx, :]
        surf_y  = y_const
        surf_z  = Z[:, idx, :]

    else:  # axis == "x"
        idx = int(np.argmin(np.abs(X[:, 0, 0] - level)))
        slice_vals = vol[idx, :, :]
        ax1_vals   = Y[idx, :, 0]
        ax2_vals   = Z[idx, 0, :]
        ax1_label, ax2_label = "y (a.u.)", "z (a.u.)"
        plane_label = f"x ≈ {float(X[idx, 0, 0]):.2f} a.u."

        x_const = np.full_like(slice_vals, X[idx, 0, 0])
        surf_x  = x_const
        surf_y  = Y[idx, :, :]
        surf_z  = Z[idx, :, :]

    vmax = float(np.nanmax(np.abs(slice_vals)))
    if vmax < 1e-12:
        vmax = 1.0

    # --- 2D Contour ---
    trace_2d = go.Contour(
        x=ax1_vals,
        y=ax2_vals,
        z=slice_vals.T,     # Plotly Contour: z[y_idx, x_idx]
        colorscale=cs,
        zmin=-vmax,
        zmax=vmax,
        ncontours=n_contours,
        contours=dict(
            coloring="heatmap",
            showlines=True,
            showlabels=False,
            start=-vmax,
            end=vmax,
            size=2 * vmax / n_contours,
        ),
        colorbar=dict(
            title=dict(text=label, side="right"),
            thickness=16,
            len=0.7,
        ) if show_colorbar else None,
        showscale=show_colorbar,
        name=f"Contour ({plane_label})",
    )

    # --- 3D Surface (coloured plane embedded in molecular scene) ---
    trace_3d = go.Surface(
        x=surf_x,
        y=surf_y,
        z=surf_z,
        surfacecolor=slice_vals,
        colorscale=cs,
        cmin=-vmax,
        cmax=vmax,
        opacity=0.80,
        showscale=False,
        name=f"Contour ({plane_label})",
        showlegend=True,
        hovertemplate=f"{label}=" + "%{customdata:.4f}<extra>{plane_label}</extra>",
        customdata=slice_vals,
    )

    return {
        "trace_2d": trace_2d,
        "trace_3d_surface": trace_3d,
        "slice_values": slice_vals,
        "ax1_vals": ax1_vals,
        "ax2_vals": ax2_vals,
        "ax1_label": ax1_label,
        "ax2_label": ax2_label,
        "plane_label": plane_label,
    }


# ---------------------------------------------------------------------------
# Helper: multi-panel 2D contour figure
# ---------------------------------------------------------------------------

def make_three_plane_figure(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    vol: np.ndarray,
    colorscale: list | None = None,
    n_contours: int = 20,
    label: str = "Δρ(r)",
    title: str = "Electron Density Difference: Cross-Sections",
) -> go.Figure:
    """
    Three-panel figure showing XY, XZ, YZ contour cross-sections through the origin.

    Parameters
    ----------
    X, Y, Z, vol : from ``scatter_to_grid``
    colorscale   : diverging colourscale
    n_contours   : contour level count
    label        : physical quantity label
    title        : figure title

    Returns
    -------
    go.Figure — a 1×3 subplot figure
    """
    from plotly.subplots import make_subplots

    cs = colorscale or _CS_DIV
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "XY plane (z=0)",
            "XZ plane (y=0)",
            "YZ plane (x=0)",
        ],
        shared_yaxes=False,
        horizontal_spacing=0.10,
    )

    configs = [
        ("z", 0.0, 1),
        ("y", 0.0, 2),
        ("x", 0.0, 3),
    ]

    for axis, lv, col in configs:
        slc = build_contour_slice(
            X, Y, Z, vol,
            axis=axis,
            level=lv,
            n_contours=n_contours,
            colorscale=cs,
            show_colorbar=(col == 3),
            label=label,
        )
        t = slc["trace_2d"]
        t.showscale = col == 3
        fig.add_trace(t, row=1, col=col)
        fig.update_xaxes(title_text=slc["ax1_label"], row=1, col=col)
        fig.update_yaxes(title_text=slc["ax2_label"], row=1, col=col)

    _apply_publication_layout(fig, title)
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_publication_layout(fig: go.Figure, title: str):
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            y=0.97,
            xanchor="center",
            font=dict(size=16, family="Inter, Arial, sans-serif"),
        ),
        font=dict(family="Inter, Arial, sans-serif", size=13, color="#e0e0e0"),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=460,
        margin=dict(t=60, l=60, r=30, b=50),
    )
    fig.update_xaxes(
        gridcolor="#30363d",
        zerolinecolor="#58a6ff",
        linecolor="#30363d",
    )
    fig.update_yaxes(
        gridcolor="#30363d",
        zerolinecolor="#58a6ff",
        linecolor="#30363d",
    )
