"""
isosurface.py — Plotly isosurface and volume traces from 3D volumetric data.

Two rendering modes
-------------------
1. **Isosurface** (``go.Isosurface``)
   Renders discrete surfaces at one or more isovalue levels.
   Best for publication figures — clean, sharp, physically precise surfaces.
   Uses Plotly's built-in marching-cubes implementation.

2. **Volume** (``go.Volume``)
   Renders a semi-transparent volume with an alpha mapped to density magnitude.
   Best for interactive exploration — shows the full 3D distribution.

Usage
-----
    iso_pos, iso_neg = build_isosurface_trace(X, Y, Z, vol, isovalue=0.3)
    vol_trace = build_volume_trace(X, Y, Z, vol_normalised)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Colour palette (professional chemistry conventions)
# ---------------------------------------------------------------------------

# Positive Δρ lobe → blue (gain of electrons)
# Negative Δρ lobe → red (loss of electrons)
_POS_COLOR = "rgba(0, 90, 200, 0.85)"
_NEG_COLOR = "rgba(200, 30, 30, 0.85)"

_COLORSCALE_DIVERGING = [
    [0.0,  "rgb(178, 24, 43)"],   # strong negative → deep red
    [0.25, "rgb(244, 109, 67)"],  # mild negative
    [0.45, "rgb(255, 255, 191)"], # near zero → pale yellow
    [0.5,  "rgb(240, 240, 240)"], # zero → white
    [0.55, "rgb(161, 218, 180)"], # near zero → pale green
    [0.75, "rgb(66, 146, 198)"],  # mild positive
    [1.0,  "rgb(33, 102, 172)"],  # strong positive → deep blue
]


# ---------------------------------------------------------------------------
# Isosurface (marching-cubes style)
# ---------------------------------------------------------------------------

def build_isosurface_trace(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    vol: np.ndarray,
    isovalue: float = 0.20,
    pos_color: str = "royalblue",
    neg_color: str = "crimson",
    opacity: float = 0.60,
    surface_count: int = 1,
    name_prefix: str = "Δρ",
) -> Tuple[go.Isosurface, go.Isosurface]:
    """
    Build two ``go.Isosurface`` traces: positive and negative density lobes.

    Parameters
    ----------
    X, Y, Z : (nx, ny, nz) ndarrays
        Coordinate meshgrid (from ``scatter_to_grid``).
    vol : (nx, ny, nz) ndarray
        Density field.  Should be normalised to [-1, 1] for stable isovalue
        selection (use ``normalise_symmetric`` from *interpolate.py*).
    isovalue : float
        Absolute isosurface level (between 0 and 1 for normalised data).
    pos_color, neg_color : str
        Plotly colour strings for positive and negative lobes.
    opacity : float
        Surface opacity [0–1].
    surface_count : int
        Number of isosurfaces per lobe (1 = outermost only).
    name_prefix : str
        Legend label prefix.

    Returns
    -------
    trace_pos, trace_neg : go.Isosurface
        Plotly traces ready to be added to a ``go.Figure``.
    """
    flat_x = X.ravel().astype(np.float32)
    flat_y = Y.ravel().astype(np.float32)
    flat_z = Z.ravel().astype(np.float32)
    flat_v = vol.ravel().astype(np.float32)

    shared_kw = dict(
        x=flat_x, y=flat_y, z=flat_z, value=flat_v,
        opacity=opacity,
        surface=dict(count=surface_count, fill=1.0),
        showscale=False,
    )

    trace_pos = go.Isosurface(
        **shared_kw,
        isomin=isovalue,
        isomax=1.0,
        colorscale=[[0, pos_color], [1, pos_color]],
        name=f"{name_prefix} > +{isovalue:.2f}",
        showlegend=True,
        caps=dict(x_show=False, y_show=False, z_show=False),
    )

    trace_neg = go.Isosurface(
        **shared_kw,
        isomin=-1.0,
        isomax=-isovalue,
        colorscale=[[0, neg_color], [1, neg_color]],
        name=f"{name_prefix} < −{isovalue:.2f}",
        showlegend=True,
        caps=dict(x_show=False, y_show=False, z_show=False),
    )

    return trace_pos, trace_neg


# ---------------------------------------------------------------------------
# Volume (semi-transparent cloud)
# ---------------------------------------------------------------------------

def build_volume_trace(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    vol: np.ndarray,
    opacity_scale: float = 0.08,
    surface_count: int = 15,
    colorscale: list | None = None,
    name: str = "Volume Δρ",
) -> go.Volume:
    """
    Build a ``go.Volume`` trace for semi-transparent view of the full 3D density.

    Parameters
    ----------
    X, Y, Z : (nx, ny, nz) ndarrays
        Coordinate meshgrid.
    vol : (nx, ny, nz) ndarray
        Normalised density field ([-1, 1]).
    opacity_scale : float
        Maximum voxel opacity (0–1). Lower values = more transparent.
    surface_count : int
        Number of internal isosurfaces.  Higher = richer depth but slower.
    colorscale : list | None
        Plotly colorscale.  Defaults to a professional blue-white-red diverging scale.
    name : str
        Trace name.

    Returns
    -------
    go.Volume
    """
    cs = colorscale or _COLORSCALE_DIVERGING
    flat_v = vol.ravel().astype(np.float32)
    vmax = float(np.nanmax(np.abs(flat_v)))
    if vmax < 1e-12:
        vmax = 1.0

    return go.Volume(
        x=X.ravel().astype(np.float32),
        y=Y.ravel().astype(np.float32),
        z=Z.ravel().astype(np.float32),
        value=flat_v,
        isomin=-vmax,
        isomax=vmax,
        colorscale=cs,
        opacity=opacity_scale,
        surface_count=surface_count,
        colorbar=dict(
            title=dict(text=name, side="right"),
            thickness=16,
            len=0.7,
            x=1.02,
        ),
        name=name,
        showlegend=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
    )
