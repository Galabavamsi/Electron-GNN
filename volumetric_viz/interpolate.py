"""
interpolate.py — Scatter-to-grid interpolation for unstructured ReSpect density data.

The ReSpect RT-TDDFT code uses a radial quadrature grid centered on each atom,
resulting in ~10 k unstructured 3D points. For isosurface / volume rendering,
we need the data on a regular Cartesian (x, y, z) grid.

This module converts that scatter data to a regular grid via linear barycentric
interpolation (scipy.interpolate.LinearNDInterpolator) with nearest-neighbour
fallback for extrapolated regions.

Usage
-----
    X, Y, Z, vol = scatter_to_grid(grid_points, delta_rho, resolution=40)
    # X, Y, Z are (nx, ny, nz) coordinate arrays
    # vol is (nx, ny, nz) density array, ready for Plotly Volume / marching cubes
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def scatter_to_grid(
    points: np.ndarray,
    values: np.ndarray,
    resolution: int = 40,
    padding: float = 0.5,
    use_fallback: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate unstructured 3D scatter data onto a regular Cartesian grid.

    Parameters
    ----------
    points : (N, 3) array
        Unstructured 3D coordinates in a.u.
    values : (N,) array
        Scalar field values at each point (e.g. Δρ or ρ).
    resolution : int
        Number of grid points along the longest axis. Other axes scale
        proportionally to maintain ~cubic voxels.
    padding : float
        Extra boundary around the point cloud in a.u. to avoid edge artefacts.
    use_fallback : bool
        Fill NaN pixels that fall outside the convex hull using nearest-neighbour
        interpolation. Strongly recommended for volumetric rendering.

    Returns
    -------
    X, Y, Z : (nx, ny, nz) ndarrays
        Coordinate meshgrid.
    vol : (nx, ny, nz) ndarray
        Interpolated density on the regular Cartesian grid.
    """
    x_min, y_min, z_min = points.min(axis=0) - padding
    x_max, y_max, z_max = points.max(axis=0) + padding

    # Build proportional resolution per axis
    ranges = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
    max_range = ranges.max()
    nx = max(4, int(np.ceil(resolution * ranges[0] / max_range)))
    ny = max(4, int(np.ceil(resolution * ranges[1] / max_range)))
    nz = max(4, int(np.ceil(resolution * ranges[2] / max_range)))

    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_min, y_max, ny)
    zi = np.linspace(z_min, z_max, nz)

    X, Y, Z = np.meshgrid(xi, yi, zi, indexing="ij")

    # Build interpolator on the full unstructured point set
    interp_linear = LinearNDInterpolator(points, values)
    vol = interp_linear(X, Y, Z)  # NaN outside convex hull

    if use_fallback and np.any(np.isnan(vol)):
        interp_nn = NearestNDInterpolator(points, values)
        nan_mask = np.isnan(vol)
        vol[nan_mask] = interp_nn(X[nan_mask], Y[nan_mask], Z[nan_mask])

    return X, Y, Z, vol.astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience: normalise / symmetrise for display
# ---------------------------------------------------------------------------

def normalise_symmetric(vol: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """
    Clip volume at ±percentile and normalise to [-1, 1] for display.
    Preserves sign (positive / negative density difference).
    """
    vmax = np.nanpercentile(np.abs(vol), percentile)
    if vmax < 1e-30:
        return np.zeros_like(vol)
    return np.clip(vol / vmax, -1.0, 1.0)


def positive_lobe(vol: np.ndarray) -> np.ndarray:
    """Return only the positive lobe (gain of electron density), zeroed elsewhere."""
    return np.where(vol > 0, vol, 0.0)


def negative_lobe(vol: np.ndarray) -> np.ndarray:
    """Return only the negative lobe (loss of electron density), zeroed elsewhere."""
    return np.where(vol < 0, vol, 0.0)
