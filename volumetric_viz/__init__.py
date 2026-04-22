"""
volumetric_viz — Isosurface and Contour Surface Visualization for RT-TDDFT Electron Density.

Modules
-------
loader      : parse ReSpect .xyz grid files and .rho density snapshots
interpolate : scatter → regular 3D Cartesian grid (RBF / linear interp)
isosurface  : marching-cubes isosurface + Plotly Volume rendering
contour     : 2D planar contour slices from the 3D volume
render      : high-level Plotly figure builders (isosurface, contour, combo)
app         : standalone Streamlit dashboard for interactive exploration
"""

from .loader import RespectLoader
from .interpolate import scatter_to_grid
from .isosurface import build_isosurface_trace, build_volume_trace
from .contour import build_contour_slice
from .render import (
    make_isosurface_figure,
    make_contour_figure,
    make_combined_figure,
    make_animated_isosurface,
)

__all__ = [
    "RespectLoader",
    "scatter_to_grid",
    "build_isosurface_trace",
    "build_volume_trace",
    "build_contour_slice",
    "make_isosurface_figure",
    "make_contour_figure",
    "make_combined_figure",
    "make_animated_isosurface",
]
