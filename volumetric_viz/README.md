# `volumetric_viz` — Isosurface & Contour Surface Visualization

> RT-TDDFT electron density visualization using isosurfaces and contour cross-sections.

---

## What This Does

This module visualizes the **time-dependent electron density difference**

```
Δρ(r, t) = ρ(r, t) − ρ(r, t=0)
```

computed by a real-time TDDFT (ReSpect) run. The positive lobe (blue) shows  
where electrons **accumulated** and the negative lobe (red) shows where they  
**depleted** in response to the applied laser pulse.

### Visualization Modes

| Mode | Description |
|---|---|
| **Isosurface** | Marching-cubes ±Δρ lobes at user-specified isovalue |
| **Volume rendering** | Full 3D semi-transparent density cloud |
| **Contour slices (2D)** | XY / XZ / YZ planar cross-sections |
| **Combined 3D** | Isosurface + embedded contour planes in one scene |
| **Animated** | Time-evolution of Δρ isosurfaces across all snapshots |

---

## Quick Start

```bash
# From the project root, activate your venv
source .venv/bin/activate

# Launch the standalone viz dashboard
streamlit run volumetric_viz/app.py
```

---

## Python API

```python
import sys
sys.path.insert(0, "/path/to/Electron-GNN")

from volumetric_viz.loader import RespectLoader
from volumetric_viz.interpolate import scatter_to_grid, normalise_symmetric
from volumetric_viz.render import (
    make_isosurface_figure,
    make_contour_figure,
    make_combined_figure,
    make_animated_isosurface,
)

# 1 — Load data
loader = RespectLoader("data/raw/ammonia_x").load()

# 2 — Extract Δρ at snapshot 8
delta_rho = loader.load_delta_rho(snapshot_idx=8)

# 3 — Interpolate to regular 3D Cartesian grid
X, Y, Z, vol = scatter_to_grid(loader.grid_points, delta_rho, resolution=40)

# 4A — Isosurface figure
fig = make_isosurface_figure(
    X, Y, Z, vol,
    atoms=loader.atoms,
    positions=loader.atom_positions,
    isovalue=0.25,  # 25% of max absolute value
    opacity=0.55,
)
fig.show()

# 4B — 2D contour cross-section
fig2d = make_contour_figure(X, Y, Z, vol, axis="z", level=0.0)
fig2d.show()

# 4C — Combined 3D
fig3d = make_combined_figure(
    X, Y, Z, vol,
    atoms=loader.atoms,
    positions=loader.atom_positions,
    isovalue=0.20,
)
fig3d.show()

# 4D — Animated time evolution
frame_data = list(loader.iter_delta_rho(step=5, max_frames=20))
# Interpolate each frame first
from volumetric_viz.interpolate import scatter_to_grid
grid_frames = [
    (step, scatter_to_grid(loader.grid_points, drho, resolution=32)[3])
    for step, drho in frame_data
]
fig_anim = make_animated_isosurface(
    X, Y, Z,
    atoms=loader.atoms,
    positions=loader.atom_positions,
    frame_data=grid_frames,
    isovalue=0.25,
)
fig_anim.show()
```

---

## Module Structure

```
volumetric_viz/
├── __init__.py       Public API exports
├── loader.py         Parse ReSpect .xyz + .rho files → numpy
├── interpolate.py    Unstructured scatter → regular Cartesian grid
├── isosurface.py     go.Isosurface + go.Volume Plotly traces
├── contour.py        go.Contour (2D) + go.Surface (3D) slice traces
├── render.py         High-level figure builders (all four modes)
├── app.py            Standalone Streamlit dashboard
└── README.md         This file
```

---

## Data Pipeline

```
data/raw/{mol}/
  rvlab.tdscf.xyz       → RespectLoader._parse_xyz()
                            → atoms[], atom_positions (N,3), grid_points (M,3) [a.u.]

  rvlab.tdscf.rho.00000 → baseline ρ₀
  rvlab.tdscf.rho.00005 → ρ(t₁)
  ...                   → Δρ = ρ(tₙ) − ρ₀
        ↓
  scatter_to_grid()
    LinearNDInterpolator (SciPy)  → regular (nx, ny, nz) Cartesian volume
    NearestNDInterpolator fallback for extrapolated regions
        ↓
  normalise_symmetric()           → rescale to [−1, 1] for stable isovalue UI
        ↓
  Plotly traces (isosurface / volume / contour surface)
        ↓
  Streamlit app / standalone figure
```

---

## Physical Colour Conventions

| Colour | Meaning |
|---|---|
| 🔵 Blue | Δρ > 0 — electron accumulation |
| 🔴 Red  | Δρ < 0 — electron depletion |
| ⬜ White | Δρ ≈ 0 — no change |

Atom colours follow **CPK convention**: H=silver, C=grey, N=blue, O=red.

---

## Notes

- The ReSpect grid is a radial quadrature grid (~10 540 unstructured points for
  these ammonia/water calculations). Interpolation to a Cartesian grid is needed
  for marching-cubes isosurfaces.
- At **resolution=36** (default), the Cartesian grid is ~36³ ≈ 46 k voxels.
  Increase to 52 for publication-quality figures (slower).
- The `isovalue` slider in the UI is expressed as a **fraction of the maximum
  |Δρ|** so it stays meaningful across time steps.
- No additional pip packages required beyond what is already in `requirements.txt`.
