"""
loader.py — Parse ReSpect output files (.xyz grid + .rho density snapshots).

The .xyz file produced by ReSpect contains two blocks:
  [Atoms] (AU)  — element symbols, atomic numbers, (x,y,z) in a.u.
  [Grid]  (AU)  — index + (x,y,z) for every quadrature grid point in a.u.

The .rho.NNNNN files contain one row per grid point:
  index  density_value

Usage
-----
    loader = RespectLoader("data/raw/ammonia_x")
    loader.load()
    print(loader.atoms)           # list of element symbols
    print(loader.atom_positions)  # (N_atoms, 3) a.u.
    print(loader.grid_points)     # (N_grid, 3) a.u.
    rho0 = loader.load_density(0) # ground-state density  (N_grid,)
    delta = loader.load_delta_rho(5)  # delta rho at step 5 relative to t=0
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class RespectLoader:
    """
    Parse a ReSpect RT-TDDFT run directory.

    Parameters
    ----------
    run_dir : str | Path
        Directory containing ``*.xyz`` and ``*.rho.*`` files.
    """

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.atoms: list[str] = []
        self.atomic_numbers: list[int] = []
        self.atom_positions: Optional[np.ndarray] = None  # (N, 3) a.u.
        self.grid_points: Optional[np.ndarray] = None      # (M, 3) a.u.
        self._rho0: Optional[np.ndarray] = None             # baseline density
        self._rho_files: list[str] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "RespectLoader":
        """Parse the .xyz file and index all .rho snapshot files."""
        xyz_files = sorted(self.run_dir.glob("*.xyz"))
        if not xyz_files:
            raise FileNotFoundError(f"No .xyz file found in {self.run_dir}")
        self._parse_xyz(xyz_files[0])

        # Index density snapshots sorted by step number
        rho_pattern = str(self.run_dir / "*.rho.*")
        self._rho_files = sorted(
            glob.glob(rho_pattern),
            key=lambda p: int(re.search(r"\.(\d+)$", p).group(1))
            if re.search(r"\.(\d+)$", p)
            else 0,
        )
        if not self._rho_files:
            raise FileNotFoundError(
                f"No .rho.* density files found in {self.run_dir}"
            )

        self._rho0 = self._read_rho(self._rho_files[0])
        self._loaded = True
        return self

    @property
    def n_snapshots(self) -> int:
        """Total number of density snapshots available."""
        return len(self._rho_files)

    @property
    def snapshot_steps(self) -> list[int]:
        """List of integer step indices (from filename)."""
        steps = []
        for p in self._rho_files:
            m = re.search(r"\.(\d+)$", p)
            steps.append(int(m.group(1)) if m else 0)
        return steps

    def load_density(self, snapshot_idx: int = 0) -> np.ndarray:
        """
        Return raw electron density ρ(r) at snapshot index.

        Parameters
        ----------
        snapshot_idx : int
            0-based index into the sorted list of .rho files.
        """
        self._ensure_loaded()
        return self._read_rho(self._rho_files[snapshot_idx])

    def load_delta_rho(self, snapshot_idx: int) -> np.ndarray:
        """
        Return Δρ(r) = ρ(t) − ρ(t=0) at snapshot index.
        This highlights electron density *changes* due to the laser pulse.
        """
        self._ensure_loaded()
        rho_t = self._read_rho(self._rho_files[snapshot_idx])
        return rho_t - self._rho0

    def iter_delta_rho(
        self, step: int = 5, max_frames: int = 40
    ):
        """
        Generator yielding (step_label, delta_rho) for animated rendering.

        Parameters
        ----------
        step : int
            Stride (every Nth snapshot).
        max_frames : int
            Maximum number of frames to yield.
        """
        self._ensure_loaded()
        indices = list(range(0, len(self._rho_files), step))[:max_frames]
        steps = self.snapshot_steps
        for idx in indices:
            yield steps[idx], self.load_delta_rho(idx)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call loader.load() first.")

    def _parse_xyz(self, filepath: Path):
        atoms, atomic_numbers, atom_pos, grid_pts = [], [], [], []
        in_atoms = in_grid = False

        with filepath.open() as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("[Atoms]"):
                    in_atoms = True
                    in_grid = False
                    continue
                if line.startswith("[Grid]"):
                    in_atoms = False
                    in_grid = True
                    continue

                if in_atoms and line:
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            atoms.append(parts[0])
                            atomic_numbers.append(int(parts[2]))
                            atom_pos.append(
                                [float(parts[3]), float(parts[4]), float(parts[5])]
                            )
                        except ValueError:
                            pass

                if in_grid and line:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            grid_pts.append(
                                [float(parts[1]), float(parts[2]), float(parts[3])]
                            )
                        except ValueError:
                            pass

        self.atoms = atoms
        self.atomic_numbers = atomic_numbers
        self.atom_positions = (
            np.array(atom_pos, dtype=np.float64) if atom_pos else np.zeros((0, 3))
        )
        self.grid_points = (
            np.array(grid_pts, dtype=np.float64) if grid_pts else np.zeros((0, 3))
        )

    @staticmethod
    def _read_rho(filepath: str) -> np.ndarray:
        values = []
        with open(filepath) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        values.append(float(parts[1]))
                    except ValueError:
                        pass
        return np.array(values, dtype=np.float64)
