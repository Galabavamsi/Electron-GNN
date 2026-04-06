# Electron Density Prediction using Graph Neural Networks (GNNs)
## Comprehensive Project Documentation

> **Project:** ML-accelerated quantum spectroscopy via RT-TDDFT + GNN  
> **System:** ReSpect v5.3.0 | Molecule: NH₃ (Ammonia) | HPC: Karolina (IT4I)  
> **Author:** Torsha Moitra  
> **Last Updated:** April 2026  
> **Status:** 🟡 In Progress — Data generation & pipeline design phase

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Physics — What Is Actually Happening](#2-the-physics)
3. [The Data — How It Is Generated](#3-the-data)
4. [The Problem Being Solved](#4-the-problem)
5. [The Paper — Hauge et al. 2023 (Core Algorithm)](#5-the-paper)
6. [The Algorithm — Step by Step](#6-the-algorithm)
7. [The Code Repository — HyQD/absorption-spectrum](#7-the-code-repository)
8. [The GNN Architecture Plan](#8-the-gnn-architecture)
9. [Full Implementation Pipeline](#9-implementation-pipeline)
10. [Your ReSpect Output — Annotated](#10-respect-output-annotated)
11. [Things to Verify and Confirm](#11-things-to-verify)
12. [Things to Explore](#12-things-to-explore)
13. [Open Questions and Risks](#13-open-questions-and-risks)
14. [Scope and Limitations](#14-scope-and-limitations)
15. [Dataset Strategy](#15-dataset-strategy)
16. [Expected Results and Benchmarks](#16-expected-results)
17. [Key References](#17-key-references)
18. [Glossary](#18-glossary)

---

## 1. Project Overview

### Core Mission

Build a Graph Neural Network (GNN) that can **predict the electronic absorption spectrum** of small molecules **without running RT-TDDFT simulations**. The GNN takes a molecular graph (atoms + geometry) as input and predicts the spectral parameters that define the absorption spectrum.

### The Speed Promise

| Method | Time for NH₃ spectrum | Scaling with molecule size |
|--------|----------------------|--------------------------|
| RT-TDDFT (ReSpect) | ~2 minutes (2000 steps, HPC) | O(N³) – catastrophic |
| Hauge et al. fitting | ~30 sec (100 steps needed) | O(N³) still, but 20× fewer steps |
| **GNN (our goal)** | **~milliseconds** | **O(N) – linear** |

### What the GNN Learns

The GNN learns the mapping:

```
Molecular geometry  ──►  Spectral parameters {ω_k, B_k}  ──►  Absorption spectrum S(ω)
(atomic positions)        (frequencies & amplitudes)           (via analytic formula)
```

The spectrum is **not** predicted directly — it is reconstructed analytically from predicted parameters. This is the key physics-informed design choice.

---

## 2. The Physics

### 2.1 What Is RT-TDDFT?

Real-Time Time-Dependent Density Functional Theory (RT-TDDFT) is a quantum mechanical simulation method that:

1. Starts from the **ground state** electron density ρ₀(r) of a molecule
2. Applies a **perturbation** (a kick — electric field pulse)
3. **Propagates** the time-dependent density ρ(r, t) forward in time by numerically integrating the time-dependent Kohn-Sham equations
4. Records the **induced dipole moment** μ(t) at each time step
5. **Fourier transforms** μ(t) → S(ω) to get the absorption spectrum

### 2.2 The Dirac-Delta Kick

In your ReSpect run, the field model is `DELTA` with:
- Amplitude: 1.0 × 10⁻³ a.u.
- Polarization: x-direction (1.0, 0.0, 0.0)
- Time: applied at t = 0

A delta kick is mathematically F(t) = κ · δ(t). It simultaneously excites **all dipole-allowed electronic transitions** of the molecule. This is the most efficient way to get a full absorption spectrum from a single simulation.

**Why weak?** The amplitude κ = 0.001 a.u. is intentionally small so that we remain in the **linear response regime**, where:

```
S(ω) ∝ Im[ FT{ μ(t) } ] / κ
```

If κ were large, nonlinear effects (two-photon absorption, etc.) would contaminate the spectrum.

### 2.3 The Exact Form of the Dipole Signal

After the delta kick and in the absence of any further external field, quantum mechanics in a finite-dimensional Hilbert space guarantees that the induced dipole moment is exactly:

```
μ_ind(t) = Σ_k  B_k · sin(ω_k · t + φ_k) · e^(−γ_k · t)
```

Where:
- **ω_k** = Bohr transition frequencies (excitation energies in a.u.)
- **B_k** = oscillator amplitudes (related to transition dipole moments)
- **φ_k** = phase (for delta kick, φ_k ≈ 0)
- **γ_k** = damping rate (artificial broadening, often set manually)

This is not an approximation — it is the **exact mathematical form** dictated by quantum mechanics. The paper by Hauge et al. exploits this to fit a short trajectory and extrapolate to arbitrary length.

### 2.4 From Dipole to Spectrum

The linear absorption cross-section is obtained via:

```
σ(ω) = (4πω)/(3c·κ) · Im[ μ̃_x(ω) + μ̃_y(ω) + μ̃_z(ω) ]
```

Where μ̃_u(ω) is the Fourier transform of μ_u(t). In your run, only x was perturbed, so only μ̃_x(ω) contributes.

### 2.5 The "Small Delta" Problem for ML

The **total electron density** ρ(r, t) = ρ₀(r) + δρ(r, t) where:
- ρ₀(r) is the large static ground-state density (~10 electrons in NH₃)
- δρ(r, t) is the tiny induced perturbation (~10⁻⁶ of ρ₀)

If you train a neural network on raw ρ(r, t) values:
- The network learns to predict ρ₀(r) perfectly (easy, large signal)
- The network ignores δρ(r, t) completely (hard, tiny signal)
- Result: a model that outputs the correct static density but zero spectral information

**This is why we predict {ω_k, B_k} instead of the density directly.**

---

## 3. The Data

### 3.1 The ReSpect Simulation — Your NH₃ Run

From `ammonia_x.out`:

| Parameter | Value |
|-----------|-------|
| Molecule | NH₃ (ammonia) |
| Basis set | ucc-pVDZ (uncontracted cc-pVDZ) |
| Functional | DFT functional ID 4 (LDA/GGA — verify) |
| Time step | Δt = 0.2 a.u. |
| Total steps | N = 2000 |
| Total time | T = 400 a.u. |
| Field direction | x-polarization |
| Field amplitude | κ = 0.001 a.u. |
| Electrons | 10 (N has 7, 3×H has 1 each) |
| DFT grid points | 10,540 |
| GTO functions | 47 (spherical), 48 (Cartesian) |
| Hamiltonian | 1-component (non-relativistic) |
| HPC | Karolina (IT4I), 4 MPI × 64 OMP threads |
| Wall time | ~2 minutes |

### 3.2 Output Files

| File | Contents | Used for |
|------|----------|----------|
| `ammonia_x.out` | Time steps, energy, μ_x(t), μ_y(t), μ_z(t), GS population | **Primary data source** |
| `rvlab.tdscf.rho.NNNNN` | 3D electron density at grid points, every 5 steps | Density visualization |
| `vgrid.tdscf` | DFT grid coordinates (10,540 points) | Grid for density files |
| `rvlab.tsdcf.xyz` | Atom positions + grid vectors (AU) | Geometry input |

### 3.3 The Dipole Moment Data (from your .out file)

The key column is the **induced electronic dipole moment**:

```
Step 0:    μ_x = 0.000000000000  (before kick)
Step 1:    μ_x = 0.001680440465  (after kick, t=0.2 a.u.)
Step 10:   μ_x = 0.004746528444
...
Step 2000: μ_x = -0.001530158642
```

This oscillating signal is your raw input to the Hauge et al. fitting algorithm.

**Important observations:**
- μ_y ≈ 0 throughout (expected — x-kick gives x-response for symmetric NH₃)
- μ_z is tiny (~10⁻⁶) throughout (expected)
- Ground state population stays at ~0.9999995 (confirms linear regime)
- Energy is nearly constant (fluctuates in 8th decimal place — numerical stability confirmed)

### 3.4 The Visualization Grid (rvlab.tsdcf.xyz)

```
[Atoms] (AU)
N    1    7     0.000000E+00   0.000000E+00   0.000000E+00
H    2    1     0.000000E+00  -0.177200E+01  -0.721119E+00
H    3    1     0.153465E+01   0.886093E+00  -0.721119E+00
H    4    1    -0.153465E+01   0.886093E+00  -0.721119E+00
```

Note: Coordinates are in **Bohr (atomic units)**, not Angstroms. Conversion: 1 Å = 1.88973 a.u.

---

## 4. The Problem

### 4.1 Spectral Resolution Bottleneck

The discrete Fourier transform resolution is:

```
Δω = 2π / (N × Δt)
```

For your run: Δω = 2π / (2000 × 0.2) = 2π / 400 ≈ 0.0157 a.u. ≈ 0.43 eV

This is **coarse**. For molecular spectroscopy, you often need Δω ≈ 0.001 a.u. (0.027 eV). To achieve this with brute-force FFT:

```
N_required = 2π / (Δω_target × Δt) = 2π / (0.001 × 0.2) ≈ 31,400 steps
```

That is **15× more steps** than your current run — 15× more HPC time, 15× more data.

### 4.2 Why ML Cannot Simply Extrapolate μ(t)

Neural networks (including GNNs) are universal interpolators but **unstable extrapolators**. A model trained to predict μ(t) for t ∈ [0, 400] will:
- Interpolate perfectly within the training range
- Produce unphysical divergences or oscillations for t > 400

The paper explicitly warns against this approach. Instead, we use physics-constrained fitting.

### 4.3 Why Standard CNNs/MLPs Fail for Density Prediction

- The density is defined on a **non-uniform irregular 3D grid** (DFT grid) — CNNs expect regular grids
- The density must be **permutation invariant** with respect to atom labelling
- The density must be **rotationally equivariant** — rotate the molecule, density rotates accordingly
- 10,540 grid points × 2001 time steps = ~21 million values per simulation — prohibitively large as direct output

---

## 5. The Paper

### 5.1 Full Citation

```
Eirill Hauge, Håkon Emil Kristiansen, Lukas Konecny, Marius Kadek, 
Michal Repisky, and Thomas Bondo Pedersen

"Cost-Efficient High-Resolution Linear Absorption Spectra through 
Extrapolating the Dipole Moment from Real-Time Time-Dependent 
Electronic-Structure Theory"

J. Chem. Theory Comput. 2023, 19, 7764–7775
DOI: 10.1021/acs.jctc.3c00727
arXiv: 2307.01511
```

### 5.2 Core Idea in One Sentence

Fit the short RT-TDDFT dipole trajectory μ(t) to a sum of sinusoids (whose form is dictated by quantum mechanics), then evaluate the fitted function at arbitrarily long times to achieve high spectral resolution without running more RT-TDDFT steps.

### 5.3 Key Results from the Paper

- Converges with as little as **100 a.u. trajectory length** for simple systems
- Reproduces high-resolution spectra that would normally require **thousands of a.u.**
- Works for RT-TDDFT (ReSpect) and RT-TDCIS (HyQD) — theory-independent
- Provides a built-in **R² error estimate** to detect when fit has converged
- Used a Butterworth **low-pass filter** (cutoff ω_max = 4 a.u.) to focus on valence excitations
- Fitting interval: [0, T_fit] where T_fit = 0.75 × T_total (last 25% used for validation)
- Padé limited to max 5000 data points for computational tractability

### 5.4 Relation to Your Project

| Paper's role | Your project's role |
|-------------|-------------------|
| Generates {ω_k, B_k} from RT-TDDFT output | Uses {ω_k, B_k} as GNN training labels |
| Runs for one molecule at a time | GNN generalizes across molecules |
| Still requires some RT-TDDFT steps | GNN requires zero RT-TDDFT at inference |
| Physics extrapolation | Physics-informed ML |

---

## 6. The Algorithm

### 6.1 Stage 1: Fourier-Padé Frequency Estimation

**Input:** μ(t) time series (2000 data points)  
**Output:** Set of candidate frequencies {ω_p} (overcomplete, possibly noisy)

The Padé approximant represents the z-transform of μ(t) as a rational polynomial P(z)/Q(z). The roots of Q(z) (found via the companion matrix eigenvalue problem) correspond to the oscillation frequencies of the signal.

```python
# Conceptual code
from numpy.linalg import eigvals

def pade_frequencies(dipole_signal, n_pade_points=5000):
    # Build the Padé denominator polynomial Q(z)
    # using n_pade_points from the signal
    # ...
    roots = eigvals(companion_matrix(Q))
    frequencies = -np.imag(np.log(roots)) / dt
    return frequencies[frequencies > 0]  # keep positive frequencies
```

**Key limitation:** O(N²) scaling with number of data points — this is the computational bottleneck of the whole fitting procedure.

### 6.2 Stage 2: K-Means Clustering

**Input:** Noisy overcomplete set {ω_p} (many duplicates and noise)  
**Output:** Clean deduplicated set {ω_k} (physically meaningful frequencies)

The Padé returns hundreds of candidate frequencies, many of which are numerical noise or near-duplicates of real peaks. K-means groups them by proximity and returns the cluster centres as the true frequencies.

```python
from sklearn.cluster import KMeans

def cluster_frequencies(raw_freqs, n_clusters):
    km = KMeans(n_clusters=n_clusters)
    km.fit(raw_freqs.reshape(-1, 1))
    return km.cluster_centers_.flatten()
```

**Challenge:** How do you choose n_clusters? The paper uses the Padé spectrum peaks as a guide. This is a hyperparameter you will need to tune per molecule type.

### 6.3 Stage 3: LASSO Regression for Amplitudes

**Input:** Frequencies {ω_k}, dipole signal μ(t)  
**Output:** Amplitudes {B_k}

With frequencies fixed, the dipole model becomes linear in the coefficients:

```
μ_model(t) = Σ_k  B_k · sin(ω_k · t)
```

This is a **linear regression problem** with the design matrix:

```
Φ[i, k] = sin(ω_k · t_i)
```

LASSO (L1-regularized regression) is used instead of ordinary least squares because:
1. It enforces **sparsity** — many B_k will be exactly zero
2. It prevents overfitting when n_frequencies >> n_physically_real_peaks
3. It is robust to noise in the frequency estimates

```python
from sklearn.linear_model import Lasso

def fit_amplitudes(freqs, times, dipole, alpha=1e-6):
    Phi = np.column_stack([np.sin(w * times) for w in freqs])
    lasso = Lasso(alpha=alpha, fit_intercept=False)
    lasso.fit(Phi, dipole)
    return lasso.coef_
```

**Key parameter:** The LASSO regularization strength α controls sparsity. Too large → too few peaks. Too small → overfitting. Cross-validate on the held-out 25% of the trajectory.

### 6.4 Stage 4: R² Error Estimation and Convergence

**Input:** Fitted model μ_fit(t), true μ(t)  
**Output:** R² score on validation interval [T_fit, T_total]

```python
from sklearn.metrics import r2_score

def check_convergence(model_pred, true_values, threshold=0.99):
    r2 = r2_score(true_values, model_pred)
    converged = r2 > threshold
    return r2, converged
```

A Butterworth low-pass filter is applied before fitting to focus on valence excitations:

```python
from scipy.signal import butter, filtfilt

def lowpass_filter(dipole, dt, omega_max=4.0):
    # omega_max in a.u., convert to Hz for scipy
    nyquist = 0.5 / dt
    cutoff = omega_max / (2 * np.pi) / nyquist
    b, a = butter(N=4, Wn=cutoff, btype='low')
    return filtfilt(b, a, dipole)
```

### 6.5 Stage 5: Spectrum Reconstruction

With {ω_k, B_k} in hand, the absorption spectrum at any resolution is:

```python
def reconstruct_spectrum(freqs, amps, omega_grid, gamma=0.005):
    S = np.zeros_like(omega_grid)
    for w_k, B_k in zip(freqs, amps):
        # Lorentzian peak at each frequency
        S += B_k * gamma / ((omega_grid - w_k)**2 + gamma**2)
    return S
```

---

## 7. The Code Repository

### 7.1 Repository Details

```
URL:         https://github.com/HyQD/absorption-spectrum
Org:         HyQD (Hylleraas Quantum Dynamics, Univ. of Oslo)
Authors:     Same group as the paper (Kristiansen, Pedersen et al.)
Language:    Python
License:     Verify on repo
```

### 7.2 What to Do with the Repo

```bash
# Step 1: Clone
git clone https://github.com/HyQD/absorption-spectrum
cd absorption-spectrum

# Step 2: Install dependencies
pip install numpy scipy scikit-learn matplotlib

# Step 3: Examine the structure
ls -la
cat README.md

# Step 4: Identify the main fitting function
# Look for:
#   - pade_approximant() or similar
#   - frequency_clustering()
#   - lasso_fit() or linear_fit()
#   - error_estimate() or r2_score()

# Step 5: Run on your data
# You need to first parse ammonia_x.out to get the dipole time series
```

### 7.3 Expected Repository Structure (to verify)

```
absorption-spectrum/
├── README.md
├── setup.py or pyproject.toml
├── absorption_spectrum/
│   ├── __init__.py
│   ├── pade.py          # Fourier-Padé implementation
│   ├── fitting.py       # K-means + LASSO pipeline
│   ├── spectrum.py      # Spectrum reconstruction
│   └── utils.py         # I/O, filtering, etc.
├── examples/
│   └── *.py             # Example scripts — run these first!
└── tests/
```

### 7.4 Things to Check in the Repo Immediately

- [ ] Is there a `requirements.txt` or `setup.py`? Install dependencies first.
- [ ] Is there an `examples/` folder? Run the examples before your own data.
- [ ] What file format does it expect as input? (CSV? NumPy array? Custom?)
- [ ] Does it have a command-line interface or only a Python API?
- [ ] Is there documentation for the `n_clusters` parameter for K-means?
- [ ] What is the expected unit system? (a.u.? eV? nm?)

---

## 8. The GNN Architecture

### 8.1 Why a GNN?

A molecule is naturally represented as a graph:
- **Nodes** = atoms (with features: atomic number Z, position r)
- **Edges** = bonds or distance-based connections (with features: distance, bond type)

GNNs are:
- **Permutation invariant** — output doesn't change if you relabel atoms
- **Rotationally equivariant** — if you rotate the molecule, vector outputs rotate accordingly
- **Size flexible** — same model handles molecules of different sizes

### 8.2 Recommended Architecture: EGNN or SchNet

#### Option A: SchNet (simpler, good starting point)

```python
# SchNet-style message passing
h_i^(l+1) = Σ_j  W(|r_i - r_j|) · h_j^(l)
```

- Uses only interatomic distances (not direction)
- Rotationally invariant (not equivariant) — sufficient for scalar outputs like ω_k
- Available in PyTorch Geometric as `SchNet`
- **Limitation:** Cannot predict direction-dependent properties (B_k for specific polarization needs care)

#### Option B: EGNN (recommended)

```python
# EGNN message passing
m_ij = φ_m(h_i, h_j, |r_i - r_j|²)
h_i^(l+1) = φ_h(h_i, Σ_j m_ij)
r_i^(l+1) = r_i + Σ_j (r_i - r_j) · φ_r(m_ij)
```

- Uses distances AND directions
- **E(3)-equivariant** — properly handles rotations and reflections
- Can predict vector outputs (dipole amplitudes) with correct symmetry
- Slightly more complex to implement than SchNet

#### Option C: SE(3)-Transformer or NequIP (most expressive, hardest)

For future exploration — uses spherical harmonics for full equivariance. Overkill for NH₃ but important for heavier elements where relativistic effects matter.

### 8.3 Node and Edge Features

**Node features for atom i:**

| Feature | Description | Dimension |
|---------|-------------|-----------|
| Z_i | Atomic number (one-hot or embedding) | 10 (for {H,C,N,O,F,S,...}) |
| mass_i | Atomic mass | 1 |
| r_i | Cartesian coordinates | 3 |
| electronegativity | Pauling scale | 1 |
| valence_electrons | Number of valence e⁻ | 1 |

**Edge features for pair (i,j):**

| Feature | Description | Dimension |
|---------|-------------|-----------|
| d_ij | Interatomic distance | 1 |
| RBF(d_ij) | Radial basis function expansion | 20-50 |
| unit_vector_ij | Direction vector | 3 |
| bond_type | Single/double/triple/none | 4 (one-hot) |

### 8.4 Output Head

```python
class SpectrumHead(nn.Module):
    def __init__(self, hidden_dim, n_frequencies):
        super().__init__()
        self.freq_head = nn.Linear(hidden_dim, n_frequencies)  # predict ω_k
        self.amp_head  = nn.Linear(hidden_dim, n_frequencies)  # predict B_k
        self.softplus  = nn.Softplus()  # ensure ω_k > 0
    
    def forward(self, global_repr):
        omega = self.softplus(self.freq_head(global_repr))  # frequencies must be positive
        B     = self.amp_head(global_repr)                  # amplitudes can be +/-
        return omega, B
```

The global representation is obtained by **global pooling** (sum or mean) over all node embeddings after message passing.

### 8.5 Loss Function

```python
def spectrum_loss(pred_omega, pred_B, true_omega, true_B, 
                  omega_grid, gamma=0.005):
    
    # Option 1: Direct parameter loss (MSE on frequencies and amplitudes)
    loss_omega = F.mse_loss(pred_omega, true_omega)
    loss_B     = F.mse_loss(pred_B, true_B)
    
    # Option 2: Spectrum-space loss (compare reconstructed spectra)
    S_pred = reconstruct_spectrum(pred_omega, pred_B, omega_grid, gamma)
    S_true = reconstruct_spectrum(true_omega, true_B, omega_grid, gamma)
    loss_spectrum = F.mse_loss(S_pred, S_true)
    
    # Recommended: combine both
    return loss_omega + loss_B + lambda_s * loss_spectrum
```

**Important:** Option 2 (spectrum-space loss) is physically more meaningful because it accounts for the fact that a small error in ω_k at a low-intensity peak matters less than the same error at a high-intensity peak.

---

## 9. Implementation Pipeline

### 9.1 Phase 1: Data Extraction (Week 1-2)

**Goal:** Parse ReSpect output → clean NumPy arrays

```python
# parser.py
import re
import numpy as np

def parse_respect_output(filepath):
    """
    Parse ammonia_x.out_tdscf to extract dipole moment time series.
    
    Returns:
        times:   np.array, shape (N_steps,), in a.u.
        mu_x:    np.array, shape (N_steps,), induced dipole x-component
        mu_y:    np.array, shape (N_steps,)
        mu_z:    np.array, shape (N_steps,)
        energy:  np.array, shape (N_steps,), total energy in Hartree
    """
    pattern = re.compile(
        r'Step EAS:\s+(\d+)\s+([\d.]+)\s+([-\d.E+]+)\s+'
        r'([-\d.E+]+)\s+([-\d.E+]+)\s+([-\d.E+]+)'
    )
    
    steps, times, energies = [], [], []
    mu_x, mu_y, mu_z = [], [], []
    
    with open(filepath) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                times.append(float(m.group(2)))
                energies.append(float(m.group(3)))
                mu_x.append(float(m.group(4)))
                mu_y.append(float(m.group(5)))
                mu_z.append(float(m.group(6)))
    
    return {
        'steps':   np.array(steps),
        'times':   np.array(times),
        'energy':  np.array(energies),
        'mu_x':    np.array(mu_x),
        'mu_y':    np.array(mu_y),
        'mu_z':    np.array(mu_z),
    }
```

**Verify after parsing:**
- Shape should be (2001,) for all arrays (steps 0 through 2000)
- times[0] = 0.0, times[-1] = 400.0
- mu_x[0] ≈ 0.0 (before kick), mu_x[1] > 0 (response to x-kick)

### 9.2 Phase 2: Signal Processing (Week 2-3)

**Goal:** Run HyQD fitting → extract {ω_k, B_k} labels for NH₃

```python
# label_generation.py
from absorption_spectrum import fit_dipole_moment  # HyQD repo

def generate_labels(dipole_data, dt=0.2, omega_max=4.0):
    """
    Run Hauge et al. fitting on parsed dipole data.
    
    Returns:
        frequencies: np.array of ω_k values (in a.u.)
        amplitudes:  np.array of B_k values
        r2_score:    float, fit quality metric
    """
    mu = dipole_data['mu_x']  # x-component for x-kick simulation
    
    # Apply low-pass filter
    mu_filtered = lowpass_filter(mu, dt, omega_max)
    
    # Run Padé + K-means + LASSO pipeline
    result = fit_dipole_moment(
        signal=mu_filtered,
        dt=dt,
        n_pade_points=5000,
        omega_max=omega_max
    )
    
    return result.frequencies, result.amplitudes, result.r2
```

**Sanity checks:**
- R² > 0.99 indicates good fit
- Frequencies should match known NH₃ UV absorption peaks (around 6-7 eV ≈ 0.22-0.26 a.u.)
- Number of significant peaks for NH₃: expect ~5-15 in the valence region

### 9.3 Phase 3: Graph Construction (Week 3-4)

**Goal:** Convert molecular geometry → PyTorch Geometric Data object

```python
# graph_builder.py
import torch
from torch_geometric.data import Data

ATOMIC_NUMBERS = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16}

def molecule_to_graph(xyz_coords, atom_types, frequencies, amplitudes,
                      cutoff_radius=5.0):
    """
    Build a PyTorch Geometric graph from molecular geometry.
    
    Args:
        xyz_coords:  np.array, shape (N_atoms, 3), in Angstroms
        atom_types:  list of str, e.g. ['N', 'H', 'H', 'H']
        frequencies: np.array, GNN target ω_k
        amplitudes:  np.array, GNN target B_k
        cutoff_radius: float, Angstroms, connect atoms within this distance
    
    Returns:
        torch_geometric.data.Data object
    """
    N = len(atom_types)
    
    # Node features: atomic number + position
    Z = torch.tensor([ATOMIC_NUMBERS[a] for a in atom_types], dtype=torch.long)
    pos = torch.tensor(xyz_coords, dtype=torch.float)
    
    # Edge construction: all pairs within cutoff
    edge_index, edge_attr = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                dist = np.linalg.norm(xyz_coords[i] - xyz_coords[j])
                if dist < cutoff_radius:
                    edge_index.append([i, j])
                    edge_attr.append(dist)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    edge_attr  = torch.tensor(edge_attr,  dtype=torch.float).unsqueeze(1)
    
    # Labels
    y_omega = torch.tensor(frequencies, dtype=torch.float)
    y_B     = torch.tensor(amplitudes,  dtype=torch.float)
    
    return Data(
        x=Z.unsqueeze(1).float(),
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_omega=y_omega,
        y_B=y_B
    )
```

### 9.4 Phase 4: GNN Training (Week 4-6)

**Goal:** Train GNN on dataset of molecules → predicts {ω_k, B_k}

```python
# model.py
import torch
import torch.nn as nn
from torch_geometric.nn import SchNet, global_mean_pool

class SpectrumGNN(nn.Module):
    def __init__(self, hidden_channels=128, n_interactions=6, n_frequencies=20):
        super().__init__()
        
        # Backbone: SchNet message passing
        self.backbone = SchNet(
            hidden_channels=hidden_channels,
            num_interactions=n_interactions,
            num_gaussians=50,
            cutoff=5.0
        )
        
        # Output heads
        self.omega_head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.SiLU(),
            nn.Linear(64, n_frequencies),
            nn.Softplus()  # frequencies must be positive
        )
        self.B_head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.SiLU(),
            nn.Linear(64, n_frequencies)
        )
    
    def forward(self, data):
        # Get per-atom embeddings
        h = self.backbone(data.z, data.pos, data.batch)
        
        # Global molecular representation
        h_global = global_mean_pool(h, data.batch)
        
        # Predict spectral parameters
        omega = self.omega_head(h_global)
        B     = self.B_head(h_global)
        
        return omega, B
```

### 9.5 Phase 5: Evaluation (Week 6-8)

**Goal:** Compare predicted spectra to reference RT-TDDFT spectra

Metrics:
1. **Peak position error:** |ω_k_pred − ω_k_true| in eV
2. **Oscillator strength error:** |B_k_pred − B_k_true| / |B_k_true|
3. **Spectral overlap:** ∫ |S_pred(ω) − S_true(ω)| dω / ∫ S_true(ω) dω
4. **Pearson correlation** of S_pred(ω) and S_true(ω) across ω grid

---

## 10. Your ReSpect Output — Annotated

### 10.1 Key Numbers to Remember

```
Ground state energy:          E = -56.4881 Hartree (~constant throughout)
Nuclear dipole moment:        μ_nuc_z = -0.8828 a.u. (static, z-direction)
Total static dipole:          μ_total_z = -0.6474 a.u. ≈ 1.645 Debye
                              (Known NH₃ experimental: ~1.47 Debye — close!)
Gauge origin (z-component):   z_gauge = -0.1281 a.u.
```

### 10.2 Orbital Energies (Final Step, 1c approximation)

```
HOMO: orbital 10 (paired) at -0.20014 Hartree = -5.45 eV
LUMO: orbital 11 (paired) at +0.04043 Hartree = +1.10 eV
HOMO-LUMO gap: ~0.24057 Hartree = ~6.55 eV
```

This gap tells you the lowest expected excitation energy. The UV absorption of NH₃ peaks around ~6.4 eV experimentally — consistent with your orbital gap.

### 10.3 Electron Count Verification

From the density files:
```
Number of electrons from VISUAL (3D): 9.9995677625
```

Expected: 10 electrons (N:7, H:1 × 3).  
Discrepancy of ~0.0004 electrons = DFT grid integration error, acceptable.

### 10.4 Convergence — What the Microiterations Mean

```
Average microiterations per step: 3.857
Convergence threshold: 1e-7
Solver: MAGNUS (4th order Magnus propagator with predictor-corrector)
```

The Magnus propagator solves: U(t+Δt) = exp(−iΩ) · U(t) self-consistently. The 3.857 average means most steps took 4 iterations to converge — good. If this exceeded 8, the simulation was struggling.

---

## 11. Things to Verify and Confirm

### 11.1 Physics Verification

- [ ] **Confirm DFT functional:** The output says "DFT functional ID 4" — verify what functional this is in ReSpect's convention. This affects accuracy and reproducibility.
- [ ] **Confirm Hamiltonian level:** The output says "1c orbitals" in the final energies — verify if this is 1-component (non-relativistic Schrödinger) or 2-component. NH₃ has no heavy atoms so relativistic effects are negligible, but important for documentation.
- [ ] **Verify the dipole moment convention:** The output gives "induced electronic dipole moment" — confirm this excludes the nuclear contribution (it should, for computing the spectrum).
- [ ] **Check gauge origin independence:** The output states "Induced electronic dipole moment (always)" is gauge-origin independent — verify this is correct for your observable.
- [ ] **Confirm units:** All dipole values in the `.out` file are in atomic units (e·a₀). For the spectrum calculation, confirm the kick amplitude κ = 0.001 a.u. is in the same unit system.
- [ ] **Cross-check NH₃ spectrum:** The known first absorption band of NH₃ is the Ã←X̃ transition at ~5.9-6.4 eV. After running the Hauge fitting, verify your extracted ω_k matches this.

### 11.2 Code Verification

- [ ] **HyQD repo compatibility:** Verify the repo works with your Python version (≥3.8 recommended).
- [ ] **ReSpect output format:** Confirm the regex pattern in the parser correctly handles the exact spacing/formatting of your `.out` file (check for edge cases at step 0, checkpointing lines, etc.).
- [ ] **Unit consistency in HyQD code:** Confirm whether the HyQD repo expects dipole in a.u. or SI, and time in a.u. or femtoseconds. Your data is in a.u. (t in a.u., μ in e·a₀).
- [ ] **K-means n_clusters selection:** Verify how the HyQD repo decides the number of clusters — is it automatic or a user parameter?
- [ ] **LASSO alpha tuning:** Verify the regularization strength used in the paper for NH₃-sized systems.
- [ ] **Low-pass filter cutoff:** Confirm ω_max = 4.0 a.u. is appropriate. In eV: 4.0 a.u. × 27.211 eV/a.u. ≈ 109 eV — this is very broad, so it includes all valence excitations but excludes core excitations (which would be > 400 eV for N). This is correct for UV/visible spectroscopy.

### 11.3 GNN Verification

- [ ] **Equivariance test:** After training, rotate a molecule 90° and verify the predicted frequencies are identical (frequencies are scalars, must be rotation-invariant). Amplitudes depend on polarization direction and require more careful testing.
- [ ] **Size consistency:** Verify that adding a distant "ghost" atom does not change the prediction (test cutoff radius correctness).
- [ ] **Permutation invariance:** Verify that reordering the atoms in the input graph gives identical output.
- [ ] **Overfitting check:** Monitor training vs. validation loss curves carefully — with small datasets, overfitting is the main risk.

### 11.4 Data Verification

- [ ] **Parsing accuracy:** After parsing `ammonia_x.out`, verify that `len(mu_x) == 2001` (steps 0 to 2000 inclusive).
- [ ] **Signal integrity:** Plot μ_x(t) and verify it shows oscillatory behavior decaying toward zero (or remaining oscillatory if no damping was applied by ReSpect).
- [ ] **No NaN/Inf values:** Check `np.isnan(mu_x).any()` before fitting.
- [ ] **Checkpoint lines don't break parsing:** The `.out` file has extra lines at steps 0, 5, 100, etc. (FILE: and Number of electrons lines). Confirm your regex skips these correctly.

---

## 12. Things to Explore

### 12.1 Near-term Explorations (this semester)

1. **Different molecules:** Run ReSpect for H₂O, CH₄, CO₂, C₂H₂ (acetylene). These span different symmetries and electronic structures. Compare GNN predictions.

2. **Conformational effects:** For NH₃, perturb the geometry slightly (stretch N-H bond, change H-N-H angle). Does the GNN capture the geometry-dependence of the spectrum?

3. **Different polarizations:** Your current run used x-kick only. Run y and z kicks and verify the isotropic spectrum (average of all three).

4. **Trajectory length vs. fit quality:** Truncate your 2000-step trajectory to 100, 200, 500, 1000 steps and run the Hauge fitting on each. Plot R² vs. trajectory length. This validates the paper's claim that 100 a.u. ≈ 500 steps (at Δt=0.2) may be sufficient.

5. **Number of significant spectral peaks:** For each molecule, how many non-negligible B_k values does the LASSO return? This determines the output dimension of the GNN.

### 12.2 Medium-term Explorations (3-6 months)

6. **Transfer learning:** Pre-train on many small molecules (QM9 dataset), fine-tune on your RT-TDDFT targets. This could dramatically reduce the RT-TDDFT dataset size needed.

7. **Uncertainty quantification:** Use MC dropout or deep ensembles to give error bars on predicted spectra. This is critical for scientific use.

8. **Relativistic effects:** Your supervisor (Torsha Moitra) works on relativistic RT-TDDFT. Explore whether 4-component (Dirac) data gives different {ω_k, B_k} than 1-component for heavier atoms (Br, I). GNN could predict relativistic corrections.

9. **Core excitations:** The current setup uses a low-pass filter at ω_max = 4 a.u. Explore removing the filter and predicting core-level excitations (XANES/NEXAFS spectroscopy region).

10. **Charge density as auxiliary target:** Use the density snapshots from `rvlab.tdscf.rho.NNNNN` as auxiliary targets in a multi-task learning setup to regularize the GNN.

### 12.3 Long-term Research Directions (6-12 months)

11. **Drug-like molecules:** Test on drug candidates from PubChem. These are where RT-TDDFT is currently infeasible — validating generalization here would be the main scientific contribution.

12. **Photovoltaic materials:** Predict UV-Vis spectra of chromophores used in solar cells (e.g., porphyrins, BODIPY dyes).

13. **Non-linear spectra:** The current framework is for linear absorption only. Extend to two-photon absorption or transient absorption spectroscopy (already in your supervisor's expertise).

14. **Active learning:** Train an initial GNN, use it to identify which molecules the model is most uncertain about, run RT-TDDFT only for those, retrain. Dramatically reduces data generation cost.

---

## 13. Open Questions and Risks

### 13.1 Scientific Open Questions

| Question | Why It Matters | How to Resolve |
|----------|----------------|----------------|
| How many RT-TDDFT simulations are needed to train a reliable GNN? | Dataset generation cost scales with this number | Start with 10-50 molecules, plot learning curves |
| Does n_frequencies need to be fixed across molecules? | Different molecules have different numbers of peaks | Explore variable-length output or maximum padding |
| Can a GNN trained on 1-component data generalize to 4-component (relativistic) targets? | Relativistic effects matter for Br, I, heavy metals | Compare 1c vs 4c NH₃ spectra as baseline |
| What cutoff radius is optimal for the molecular graph? | Too small misses long-range electrostatics; too large creates dense graphs | Ablation study: vary cutoff from 3-8 Å |
| Are {ω_k, B_k} the right intermediate representation? | Alternative: predict S(ω) directly on a fixed frequency grid | Compare both approaches |

### 13.2 Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| HyQD repo produces inconsistent labels across molecules | High | Manually verify against ReSpect's `spectrum.py` utility |
| Too few training molecules for GNN to generalize | High | Use data augmentation (rotations, reflections) |
| LASSO alpha hyperparameter gives different n_peaks per molecule | Medium | Normalize to fixed output vector with zero-padding |
| GNN overfits to NH₃ geometry | Medium | Include diverse training set from the start |
| K-means clustering misidentifies noise as real peaks | Medium | Tune n_clusters using cross-validation |

### 13.3 Methodological Risks

| Risk | Mitigation |
|------|------------|
| Hauge fitting converges to wrong local minimum | Run fitting with multiple random seeds; check R² |
| Phase ambiguity in sinusoid fitting (sin vs cos) | Document the convention used by HyQD repo |
| GNN predicts positive amplitudes when true amplitude is negative | Remove Softplus on B_k head; allow signed amplitudes |

---

## 14. Scope and Limitations

### 14.1 What This Project Can Do

- Predict linear electronic absorption spectra of **closed-shell, neutral, light-atom molecules** in the gas phase
- Handle molecules with atoms: H, C, N, O, F (and potentially S, Cl with more data)
- Produce spectra in the UV-Visible range (0-10 eV / 0-0.37 a.u.)
- Generalize to molecules not seen during training (with sufficient training diversity)

### 14.2 What This Project Cannot Do (in v1)

- **Open-shell molecules** (radicals, diradicals) — requires spin-polarized RT-TDDFT
- **Charged molecules** (anions, cations) — different electronic structure
- **Condensed phase / solvation effects** — RT-TDDFT requires PCM or QM/MM for this
- **Strong-field / nonlinear spectra** — the linear response assumption breaks down
- **Heavy atom molecules** (containing Br, I, Pt, etc.) — relativistic effects need 4-component Hamiltonian
- **Excited-state absorption** (transient absorption spectroscopy) — different formalism
- **Vibronic structure** — this model captures only electronic excitations, not nuclear motion

### 14.3 Scalability of the Approach

| Molecule size | RT-TDDFT cost | GNN inference cost | Status |
|---------------|---------------|-------------------|--------|
| NH₃ (4 atoms) | ~2 min | ~1 ms | ✅ Working |
| Benzene (12 atoms) | ~30 min | ~1 ms | 🟡 Plan to test |
| Caffeine (24 atoms) | ~6 hours | ~1 ms | 🟡 Stretch goal |
| BODIPY dye (30+ atoms) | >24 hours | ~1 ms | 🔴 Future work |

---

## 15. Dataset Strategy

### 15.1 Minimum Viable Dataset

For a first working GNN, target:

- **20-50 molecules** minimum
- Diverse: vary atom types, bond orders, ring systems, symmetry
- Include geometric variations (different bond lengths/angles of same molecule)

**Suggested initial molecule set:**

| Molecule | Formula | Reason to include |
|----------|---------|-------------------|
| Ammonia | NH₃ | Your current molecule, baseline |
| Water | H₂O | Simplest oxygen compound |
| Methane | CH₄ | Carbon, tetrahedral, no dipole |
| Formaldehyde | CH₂O | Carbonyl, UV absorption ~3.5 eV |
| Ethylene | C₂H₄ | π-system, low-lying excitation |
| HCN | HCN | Triple bond, linear molecule |
| Acetylene | C₂H₂ | Linear, high symmetry |
| Hydrogen fluoride | HF | Most electronegative, small |
| Methanol | CH₃OH | Oxygen + carbon |
| Nitrogen | N₂ | Homonuclear, no permanent dipole |

### 15.2 Data Augmentation

For each molecule, you can generate additional training points for free:
- **Random rotations** (3D rotations of geometry — labels {ω_k} unchanged, {B_k} components rotate)
- **Isotope variants** (different masses: NH₃ vs ND₃) — changes vibrational but not electronic spectrum
- **Geometry perturbations** (small distortions ±0.05 Å) — probes geometry-sensitivity

### 15.3 Data Format

Standardize all data in an HDF5 file:

```
dataset.h5
├── molecules/
│   ├── NH3/
│   │   ├── geometry       (4, 3) float32   # atom positions in Angstrom
│   │   ├── atomic_numbers (4,)   int32     # [7, 1, 1, 1]
│   │   ├── frequencies    (K,)   float32   # ω_k in a.u.
│   │   ├── amplitudes     (K,)   float32   # B_k in a.u.
│   │   ├── r2_score       ()     float32   # fit quality
│   │   └── metadata       attrs            # functional, basis, dt, T
│   ├── H2O/
│   └── ...
```

---

## 16. Expected Results and Benchmarks

### 16.1 Signal Processing Stage (Phase 2)

After running Hauge fitting on your NH₃ data, expect:
- **R² ≥ 0.99** (400 a.u. trajectory is more than enough for NH₃)
- **~5-15 non-zero amplitudes** from LASSO
- **Primary peak around 6.0-6.5 eV** (≈ 0.22-0.24 a.u.) — the A-band of NH₃
- **Possible weak peaks at higher energies** (Rydberg transitions)

### 16.2 GNN Training Stage (Phase 4)

Reasonable targets for a proof-of-concept model:
- **Peak position error < 0.2 eV** (comparable to typical TDDFT accuracy)
- **Oscillator strength error < 20%** for major peaks
- **Training time:** < 1 hour for 50 molecules on a GPU
- **Inference time:** < 10 ms per molecule

### 16.3 Comparison Baselines

Always compare against:
1. **Simple FFT spectrum** from your 400 a.u. trajectory (coarse resolution)
2. **Hauge-extrapolated spectrum** (high resolution, but still requires RT-TDDFT)
3. **Linear-response TDDFT** (LR-TDDFT) from standard quantum chemistry codes like ORCA or Gaussian

---

## 17. Key References

### Primary References

```
[1] Hauge, E. et al.
    "Cost-Efficient High-Resolution Linear Absorption Spectra through Extrapolating 
    the Dipole Moment from Real-Time Time-Dependent Electronic-Structure Theory"
    J. Chem. Theory Comput. 2023, 19, 7764–7775
    DOI: 10.1021/acs.jctc.3c00727
    arXiv: 2307.01511
    → Core algorithm: the Padé+LASSO fitting pipeline

[2] Repisky, M. et al.
    "Excitation Energies from Real-Time Propagation of the Four-Component 
    Dirac–Kohn–Sham Equation"
    J. Chem. Theory Comput. 2015, 11, 980–991
    DOI: 10.1021/ct501078d
    → Foundation of RT-TDDFT in ReSpect

[3] Repisky, M. et al.
    "ReSpect: Relativistic Spectroscopy DFT Program Package"
    J. Chem. Phys. 2020, 152, 184101
    DOI: 10.1063/5.0005094
    → Full ReSpect program reference

[4] Moitra, T. et al.
    "Accurate Relativistic Real-Time TDDFT for Valence and Core 
    Attosecond Transient Absorption Spectroscopy"
    J. Phys. Chem. Lett. 2023, 14, 1714–1724
    DOI: 10.1021/acs.jpclett.2c03599
    → Your supervisor's work on relativistic RT-TDDFT
```

### GNN Architecture References

```
[5] Schütt, K.T. et al.
    "SchNet: A Continuous-Filter Convolutional Neural Network for 
    Modeling Quantum Interactions"
    NeurIPS 2017
    → SchNet architecture

[6] Satorras, V.G. et al.
    "E(n) Equivariant Graph Neural Networks"
    ICML 2021
    → EGNN architecture (recommended for this project)

[7] Batzner, S. et al.
    "E(3)-equivariant graph neural networks for data-efficient and 
    accurate interatomic potentials"
    Nat. Commun. 2022, 13, 2453
    → NequIP, most expressive equivariant GNN

[8] Xu, K. et al.
    "How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks"
    ICLR 2021
    arXiv: 2009.11848
    → Cited IN the Hauge paper — explains why NNs can't extrapolate μ(t)
```

### Code Repositories

```
HyQD/absorption-spectrum:  https://github.com/HyQD/absorption-spectrum
PyTorch Geometric:         https://pytorch-geometric.readthedocs.io
ReSpect program:           http://www.respectprogram.org
SchNet (PyG):              torch_geometric.nn.models.SchNet
EGNN (reference impl):     https://github.com/vgsatorras/egnn
```

---

## 18. Glossary

| Term | Definition |
|------|-----------|
| **RT-TDDFT** | Real-Time Time-Dependent Density Functional Theory — quantum simulation that propagates electron density in time |
| **δ-kick** | Dirac-delta electric field pulse that excites all electronic transitions simultaneously |
| **μ(t)** | Time-dependent electric dipole moment — the key observable extracted from RT-TDDFT |
| **ω_k** | Transition frequency (excitation energy) of the k-th electronic state, in atomic units |
| **B_k** | Amplitude of the k-th transition — related to oscillator strength |
| **S(ω)** | Absorption spectrum (absorption cross-section as a function of frequency) |
| **Padé approximant** | Rational polynomial approximation to a time series, used to extract frequencies |
| **LASSO** | L1-regularized linear regression that enforces sparsity in the fitted coefficients |
| **K-means** | Unsupervised clustering algorithm used to deduplicate Padé frequency estimates |
| **GNN** | Graph Neural Network — neural network operating on graph-structured data |
| **EGNN** | E(n) Equivariant GNN — GNN that respects 3D rotational and translational symmetry |
| **Equivariance** | Property where output transforms predictably when input is rotated/reflected |
| **a.u.** | Atomic units — natural unit system for quantum chemistry (1 a.u. energy = 27.211 eV) |
| **Bohr** | Atomic unit of length = 0.5292 Å (used in ReSpect coordinates) |
| **Hartree** | Atomic unit of energy = 27.211 eV |
| **HPC** | High-Performance Computing — the supercomputer cluster (Karolina, IT4I) |
| **Linear response** | Regime where perturbation is small enough that response is proportional to it |
| **Oscillator strength** | Dimensionless measure of transition probability — proportional to |B_k|² |
| **R²** | Coefficient of determination — measures how well the fitted model matches data (1.0 = perfect) |
| **Magnus propagator** | Numerical method for time-propagating quantum systems, used in ReSpect |
| **DFT grid** | Non-uniform 3D grid of integration points around atoms used in DFT calculations |

---

*This document will be updated as the project progresses. Next milestone: Parse ammonia_x.out → run HyQD fitting → extract first {ω_k, B_k} label set.*
## Phase 6: Full Training and Inference Results (Completed)
- Extracted peaks for both Ammonia (`nh3_x`) and Water (`water_x`) datasets. Ammonia yielded 39 peaks, Water yielded 55 peaks.
- Successfully trained the `SpectralEquivariantGNN` on both datasets for 50 epochs.
- The model smoothly learned from the $E(3)$ equivariant vector representations, using the bipartite matching loss to assign predicted peaks to true targets.
- **Results:** 
  - Bipartite Loss decreased from ~30.6 down to **0.2532**.
  - Spectrum Loss decreased from ~0.43 down to **0.0002**.
  - A comprehensive plot has been generated at `results/loss_curves.png`.
- The dashboard supports visual comparison of true vs. predicted spectra, completing the end-to-end workflow!
