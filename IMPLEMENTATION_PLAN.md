# Extensive Implementation Plan
> **ML-Accelerated Quantum Spectroscopy via RT-TDDFT + GNN**

This living document dictates the step-by-step technical pathway to reaching the end-to-end robust Machine Learning and Quantum Pipeline. We will mark off phases as they finish.

---

## 🟢 Phase 1: Environment & Architecture Framework (COMPLETED)
- [x] Structure workspace directories (`utils/`, `scripts/`, `models/`, `train/`, `dashboard/`).
- [x] Draft advanced physics-informed theoretical document.
- [x] Implement spectrum plots and diagnostic utilities (Hungarian Matching, Complex Poles, Overlap Integral).
- [x] Build the interactive Streamlit dashboard for real-time monitoring.

## 🟡 Phase 2: Data Extraction & Dataset Generation (NEXT)
The objective here is turning your ReSpect output logs into usable ML targets: $X$ (Graphs) and $Y$ (Frequencies \& Amplitudes sets).

- [x] **Step 2.1:** Create `scripts/parser.py`.
  - Parse the raw `ammonia_x.out` files line-by-line to extract the time steps ($t$) and induced dipole moment ($\mu(t)$).
  - Extract spatial coordinates ($r_i$) and elements ($Z_i$) from geometry block to construct PyTorch Geometric `Data` graphs.
- [x] **Step 2.2:** Clone the `HyQD/absorption-spectrum` GitHub repository to `lib/` or install it locally.
- [x] **Step 2.3:** Create `scripts/extract_peaks.py`.
  - Pass the parsed ReSpect $\mu(t)$ trajectory through Hauge's pipeline: Butterworth Low-Pass $\rightarrow$ Fourier-Padé (roots) $\rightarrow$ K-Means Clustering $\rightarrow$ LASSO Regression.
  - Produce a serialized `.pt` dataset of pairs: `Molecular Graph $\rightarrow$ {ω_k, B_k}`.

## � Phase 3: The Equivariant GNN Architecture (COMPLETED)
Constructing the $E(3)$ Equivariant model via MACE or PaiNN to learn the mapping from molecules to spectral signatures.

- [x] **Step 3.1:** Scaffold PyTorch geometric definitions (`models/molecule_graph.py`). Set up atom node embeddings (one-hot element encoding) and edge features (distances/radial basis functions).
- [x] **Step 3.2:** Define the deep network (`models/mace_net.py`).
  - Configure the message passing layers for Scalar ($h_i$) and Vector ($v_i$) branches.
  - Global pooling (sum) over nodes.
- [x] **Step 3.3:** The Custom Multi-Head.
  - Invariant MLP Head from $h_{\text{pool}}$ to forecast scalar energy frequencies $\hat{\omega}_k$.
  - Equivariant Tensor Head from $v_{\text{pool}}$ to yield amplitudes $\hat{B}_k$ vectors.
  - Both heads output exactly $K_{\text{max}} = 50$ slots. Include a Sigmoid Mask $\hat{p}_k$ to prune slots to zero.

## 🔴 Phase 4: Training & The Hungarian Loop (DONE)
Optimizing the Bipartite matching algorithm across the set batches.

- [x] **Step 4.1:** Write the PyTorch dataloaders (`train/dataset.py`). Handle batching graphs with varying lengths of dense $SO(3)$ targets.
- [x] **Step 4.2:** Formulate `train/losses.py`:
  - `Bipartite Match Loss`: Connect the $50$ forecasted output slots to the actual LASSO peak targets utilizing `scipy.optimize.linear_sum_assignment` and computing distance.
  - `Spectrum Analytical Trace Loss`: Perform MSE between $\sum B_k \sin(\omega_k t)$ and actual expected continuous limits. (Optional but brilliant physics regularizer).
- [x] **Step 4.3:** Execute `train/train.py` looping over GPUs, syncing checkpoints, and routing parity plot visualizations straight out directly.

## � Phase 5: Finalization & HPC Inference Scaling (DONE)
Taking our verified local tests on ammonia to the mass-molecular inference level.

- [x] **Step 5.1:** Wrap the saved GNN `.pth` weights actively into `dashboard/app.py` for live "Drop an .xyz to predict spectra instantly".
- [x] **Step 5.2:** Test scale-invariance against larger molecules without dropping $R^2$ parity.
- [ ] **Step 5.3:** Final documentation & paper drafting.