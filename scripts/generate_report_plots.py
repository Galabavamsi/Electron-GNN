import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mace_net import SpectralEquivariantGNN
from train.dataset import SpectrumDataset
from utils.model_diagnostics import plot_predict_vs_real_parity

os.makedirs('docs/assets/report', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SpectralEquivariantGNN(node_features_in=5, K_max=50)
ckpt = 'checkpoints/best_model.pth'

if not os.path.exists(ckpt):
    print(f"No checkpoint found at {ckpt}")
    sys.exit(1)

model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
model.to(device)
model.eval()

data_dir = 'data/processed'
dataset = SpectrumDataset(data_dir)

def process_molecule(name_filter):
    idx = -1
    for i, f in enumerate(dataset.data_files):
        if name_filter in f:
            idx = i
            break
    if idx == -1: return None, None

    graph_data = dataset[idx]
    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
    graph_data.y_freq_batch = torch.zeros(graph_data.y_freq.shape[0], dtype=torch.long)
    graph_data = graph_data.to(device)

    with torch.no_grad():
        pred_dict = model(graph_data)
    
    prob_key = "prob" if "prob" in pred_dict else "peak_probs"
    freq_key = "freq" if "freq" in pred_dict else "frequencies"
    amp_key = "amp" if "amp" in pred_dict else "amplitudes"

    probs_t = pred_dict[prob_key].squeeze(0)
    if prob_key == "peak_probs": probs_t = torch.sigmoid(probs_t)
    freqs_t = pred_dict[freq_key].squeeze(0)
    amps_t = pred_dict[amp_key].squeeze(0)

    probs = probs_t.detach().cpu().numpy()
    pred_w_all = freqs_t.detach().cpu().numpy()
    amp_np = amps_t.detach().cpu().numpy()

    if amp_np.ndim == 2:
        pred_b_all = np.linalg.norm(amp_np, axis=1)
    else:
        pred_b_all = np.abs(amp_np)
        
    mask = probs > 0.65
    if np.count_nonzero(mask) == 0:
        top_k = min(5, probs.shape[0])
        top_idx = np.argsort(probs)[-top_k:]
        pred_w = pred_w_all[top_idx]
        pred_b = pred_b_all[top_idx]
    else:
        pred_w = pred_w_all[mask]
        pred_b = pred_b_all[mask]
        
    true_w = graph_data.y_freq.detach().cpu().numpy()
    true_b = np.abs(graph_data.y_amp.detach().cpu().numpy())
    return (pred_w, pred_b), (true_w, true_b)

for mol in ['ammonia', 'water']:
    print(f"Generating plots for {mol}...")
    preds, truths = process_molecule(mol)
    if preds is None: continue
    
    pred_w, pred_b = preds
    true_w, true_b = truths

    # 1. Parity Plot
    fig_parity = plot_predict_vs_real_parity(true_w, pred_w, true_b, pred_b)
    fig_parity.savefig(f'docs/assets/report/parity_{mol}.png', bbox_inches='tight')
    plt.close(fig_parity)
    
    # 2. Spectrum Plot
    omega = np.linspace(0.01, 1.5, 1000)
    gamma = 0.015
    spec_t = np.zeros_like(omega)
    for w_k, b_k in zip(true_w, true_b):
        spec_t += b_k * (gamma / ((omega - w_k)**2 + gamma**2))
    spec_p = np.zeros_like(omega)
    for w_k, b_k in zip(pred_w, pred_b):
        spec_p += b_k * (gamma / ((omega - w_k)**2 + gamma**2))
        
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(omega, spec_t, 'k-', label="True Spectrum (RT-TDDFT)")
    ax.fill_between(omega, spec_t, color="gray", alpha=0.1)
    ax.plot(omega, spec_p, 'm--', label="GNN Prediction")
    ax.fill_between(omega, spec_p, color="purple", alpha=0.2)
    ax.set_title(f"{mol.title()} Absorption Spectrum")
    ax.set_xlabel(r"Frequency $\omega$ (a.u.)")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.savefig(f'docs/assets/report/spectrum_{mol}.png', bbox_inches='tight')
    plt.close(fig)
    
    # 3. Dipole Moment Time Series
    times = np.linspace(0, 400, 400)
    true_signal = np.zeros_like(times)
    pred_signal = np.zeros_like(times)
    for w_k, b_k in zip(true_w, true_b):
        true_signal += b_k * np.sin(w_k * times)
    for w_k, b_k in zip(pred_w, pred_b):
        pred_signal += b_k * np.sin(w_k * times)
        
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(times, true_signal, 'k-', label="True Signal")
    ax.plot(times, pred_signal, 'r--', label=r"Predicted $\sum B_k \sin(\omega_k t)$")
    ax.set_title(f"{mol.title()} Dipole Moment $\mu(t)$ (Global Observable)")
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("Linear Response")
    ax.legend()
    fig.savefig(f'docs/assets/report/dipole_{mol}.png', bbox_inches='tight')
    plt.close(fig)
    
print("Done generating plots for supervisor report.")
