import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def match_peaks_hungarian(w_true, w_pred, b_true, b_pred):
    """
    Solves the permutation ambiguity by matching predicted peaks to true peaks
    using the Hungarian algorithm. Extremely useful for debugging predict vs real.
    Distance metric is a weighted sum of frequency error and amplitude error.
    """
    n_true = len(w_true)
    n_pred = len(w_pred)
    
    # Cost matrix computation
    cost_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            # Weight frequency error heavily as it dictates peak position
            cost_matrix[i, j] = 10.0 * np.abs(w_true[i] - w_pred[j]) + np.abs(b_true[i] - b_pred[j])
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Return matched pairs
    return w_true[row_ind], w_pred[col_ind], b_true[row_ind], b_pred[col_ind]

def plot_predict_vs_real_parity(w_true, w_pred, b_true, b_pred):
    """
    Generates Parity Plots (Predicted vs Ground Truth) for both frequencies and amplitudes.
    Automatically handles permutation matching.
    """
    w_t, w_p, b_t, b_p = match_peaks_hungarian(w_true, w_pred, b_true, b_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Frequency Parity
    ax1.scatter(w_t, w_p, color='red', alpha=0.7, edgecolors='k')
    min_w, max_w = min(np.min(w_t), np.min(w_p)), max(np.max(w_t), np.max(w_p))
    ax1.plot([min_w, max_w], [min_w, max_w], 'k--', zorder=0)
    ax1.set_title("Transition Frequencies ω_k (a.u.)")
    ax1.set_xlabel("True (LASSO Extracted)")
    ax1.set_ylabel("GNN Predicted")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Amplitude Parity
    ax2.scatter(b_t, b_p, color='blue', alpha=0.7, edgecolors='k')
    min_b, max_b = min(np.min(b_t), np.min(b_p)), max(np.max(b_t), np.max(b_p))
    ax2.plot([min_b, max_b], [min_b, max_b], 'k--', zorder=0)
    ax2.set_title("Dipole Amplitudes B_k (a.u.)")
    ax2.set_xlabel("True (LASSO Extracted)")
    ax2.set_ylabel("GNN Predicted")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("Model Assessment: Parity Plots (Bipartite Matched)", y=1.05, fontsize=14)
    plt.tight_layout()
    return fig

def plot_complex_poles(roots, true_frequencies=None):
    """
    Scientific debugging tool: Plots the complex roots of the Padé polynomial Q(z)=0.
    Physical frequencies lie exactly on the upper-half unit circle. 
    Numerical noise falls near the origin or infinity.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    
    # Plot roots
    ax.scatter(np.real(roots), np.imag(roots), color='gray', alpha=0.5, label='All Padé Poles')
    
    if true_frequencies is not None:
        # Assuming true_frequencies are in radians/step for mapping back to z-plane
        z_true = np.exp(1j * true_frequencies)
        ax.scatter(np.real(z_true), np.imag(z_true), color='red', marker='x', s=100, label='Clustered Physical Poles')
        
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title("Padé Approximant Poles in Complex Plane")
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    
    return fig

def calc_spectral_overlap_score(spec_true, spec_pred):
    """
    Calculates the cosine similarity / overlap integral between two smooth spectra.
    A value of 1.0 means perfect functional overlap.
    """
    numerator = np.trapezoid(spec_true * spec_pred)
    denominator = np.sqrt(np.trapezoid(spec_true**2) * np.trapezoid(spec_pred**2))
    return numerator / (denominator + 1e-10)
