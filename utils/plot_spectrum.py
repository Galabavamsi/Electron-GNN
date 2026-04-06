import numpy as np
import matplotlib.pyplot as plt

def plot_absorption_spectrum(frequencies, amplitudes, omega_grid, gamma=0.005, title="Predicted Absorption Spectrum S(ω)"):
    """
    Analytically reconstructs and plots the absorption spectrum from the extracted spectral parameters.
    
    Parameters:
    - frequencies (array): ω_k predicted by the invariant GNN head
    - amplitudes (array): B_k predicted by the equivariant GNN head
    - omega_grid (array): The frequency range / grid exactly matching TDDFT resolution
    - gamma (float): Damping/broadening parameter to create physical Lorentzian peaks
    - title (str): Plot title
    """
    spectrum = np.zeros_like(omega_grid)
    
    for w_k, b_k in zip(frequencies, amplitudes):
        # A simple Lorentzian lineshape function generated analytically
        spectrum += b_k * (gamma / ((omega_grid - w_k)**2 + gamma**2))
        
    plt.figure(figsize=(10, 5))
    plt.plot(omega_grid, spectrum, color="blue", linewidth=1.5, label="Reconstructed Spectrum")
    plt.fill_between(omega_grid, spectrum, color="lightblue", alpha=0.4)
    
    plt.title(title, fontsize=14, pad=10)
    plt.xlabel("Frequency ω (a.u.)", fontsize=12)
    plt.ylabel("Intensity / Cross-Section S(ω)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
