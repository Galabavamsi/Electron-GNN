import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add the root directory to path to import local scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.plot_spectrum import plot_absorption_spectrum
from utils.signal_utils import extrapolate_dipole_moment

def generate_documentation_plots():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "ammonia_targets.pt"))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "assets"))
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: Target data not found at {data_path}. Run extract_peaks.py first.")
        return
        
    data = torch.load(data_path, weights_only=False)
    
    # Extract tensors
    w_k = data["frequencies"].numpy()
    B_k = data["amplitudes_x"].numpy()
    time_grid = data["raw_time"]
    raw_dipole = data["raw_dipole_x"]
    
    print(f"Loaded {len(w_k)} peaks.")
    
    # 1. Plot the Ground Truth Spectrum Reconstructed Analytically
    omega_grid = np.linspace(0.01, 1.0, 2000)
    spectrum = np.zeros_like(omega_grid)
    gamma = 0.005 # Extracted from the Hauge paper typical parameters
    
    for w, b in zip(w_k, B_k):
        spectrum += b * (gamma / ((omega_grid - w)**2 + gamma**2))
        
    plt.figure(figsize=(10, 5))
    plt.plot(omega_grid, spectrum, color="darkred", linewidth=1.5, label="Reconstructed S(ω)")
    plt.fill_between(omega_grid, spectrum, color="red", alpha=0.3)
    # Highlight the extracted peaks
    for w, b in zip(w_k, B_k):
        if b > 1e-4:  # Only label significant peaks
            plt.axvline(x=w, color='gray', linestyle='--', alpha=0.5, ymax=0.8)
            
    plt.title(f"Analytically Reconstructed Absorption Spectrum (NH3, X-polarization) \n{len(w_k)} LASSO Peaks", fontsize=14)
    plt.xlabel("Frequency ω (a.u.)", fontsize=12)
    plt.ylabel("Intensity / Cross-Section S(ω)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    spec_path = os.path.join(output_dir, "ammonia_spectrum.png")
    plt.savefig(spec_path, dpi=300)
    print(f"Saved Spectrum Plot -> {spec_path}")
    plt.close()
    
    # 2. Plot the Raw Dipole vs The Extrapolated Dipole
    plt.figure(figsize=(12, 4))
    
    # Extrapolate beyond the ReSpect simulation limits (t=400)
    extrap_time = np.linspace(0, 1000, 4000) 
    extrap_dipole = extrapolate_dipole_moment(w_k, B_k, extrap_time)
    
    plt.plot(extrap_time, extrap_dipole, label='Extrapolated (Padé+LASSO)', color='blue', alpha=0.8, linewidth=1.2)
    plt.plot(time_grid, raw_dipole, label='Original ReSpect RT-TDDFT (T=400)', color='black', alpha=0.8, linewidth=1.5, linestyle='--')
    
    plt.axvline(x=400, color='red', linestyle=':', label='Simulation Boundary')
    
    plt.title("Time-Dependent Induced Dipole Moment Extrapolation μ(t)", fontsize=14)
    plt.xlabel("Time (a.u.)", fontsize=12)
    plt.ylabel("Dipole Moment μ_x", fontsize=12)
    plt.xlim(0, 1000)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    dipole_path = os.path.join(output_dir, "ammonia_dipole_extrapolation.png")
    plt.savefig(dipole_path, dpi=300)
    print(f"Saved Dipole Extrapolation Plot -> {dipole_path}")
    plt.close()

if __name__ == "__main__":
    print("Generating High-Resolution Documentation Plots...")
    generate_documentation_plots()
