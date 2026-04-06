import os
import sys
import numpy as np
import torch

# Add the root directory to path to import local scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the cloned Hauge repository to path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib', 'absorption-spectrum')))

from scripts.parser import extract_molecule_dataset
try:
    from spectra.dipole import BroadbandDipole
except ImportError:
    print("Failed to import HyQD spectra library. Ensure it is cloned into lib/absorption-spectrum/")
    sys.exit(1)

def extract_peaks_for_molecule(data_dir, axis='x', cutoff_freq=4.0):
    """
    Given a raw data directory, parses the ReSpect output and extracts 
    the target physical peaks (frequencies ω_k and amplitudes B_k)
    using the Hauge et al. Padé+LASSO analytical pipeline.
    """
    print(f"--- Extracting Peaks via HyQD Hauge Pipeline for {os.path.basename(data_dir)} ({axis}-axis) ---")
    data = extract_molecule_dataset(data_dir)
    
    time_grid = data['time_grid']
    dipole_signal = data['dipole_response'][axis]
    
    # Initialize the Physics-informed Padé + K-Means + LASSO extractor
    print(f"Initializing BroadbandDipole with Cutoff Frequency = {cutoff_freq} a.u.")
    extractor = BroadbandDipole(cutoff_frequency=cutoff_freq)
    
    # Run the fit (Finds roots of Q(z)=0, Clusters, and Applies positive LASSO)
    print("Fitting Dipole trajectory... (This runs Padé, K-Means, and L1 Regressions)")
    success = extractor.fit(dipole_signal, time_grid)
    
    omega_k = extractor.frequencies
    B_k = extractor.B
    
    # Filter out strictly zero peaks enforced by LASSO
    active_mask = B_k > 1e-8
    omega_k_active = omega_k[active_mask]
    B_k_active = B_k[active_mask]
    
    print(f"Extraction {'Successful' if success else 'Warning (Validation Error > Tol)'}! Found {len(omega_k_active)} active quantum transitions.")
    
    # Compile the final ML target structure
    result = {
        "atomic_numbers": torch.tensor(data['atomic_numbers'], dtype=torch.long),
        "positions": torch.tensor(data['positions_au'], dtype=torch.float32),
        "frequencies": torch.tensor(omega_k_active, dtype=torch.float32),
        "amplitudes_x": torch.tensor(B_k_active, dtype=torch.float32), # B^x because pulse was x
        "raw_time": time_grid,
        "raw_dipole_x": dipole_signal
    }
    
    return result

if __name__ == "__main__":
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    os.makedirs(output_dir, exist_ok=True)
    
    for folder in os.listdir(raw_dir):
        if folder.endswith("_x"):
            target_data = os.path.join(raw_dir, folder)
            print(f"Processing {target_data}...")
            try:
                sample_data = extract_peaks_for_molecule(target_data, axis='x', cutoff_freq=4.0)
                out_path = os.path.join(output_dir, f"{folder.replace('_x', '')}_targets.pt")
                torch.save(sample_data, out_path)
                print(f"Saved PyTorch geometric targets to {out_path}\n")
            except Exception as e:
                print(f"Peak extraction failed for {folder}: {e}")
