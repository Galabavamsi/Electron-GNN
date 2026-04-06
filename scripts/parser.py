import re
import numpy as np
import os
import glob

def parse_respect_out(filepath):
    """
    Parses the main ReSpect output file for the time-dependent dipole moment.
    Extracts time grid (a.u.) and induced dipole vector components.
    """
    times = []
    dipole_x, dipole_y, dipole_z = [], [], []
    
    with open(filepath, 'r') as f:
        for line in f:
            if "Step EAS:" in line:
                parts = line.strip().split()
                try:
                    # Indexing assumes: Step EAS: 0  0.000  -56.48  0.0  0.0  0.0  1.0  00:00  1
                    time_val = float(parts[3])
                    x_val = float(parts[5])
                    y_val = float(parts[6])
                    z_val = float(parts[7])
                    
                    times.append(time_val)
                    dipole_x.append(x_val)
                    dipole_y.append(y_val)
                    dipole_z.append(z_val)
                except ValueError:
                    continue
                    
    return np.array(times), np.array(dipole_x), np.array(dipole_y), np.array(dipole_z)

def parse_respect_xyz(filepath):
    """
    Parses the [Atoms] (AU) block from ReSpect xyz grids to natively construct 
    nodes and positions for Molecular Graph processing.
    """
    atomic_numbers = []
    positions = []
    
    in_atoms_block = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("[Atoms]"):
                in_atoms_block = True
                continue
            if line.startswith("[Grid]"):
                break
                
            if in_atoms_block and line:
                parts = line.split()
                if len(parts) >= 6:
                    z = int(parts[2])
                    x = float(parts[3])
                    y = float(parts[4])
                    z_pos = float(parts[5])
                    
                    atomic_numbers.append(z)
                    positions.append([x, y, z_pos])
                    
    return np.array(atomic_numbers), np.array(positions)

def extract_molecule_dataset(data_dir):
    """
    Scans a ReSpect run directory, parses both out and xyz files, 
    and returns a clean dictionary mapping geometry to dipole response.
    """
    out_file = glob.glob(os.path.join(data_dir, "*.out"))
    xyz_file = glob.glob(os.path.join(data_dir, "*.xyz"))
    
    if not out_file or not xyz_file:
        raise FileNotFoundError(f"Missing core ReSpect files in {data_dir}. Checked *.out and *.xyz.")
        
    times, px, py, pz = parse_respect_out(out_file[0])
    z_nums, pos = parse_respect_xyz(xyz_file[0])
    
    return {
        "atomic_numbers": z_nums,
        "positions_au": pos,
        "time_grid": times,
        "dipole_response": {
            "x": px,
            "y": py,
            "z": pz
        }
    }

if __name__ == "__main__":
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ammonia_x"))
    print(f"Testing extraction on {test_dir}...")
    try:
        data = extract_molecule_dataset(test_dir)
        print("Successfully Parsed!")
        print(f"Number of Atoms: {len(data['atomic_numbers'])}")
        print(f"Time steps extracted: {len(data['time_grid'])}")
        print(f"Final t_max: {np.max(data['time_grid'])} a.u.")
        print(f"Dipole Vector Array lengths (X, Y, Z): {len(data['dipole_response']['x'])}, {len(data['dipole_response']['y'])}, {len(data['dipole_response']['z'])}")
    except Exception as e:
        print(f"Pipeline extraction failed: {str(e)}")
