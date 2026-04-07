import os
import glob
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import sys

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.molecule_graph import build_molecule_graph

class SpectrumDataset(Dataset):
    """
    PyTorch Geometric Dataset loading processed molecule targets.
    Maps XYZ atomic configurations to Hauge-extracted (ω_k, B_k) peaks.
    """
    def __init__(self, processed_dir):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.data_dir_path = processed_dir
        self.data_files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")) )
        if not self.data_files:
            print(f"Warning: No processed .pt files found in {processed_dir}")

    def len(self):
        return len(self.data_files)

    def get(self, idx):
        # Load the dictionary of truth data
        file_path = self.data_files[idx]
        data_dict = torch.load(file_path, weights_only=False)
        
        # Build the graph structure 
        # (Nodes: OHE Elements, Edges: cutoff distances)
        graph_data = build_molecule_graph(
            data_dict["atomic_numbers"].numpy(), 
            data_dict["positions"].numpy()
        )
        
        # Append target variables to the PyG Data object
        graph_data.y_freq = data_dict["frequencies"]  # Shape: (K_true,)
        graph_data.y_amp = data_dict["amplitudes_x"]  # Shape: (K_true,)
        # Track per-graph target size so batched losses can split flattened targets safely.
        graph_data.num_peaks = torch.tensor([graph_data.y_freq.numel()], dtype=torch.long)
        
        # PyTorch Geometric collates variable-length tensors into flat vectors.
        # We store num_peaks so losses can split targets per graph robustly.
        return graph_data

def build_dataloader(processed_dir, batch_size=1, shuffle=True):
    dataset = SpectrumDataset(processed_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # Test DataLoader
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    loader = build_dataloader(test_dir, batch_size=1)
    
    for batch in loader:
        print("Batch Load Successful!")
        print("Graph representation:", batch)
        print("Target Frequencies:", batch.y_freq.size())
        print("Target Amplitudes:", batch.y_amp.size())
        break
