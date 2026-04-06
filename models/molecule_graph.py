import torch
from torch_geometric.data import Data

def build_molecule_graph(atomic_numbers, positions, cutoff_radius=5.0):
    """
    Constructs a PyTorch Geometric Data object from atomic numbers and positions.
    Uses a distance cutoff to establish edges (bonds / interactions).
    """
    num_atoms = len(atomic_numbers)
    
    # 1. Node Features (One-hot encoding of elements)
    # Simple embedding for H, C, N, O, F (assuming atomic numbers 1, 6, 7, 8, 9)
    valid_zs = [1, 6, 7, 8, 9]
    x = torch.zeros(num_atoms, len(valid_zs), dtype=torch.float32)
    for i, z in enumerate(atomic_numbers):
        if z in valid_zs:
            x[i, valid_zs.index(z)] = 1.0
        else:
            # Fallback for unknown atoms
            x[i, -1] = 1.0
            
    # 2. Edge Indices (Fully connected within cutoff, excluding self loops)
    edge_index_list = []
    edge_attr_list = []
    
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                continue
            
            p_i = torch.tensor(positions[i], dtype=torch.float32)
            p_j = torch.tensor(positions[j], dtype=torch.float32)
            dist = torch.norm(p_i - p_j)
            
            if dist <= cutoff_radius:
                edge_index_list.append([i, j])
                
                # 3. Edge Features (Distance & Relative Vector)
                rel_vec = p_i - p_j
                edge_attr_list.append(torch.cat([torch.tensor([dist]), rel_vec]))
                
    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr_list)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float32)
        
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=torch.tensor(positions, dtype=torch.float32), z=torch.tensor(atomic_numbers, dtype=torch.long))

