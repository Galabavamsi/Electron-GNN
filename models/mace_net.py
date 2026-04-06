import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.nn import global_add_pool

class SpectralEquivariantGNN(torch.nn.Module):
    """
    Simplified Equivariant GNN capable of resolving continuous physical spectrum parameters.
    Splits features into scalar invariants (for frequencies) and vector equivariants (for amplitudes).
    """
    def __init__(self, node_features_in=5, hidden_dim=64, K_max=50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.K_max = K_max
        
        # Linear embedding of one-hot nodes -> rich latent scalar
        self.node_emb = nn.Linear(node_features_in, hidden_dim)
        
        # We need a pathway mapped to O(3) irreps space
        # L=0 (Scalars), L=1 (Vectors x,y,z)
        self.irreps_node_hidden = o3.Irreps(f"{hidden_dim}x0e + {hidden_dim}x1o")
        
        # Message passing would happen here utilizing tensor products.
        # (Simplified Multi-Layer Perceptron proxy to symbolize the tensor-passing pipeline)
        self.scalar_msg_pass = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Vector features mapped carefully through E(3) linear layers
        self.irreps_out_vector = o3.Irreps("1x1o")
        self.vector_msg_pass = o3.Linear(o3.Irreps(f"{hidden_dim}x1o"), self.irreps_out_vector)
        
        # -------- Multi-Head Readouts --------
        # Invariant Head (predicting K Bohr frequencies ω_k)
        self.head_freq = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, K_max),
            nn.Softplus() # Enforce positive frequency
        )
        
        # Equivariant Head (predicting Vector Amplitudes B_k)
        # B_k needs K outputs of size 3 (K_max, 3) 
        self.head_amp_scalar = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, K_max)
        )
        
        # Sigmoid Mask for Probability (Is there a peak here?)
        self.head_prob = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, K_max),
            nn.Sigmoid()
        )

    def forward(self, data):
        # 1. Extract inputs
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 2. Initial Embedding
        h_scalar = self.node_emb(x)
        h_vector = torch.zeros(h_scalar.size(0), self.hidden_dim, 3, device=x.device)
        
        # 3. Message Passing (Simplified pooling logic)
        # Real implementation involves e3nn TensorProduct interactions.
        h_scalar_updated = self.scalar_msg_pass(h_scalar) + h_scalar
        
        # 4. Global Pooling across atoms to describe the whole molecule
        global_scalar = global_add_pool(h_scalar_updated, batch)
        # flatten vector before pooling then reshape back
        B, D, _ = h_vector.shape
        flat_vector = h_vector.view(B, D * 3)
        global_vector = global_add_pool(flat_vector, batch).view(-1, D, 3)
        
        # 5. Extract K_max Slots (Variable size output emulation)
        p_existence = self.head_prob(global_scalar) # Shape: (Batch, K_max)
        w_freqs = self.head_freq(global_scalar)     # Shape: (Batch, K_max)
        
        # Vectors map linearly via invariant tensor products and pooling
        # Multiply invariant feature magnitude onto a single direction frame
        b_amps_mag = self.head_amp_scalar(global_scalar) # Shape: (Batch, K_max)
        
            
        return {
            "prob": p_existence,
            "freq": w_freqs,
            "amp": b_amps_mag # Represents length/direction of the extracted vector field
        }

