import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.nn import global_add_pool


class SpectralEquivariantGNNV1(torch.nn.Module):
    """
    Legacy V1 architecture used before the current GATv2 + set-decoder model.
    Keeps a simple invariant/equivariant split and predicts fixed-size peak slots.
    """

    def __init__(self, node_features_in=5, hidden_dim=64, K_max=50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.K_max = K_max

        self.node_emb = nn.Linear(node_features_in, hidden_dim)

        # Legacy irreps descriptors retained for architecture compatibility.
        self.irreps_node_hidden = o3.Irreps(f"{hidden_dim}x0e + {hidden_dim}x1o")
        self.irreps_out_vector = o3.Irreps("1x1o")

        self.scalar_msg_pass = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.vector_msg_pass = o3.Linear(o3.Irreps(f"{hidden_dim}x1o"), self.irreps_out_vector)

        self.head_freq = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, K_max),
            nn.Softplus(),
        )
        self.head_amp_scalar = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, K_max),
        )
        self.head_prob = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, K_max),
            nn.Sigmoid(),
        )

    def forward(self, data):
        x, batch = data.x, data.batch

        h_scalar = self.node_emb(x)
        h_vector = torch.zeros(h_scalar.size(0), self.hidden_dim, 3, device=x.device)

        h_scalar_updated = self.scalar_msg_pass(h_scalar) + h_scalar

        global_scalar = global_add_pool(h_scalar_updated, batch)

        num_nodes, hidden_dim, _ = h_vector.shape
        flat_vector = h_vector.view(num_nodes, hidden_dim * 3)
        _global_vector = global_add_pool(flat_vector, batch).view(-1, hidden_dim, 3)

        p_existence = self.head_prob(global_scalar)
        w_freqs = self.head_freq(global_scalar)
        b_amps_mag = self.head_amp_scalar(global_scalar)

        return {
            "prob": p_existence,
            "freq": w_freqs,
            "amp": b_amps_mag,
        }
