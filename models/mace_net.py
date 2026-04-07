import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch

class SpectralEquivariantGNN(torch.nn.Module):
    """
    Peak-set predictor for spectra using an edge-aware graph encoder and
    a query-based set decoder (DETR-style slots for unordered transitions).
    """
    def __init__(
        self,
        node_features_in=5,
        hidden_dim=128,
        K_max=64,
        num_layers=4,
        num_heads=4,
        dropout=0.0,
        amp_scale=1e-3,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.K_max = K_max
        self.amp_scale = amp_scale

        self.node_emb = nn.Sequential(
            nn.Linear(node_features_in, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_emb = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        head_dim = hidden_dim // num_heads
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=head_dim,
                    heads=num_heads,
                    concat=True,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.query_embed = nn.Parameter(torch.randn(K_max, hidden_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.set_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.slot_refine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.head_prob = nn.Linear(hidden_dim, 1)
        self.head_count = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head_freq = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.head_amp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h = self.node_emb(x)
        e = self.edge_emb(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, e)
            h = norm(h)
            h = F.gelu(h)
            h = h + h_res

        dense_nodes, node_mask = to_dense_batch(h, batch)
        batch_size = dense_nodes.shape[0]

        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        slot_features = self.set_decoder(
            tgt=queries,
            memory=dense_nodes,
            memory_key_padding_mask=~node_mask,
        )

        pooled = torch.cat([
            global_mean_pool(h, batch),
            global_max_pool(h, batch),
        ], dim=-1)
        global_context = self.global_context(pooled)
        context = global_context.unsqueeze(1).expand(-1, self.K_max, -1)
        slot_features = self.slot_refine(torch.cat([slot_features, context], dim=-1))

        # Predicted active-peak count (cardinality) for top-N decoding.
        count_pred = F.softplus(self.head_count(global_context).squeeze(-1))

        prob_logits = self.head_prob(slot_features).squeeze(-1)
        p_existence = torch.sigmoid(prob_logits)

        w_freqs = F.softplus(self.head_freq(slot_features).squeeze(-1)) + 1e-5
        b_amps_mag = F.softplus(self.head_amp(slot_features).squeeze(-1)) * self.amp_scale

        return {
            "prob": p_existence,
            "prob_logits": prob_logits,
            "freq": w_freqs,
            "amp": b_amps_mag,
            "count": count_pred,
        }

