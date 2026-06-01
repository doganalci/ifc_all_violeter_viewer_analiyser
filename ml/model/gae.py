"""Graph AutoEncoder — anomali tespiti için (denetimsiz).

Sadece NORMAL (compliant) baseline grafiklerle eğitilir. Her node'un
feature'ını komşuluğundan yeniden üretmeyi öğrenir. Inference'ta yeniden
üretemediği (yüksek reconstruction error) node'lar = anomali = potansiyel
ihlal.

Sınıflandırıcıdan farkı: 'ihlal' etiketi GÖRMEZ. Sadece 'normal neye
benzer' öğrenir; sapmaları işaretler → görülmemiş ihlal tiplerini de
yakalayabilir.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GraphAutoEncoder(nn.Module):
    """GATv2 encoder + MLP decoder; node feature reconstruction.

    Anomali skoru = ||x - x_hat||^2 (node başına).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64,
                 latent_dim: int = 32, num_edge_types: int = 7,
                 edge_emb_dim: int = 8, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.edge_emb = nn.Embedding(num_edge_types, edge_emb_dim)
        self.enc1 = GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout,
                              edge_dim=edge_emb_dim, concat=True)
        self.enc2 = GATv2Conv(hidden_dim * heads, latent_dim, heads=1,
                              dropout=dropout, edge_dim=edge_emb_dim, concat=False)
        # Decoder: latent → orijinal feature uzayı
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
        )

    def encode(self, x, edge_index, edge_type):
        e = self.edge_emb(edge_type) if edge_type.numel() else None
        h = F.elu(self.enc1(x, edge_index, edge_attr=e))
        z = self.enc2(h, edge_index, edge_attr=e)
        return z

    def forward(self, x, edge_index, edge_type):
        z = self.encode(x, edge_index, edge_type)
        x_hat = self.dec(z)
        return x_hat

    @torch.no_grad()
    def anomaly_score(self, x, edge_index, edge_type) -> torch.Tensor:
        """Node başına reconstruction error (yüksek = anormal)."""
        x_hat = self.forward(x, edge_index, edge_type)
        return ((x - x_hat) ** 2).mean(dim=1)
