"""Homogeneous-graph GAT baseline.

Two GATv2 layers with multi-head attention, edge-type embedded as an
edge feature so the attention module can distinguish e.g. `contains`
from `bounds`.

The forward returns *logits* (not sigmoided) so the caller can pair
with `BCEWithLogitsLoss` for numerical stability.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATNodeClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_edge_types: int = 7,
        edge_emb_dim: int = 8,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.edge_emb = nn.Embedding(num_edge_types, edge_emb_dim)
        self.conv1 = GATv2Conv(
            in_dim, hidden_dim, heads=heads, dropout=dropout,
            edge_dim=edge_emb_dim, add_self_loops=True,
        )
        self.conv2 = GATv2Conv(
            hidden_dim * heads, hidden_dim, heads=1, concat=False,
            dropout=dropout, edge_dim=edge_emb_dim, add_self_loops=True,
        )
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        e = self.edge_emb(edge_type) if edge_type.numel() else None
        h = self.conv1(x, edge_index, edge_attr=e)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_attr=e)
        h = F.elu(h)
        return self.head(h).squeeze(-1)  # [N] logits
