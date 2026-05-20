"""Relational variant: per-edge-type GATv2 wrapped in `to_hetero` style.

Rather than building `HeteroData` (which would require multi-typed nodes
too), we split the homogeneous edge index by `edge_type`, run a GATv2
conv per relation, and average the outputs. This keeps the data
pipeline identical to the homogeneous case while still letting each
relation learn its own attention parameters.

For an MVP baseline, the homogeneous `GATNodeClassifier` is usually
sufficient and trains noticeably faster. This is here as the next
ablation step.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class _RelationalGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int,
                 heads: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.convs = nn.ModuleList(
            [GATv2Conv(in_dim, out_dim, heads=heads, concat=False,
                       dropout=dropout, add_self_loops=False)
             for _ in range(num_relations)]
        )
        # A residual self-loop conv so isolated nodes still get an update.
        self.self_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        out = self.self_proj(x)
        for r, conv in enumerate(self.convs):
            mask = edge_type == r
            if not mask.any():
                continue
            ei = edge_index[:, mask]
            out = out + conv(x, ei)
        return out / (self.num_relations + 1)


class HeteroGATNodeClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_edge_types: int = 7,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.layer1 = _RelationalGATLayer(in_dim, hidden_dim, num_edge_types,
                                          heads=heads, dropout=dropout)
        self.layer2 = _RelationalGATLayer(hidden_dim, hidden_dim, num_edge_types,
                                          heads=heads, dropout=dropout)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        h = F.elu(self.layer1(x, edge_index, edge_type))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.layer2(h, edge_index, edge_type))
        return self.head(h).squeeze(-1)
