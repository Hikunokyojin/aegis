"""Relational graph attention model for classifying Code Property Graph
nodes as belonging to a vulnerable or safe function.

AegisHeteroGNN stacks relation-aware graph attention layers (RGATConv) so
the model learns a different aggregation weight per edge type (AST vs.
control-flow vs. data-dependency) instead of treating every edge the same
way, then pools each graph's node representations into a single per-function
prediction.

FocalLoss down-weights easy (already well-classified) examples during
training so the rare "vulnerable" class isn't drowned out by the abundant
"safe" class — plain cross-entropy alone tends to converge to "always
predict safe" on this kind of heavily imbalanced data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv, global_mean_pool


class AegisHeteroGNN(nn.Module):
    """Relational GAT encoder over a Code Property Graph.

    edge_type indices follow aegis.graphs.heterodata_builder.RELATION_ORDER:
    0 = AST, 1 = CFG, 2 = REACHING_DEF (data-dependency).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, num_relations: int = 3, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            RGATConv(hidden_dim, hidden_dim, num_relations=num_relations, heads=4, concat=False)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, 2)  # binary: vulnerable / safe

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.layers, self.norms):
            h = norm(F.relu(conv(h, edge_index, edge_type)) + h)  # residual connection
        pooled = global_mean_pool(h, batch)
        return self.classifier(pooled)


class FocalLoss(nn.Module):
    """Down-weights easy (safe-code) examples so the rare vulnerable class
    isn't drowned out during training — standard cross-entropy alone
    converges to 'always predict safe' on imbalanced vulnerability data."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()
