"""Heterogeneous graph attention model for classifying a function's Code
Property Graph as vulnerable or safe.

AegisHeteroGNN stacks Heterogeneous Graph Transformer layers (HGTConv), which
learn separate attention per edge type (AST vs. control-flow vs.
data-dependency) instead of treating every edge the same way, then pool each
graph's nodes into a single per-function prediction.

HGTConv replaced RGATConv: RGATConv builds a full weight matrix for every
single edge (~1MB per edge at hidden size 256), which tried to allocate 19GB
on one batch of real BigVul functions. HGTConv transforms each node once per
edge type, so memory grows with graph size instead.

FocalLoss down-weights easy (already well-classified) examples during
training so the rare "vulnerable" class isn't drowned out by the abundant
"safe" class — plain cross-entropy alone tends to converge to "always
predict safe" on this kind of heavily imbalanced data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, global_mean_pool

from aegis.graphs.heterodata_builder import RELATION_ORDER

NODE = "NODE"
GRAPH_METADATA = ([NODE], [(NODE, relation, NODE) for relation in RELATION_ORDER])


class AegisHeteroGNN(nn.Module):
    """HGT encoder over a Code Property Graph with one node type ("NODE")
    and three edge types (AST, CFG, REACHING_DEF)."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, num_layers: int = 3, heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([HGTConv(hidden_dim, hidden_dim, GRAPH_METADATA, heads=heads)
                                     for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, 2)  # binary: vulnerable / safe

    def forward(self, x: torch.Tensor, edge_index_dict: dict, batch: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.layers, self.norms):
            out = conv({NODE: h}, edge_index_dict)[NODE]
            if out is None:  # no edges of any type anywhere in the batch
                out = torch.zeros_like(h)
            h = norm(F.relu(out) + h)  # residual connection
        return self.classifier(global_mean_pool(h, batch))


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
