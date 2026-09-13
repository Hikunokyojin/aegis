"""Flatten a torch_geometric HeteroData graph (as produced by
aegis.graphs.heterodata_builder) into the (x, edge_index, edge_type) tensors
the relational model in aegis.model.hetero_gnn expects.

The single-node-type, multi-relation HeteroData shape used throughout Aegis
(all nodes typed "NODE", edges typed ("NODE", relation, "NODE")) already
carries everything the model needs — this module just re-packs it into the
flat form RGATConv's API takes, using a stable relation -> integer mapping.
"""
import torch
from torch_geometric.data import HeteroData

from aegis.graphs.heterodata_builder import RELATION_ORDER


def flatten_hetero_data(data: HeteroData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (x, edge_index, edge_type) for a single-graph HeteroData with
    one "NODE" node type and edges typed ("NODE", relation, "NODE") for each
    relation in RELATION_ORDER. Relations absent from the graph (or present
    with zero edges) simply contribute no edges."""
    x = data["NODE"].x

    edge_index_parts = []
    edge_type_parts = []
    for relation_idx, relation in enumerate(RELATION_ORDER):
        edge_type_key = ("NODE", relation, "NODE")
        if edge_type_key not in data.edge_types:
            continue
        rel_edge_index = data[edge_type_key].edge_index
        num_edges = rel_edge_index.shape[1]
        if num_edges == 0:
            continue
        edge_index_parts.append(rel_edge_index)
        edge_type_parts.append(torch.full((num_edges,), relation_idx, dtype=torch.long))

    if edge_index_parts:
        edge_index = torch.cat(edge_index_parts, dim=1)
        edge_type = torch.cat(edge_type_parts, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)

    return x, edge_index, edge_type
