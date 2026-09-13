"""User journey covered:

- As a developer wiring the graph-construction output into the model, I want
  a HeteroData graph converted into the flat (x, edge_index, edge_type)
  tensors the relational model expects, with a stable, documented relation
  ordering, so graphs built today and graphs built months from now produce
  the same edge_type encoding.
"""
import torch
from torch_geometric.data import HeteroData

from aegis.model.graph_batching import flatten_hetero_data, RELATION_ORDER


def _make_two_node_graph() -> HeteroData:
    data = HeteroData()
    data["NODE"].x = torch.randn(2, 8)
    data["NODE", "AST", "NODE"].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    data["NODE", "CFG", "NODE"].edge_index = torch.tensor([[1], [0]], dtype=torch.long)
    data["NODE", "REACHING_DEF", "NODE"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    return data


def test_relation_order_is_stable_and_covers_all_three_relations():
    assert RELATION_ORDER == ("AST", "CFG", "REACHING_DEF")


def test_flatten_produces_matching_edge_type_per_edge():
    data = _make_two_node_graph()

    x, edge_index, edge_type = flatten_hetero_data(data)

    assert x.shape == (2, 8)
    assert edge_index.shape[1] == edge_type.shape[0] == 2  # 1 AST edge + 1 CFG edge
    # First edge came from the AST relation (index 0), second from CFG (index 1)
    assert edge_type.tolist() == [0, 1]


def test_flatten_empty_relation_contributes_no_edges():
    data = _make_two_node_graph()

    _, edge_index, edge_type = flatten_hetero_data(data)

    # REACHING_DEF (relation index 2) had zero edges in the fixture
    assert 2 not in edge_type.tolist()
