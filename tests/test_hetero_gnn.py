"""User journeys covered:

- As a security engineer, I want the model to combine CodeBERT-embedded node
  features with AST/CFG/data-dependency edge types using relation-aware
  attention, so it can learn to distinguish vulnerable from safe code based
  on structural context, not just token similarity.
- As a developer training the model, I want a loss function that
  down-weights easy (well-classified) examples, so the rare vulnerable class
  isn't drowned out by the abundant safe class during training.
"""
import torch

from aegis.model.hetero_gnn import AegisHeteroGNN, FocalLoss


def test_forward_pass_produces_per_graph_binary_logits():
    """A batch of 3 single-node graphs should produce 3 rows of 2 logits
    (vulnerable / safe), regardless of input feature width."""
    in_dim = 16
    model = AegisHeteroGNN(in_dim=in_dim, hidden_dim=8, num_relations=3, num_layers=2)

    num_nodes = 5
    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    batch = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)  # 3 graphs

    logits = model(x, edge_index, edge_type, batch)

    assert logits.shape == (3, 2)


def test_forward_pass_handles_single_graph_with_no_edges():
    """An isolated single-node graph (no edges at all) must not crash the
    relational attention layers."""
    in_dim = 16
    model = AegisHeteroGNN(in_dim=in_dim, hidden_dim=8, num_relations=3, num_layers=2)

    x = torch.randn(1, in_dim)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.zeros((0,), dtype=torch.long)
    batch = torch.zeros(1, dtype=torch.long)

    logits = model(x, edge_index, edge_type, batch)

    assert logits.shape == (1, 2)


def test_focal_loss_is_nonnegative_scalar():
    loss_fn = FocalLoss()
    logits = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 1, 0])

    loss = loss_fn(logits, targets)

    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_focal_loss_penalizes_confident_wrong_predictions_more_than_right_ones():
    """A batch where the model is confidently WRONG should have higher focal
    loss than a batch where it is confidently RIGHT, for the same targets."""
    loss_fn = FocalLoss()
    targets = torch.tensor([1])

    confident_correct = torch.tensor([[-5.0, 5.0]])  # strongly predicts class 1
    confident_wrong = torch.tensor([[5.0, -5.0]])     # strongly predicts class 0

    loss_correct = loss_fn(confident_correct, targets)
    loss_wrong = loss_fn(confident_wrong, targets)

    assert loss_wrong.item() > loss_correct.item()
