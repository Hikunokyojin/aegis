"""User journeys covered:

- As a security engineer, I want the model to combine CodeBERT-embedded node
  features with AST/CFG/data-dependency edge types using type-aware
  attention (HGTConv), so it can learn to distinguish vulnerable from safe
  code based on structural context, not just token similarity.
- As someone training on real BigVul functions (some with 1,000+ nodes and
  8,000+ edges), I want memory to grow with graph size, not explode — the
  earlier RGATConv model tried to allocate 19GB on one real batch.
- As a developer training the model, I want a loss function that
  down-weights easy (well-classified) examples, so the rare vulnerable class
  isn't drowned out by the abundant safe class during training.
"""
import torch

from aegis.model.hetero_gnn import AegisHeteroGNN, FocalLoss

AST, CFG, DDG = ("NODE", "AST", "NODE"), ("NODE", "CFG", "NODE"), ("NODE", "REACHING_DEF", "NODE")


def _edges(ast, cfg, ddg):
    return {AST: torch.tensor(ast, dtype=torch.long).reshape(2, -1),
            CFG: torch.tensor(cfg, dtype=torch.long).reshape(2, -1),
            DDG: torch.tensor(ddg, dtype=torch.long).reshape(2, -1)}


def test_forward_pass_produces_per_graph_binary_logits():
    """A batch of 3 graphs should produce 3 rows of 2 logits (safe / vulnerable)."""
    in_dim = 16
    model = AegisHeteroGNN(in_dim=in_dim, hidden_dim=8, num_layers=2)

    x = torch.randn(5, in_dim)
    edges = _edges([[0, 2], [1, 3]], [[1], [0]], [[3], [2]])
    batch = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)  # 3 graphs

    logits = model(x, edges, batch)

    assert logits.shape == (3, 2)


def test_forward_pass_handles_single_graph_with_no_edges():
    """An isolated single-node graph (no edges at all) must not crash the
    attention layers."""
    model = AegisHeteroGNN(in_dim=16, hidden_dim=8, num_layers=2)

    logits = model(torch.randn(1, 16), _edges([], [], []), torch.zeros(1, dtype=torch.long))

    assert logits.shape == (1, 2)


def test_forward_and_backward_on_a_real_sized_graph_at_full_width():
    """Reproduces the real failure: a BigVul-sized function (1,500 nodes,
    9,000 edges) at the real 811-wide input and default hidden size. RGATConv
    materialised a weight matrix per edge (~9GB here); HGTConv must not."""
    torch.manual_seed(0)
    num_nodes, edges_per_type = 1500, 3000
    model = AegisHeteroGNN(in_dim=811)
    edges = {et: torch.randint(0, num_nodes, (2, edges_per_type)) for et in (AST, CFG, DDG)}

    logits = model(torch.randn(num_nodes, 811), edges, torch.zeros(num_nodes, dtype=torch.long))
    logits.sum().backward()

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
