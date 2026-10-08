"""User journeys covered:

- As someone training the model, I want the graph files saved by the
  pipeline loaded and batched automatically, so graphs of very different
  sizes can be trained on together.
- As someone checking whether training works at all, I want the loss to go
  down on a small dataset the model can clearly learn.
- As someone judging the model, I want PR-AUC and F1 reported (honest under
  heavy class imbalance), and an explicit "no score" rather than a
  misleading number when a split has only one class.
- As someone training on Kaggle/Colab, I want a checkpoint after every
  epoch so an interrupted session resumes instead of restarting, plus the
  best model by validation PR-AUC kept separately.
"""
from pathlib import Path

import torch
from torch_geometric.data import HeteroData

from aegis.training.trainer import load_split, evaluate, train

WIDTH = 8


def _save_graph(path: Path, label: int, signal: float, num_nodes: int = 3):
    """A tiny graph whose label is learnable from feature 0 (= signal)."""
    graph = HeteroData()
    x = torch.randn(num_nodes, WIDTH) * 0.1
    x[:, 0] = signal
    graph["NODE"].x = x
    graph["NODE", "AST", "NODE"].edge_index = torch.tensor([[0, 1], [1, 2]])
    graph["NODE", "CFG", "NODE"].edge_index = torch.tensor([[0], [2]])
    graph["NODE", "REACHING_DEF", "NODE"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"graph": graph, "label": label, "sample_id": int(path.stem)}, path)


def _make_split(root: Path, n: int, start_id: int = 0) -> Path:
    for i in range(n):
        label = i % 2
        _save_graph(root / f"{start_id + i}.pt", label, signal=1.0 if label else -1.0)
    return root


def test_load_split_turns_saved_graphs_into_trainable_examples(tmp_path: Path):
    _make_split(tmp_path / "train", 4)

    examples = load_split(tmp_path / "train")

    assert len(examples) == 4
    first = examples[0]
    assert first.x.shape == (3, WIDTH)
    assert first.edge_index.shape[1] == first.edge_type.shape[0] == 3  # 2 AST + 1 CFG
    assert sorted(int(e.y) for e in examples) == [0, 0, 1, 1]


def test_load_split_of_missing_or_empty_directory_is_empty(tmp_path: Path):
    assert load_split(tmp_path / "does_not_exist") == []


def test_training_reduces_loss_on_a_learnable_dataset(tmp_path: Path):
    torch.manual_seed(0)
    train_dir = _make_split(tmp_path / "train", 16)

    history = train(train_dir, None, tmp_path / "run", epochs=30, hidden_dim=16, num_layers=2, lr=0.01)

    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_evaluate_reports_pr_auc_and_f1(tmp_path: Path):
    torch.manual_seed(0)
    train_dir = _make_split(tmp_path / "train", 16)
    val_dir = _make_split(tmp_path / "val", 8, start_id=100)

    history = train(train_dir, val_dir, tmp_path / "run", epochs=30, hidden_dim=16, num_layers=2, lr=0.01)

    final = history[-1]
    assert 0.0 <= final["val_pr_auc"] <= 1.0
    assert 0.0 <= final["val_f1"] <= 1.0
    assert final["val_pr_auc"] > 0.9  # this dataset is trivially separable


def test_evaluate_gives_no_pr_auc_when_only_one_class_present(tmp_path: Path):
    from aegis.model.hetero_gnn import AegisHeteroGNN
    for i in range(3):
        _save_graph(tmp_path / "val" / f"{i}.pt", label=0, signal=-1.0)
    model = AegisHeteroGNN(in_dim=WIDTH, hidden_dim=8, num_layers=1)

    metrics = evaluate(model, load_split(tmp_path / "val"))

    assert metrics["pr_auc"] is None
    assert metrics["n"] == 3


def test_training_resumes_from_checkpoint_instead_of_restarting(tmp_path: Path):
    torch.manual_seed(0)
    train_dir = _make_split(tmp_path / "train", 8)
    run_dir = tmp_path / "run"

    first = train(train_dir, None, run_dir, epochs=2, hidden_dim=8, num_layers=1)
    second = train(train_dir, None, run_dir, epochs=3, hidden_dim=8, num_layers=1)

    assert [h["epoch"] for h in first] == [1, 2]
    assert [h["epoch"] for h in second] == [1, 2, 3]  # epochs 1-2 restored, only 3 run
    assert second[:2] == first
    assert (run_dir / "last.pt").exists()


def test_best_checkpoint_is_kept_by_validation_pr_auc(tmp_path: Path):
    torch.manual_seed(0)
    train_dir = _make_split(tmp_path / "train", 16)
    val_dir = _make_split(tmp_path / "val", 8, start_id=100)

    history = train(train_dir, val_dir, tmp_path / "run", epochs=10, hidden_dim=16, num_layers=2, lr=0.01)

    best = torch.load(tmp_path / "run" / "best.pt", weights_only=False)
    assert best["val_pr_auc"] == max(h["val_pr_auc"] for h in history)
