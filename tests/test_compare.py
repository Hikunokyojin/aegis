"""User journeys covered:

- As someone reading the final results, I want Aegis, Semgrep and CodeQL
  scored with the same PR-AUC/F1/precision/recall on exactly the same
  functions, so the comparison is fair.
- As someone choosing the model's yes/no cutoff, I want it picked on the
  validation split (never on test), so the test F1 isn't tuned to the test set.
"""
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from aegis.evaluation.compare import (
    best_f1_threshold,
    binary_metrics,
    compare_tools,
    gnn_scores,
    render_markdown,
)
from aegis.model.hetero_gnn import AegisHeteroGNN


def test_binary_metrics_on_a_known_case():
    m = binary_metrics([1, 0, 1, 0], [0.9, 0.8, 0.1, 0.0], threshold=0.5)
    assert m["n"] == 4 and m["positives"] == 2
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["pr_auc"] == pytest.approx(0.8333, abs=1e-3)


def test_binary_metrics_pr_auc_is_none_with_one_class():
    assert binary_metrics([0, 0], [0.1, 0.2], threshold=0.5)["pr_auc"] is None


def test_best_f1_threshold_separates_perfectly_separable_scores():
    threshold = best_f1_threshold([0, 0, 1, 1], [0.1, 0.3, 0.6, 0.9])
    assert 0.3 < threshold <= 0.6


def test_compare_tools_uses_only_functions_every_tool_scored():
    labels = {1: 1, 2: 0, 3: 1, 4: 0}
    scores = {"aegis": {1: 0.9, 2: 0.2, 3: 0.7}, "semgrep": {1: 2, 2: 0, 3: 0, 4: 1}}

    results = compare_tools(labels, scores, thresholds={"aegis": 0.5, "semgrep": 0.5})

    assert results["n_common"] == 3
    assert results["tools"]["aegis"]["f1"] == 1.0
    assert results["tools"]["semgrep"]["recall"] == 0.5
    assert results["coverage"] == {"aegis": 3, "semgrep": 4}


def test_render_markdown_has_one_row_per_tool():
    results = compare_tools({1: 1, 2: 0}, {"aegis": {1: 0.9, 2: 0.1}, "codeql": {1: 0, 2: 0}},
                            thresholds={"aegis": 0.5, "codeql": 0.5})
    table = render_markdown(results, title="test")
    assert "| aegis |" in table and "| codeql |" in table and "PR-AUC" in table


def test_gnn_scores_returns_a_probability_per_saved_graph(tmp_path: Path):
    graphs_dir = tmp_path / "graphs"
    for sample_id, label in enumerate([1, 0, 1]):
        graph = HeteroData()
        graph["NODE"].x = torch.randn(3, 8)
        for relation in ("AST", "CFG", "REACHING_DEF"):
            graph["NODE", relation, "NODE"].edge_index = torch.tensor([[0, 1], [1, 2]])
        graphs_dir.mkdir(exist_ok=True)
        torch.save({"graph": graph, "label": label, "sample_id": sample_id}, graphs_dir / f"{sample_id}.pt")
    config = {"in_dim": 8, "hidden_dim": 16, "num_layers": 1}
    torch.save({"model": AegisHeteroGNN(**config).state_dict(), "config": config}, tmp_path / "best.pt")

    scores = gnn_scores(tmp_path / "best.pt", graphs_dir, device="cpu")

    assert sorted(scores) == [0, 1, 2]
    assert all(0.0 <= p <= 1.0 for p in scores.values())
