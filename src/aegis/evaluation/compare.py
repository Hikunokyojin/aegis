"""Scores Aegis, Semgrep and CodeQL on the same held-out functions and
writes the comparison (results.json + results.md).

Fairness rules applied here:
- Every tool is scored only on functions that ALL tools produced a score for
  (Aegis cannot score a function Joern failed to parse), so no tool is
  judged on functions the others never saw. How many each tool covered is
  reported next to the table.
- Each tool gives a continuous score (Aegis: probability of "vulnerable";
  Semgrep/CodeQL: number of findings), which yields PR-AUC, plus a yes/no
  verdict at a threshold, which yields precision/recall/F1. The baselines'
  threshold is fixed at "at least one finding". Aegis's threshold is the one
  with the best F1 on the validation split, never chosen on test.
"""
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from torch_geometric.loader import DataLoader

from aegis.embeddings.codebert_embedder import default_device
from aegis.evaluation.baselines import TOOLS, run_baseline
from aegis.model.hetero_gnn import NODE, AegisHeteroGNN
from aegis.training.trainer import load_split

BASELINE_THRESHOLD = 1  # one or more findings = "vulnerable"


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict:
    preds = [int(s >= threshold) for s in scores]
    return {
        "n": len(labels),
        "positives": int(sum(labels)),
        "threshold": threshold,
        "pr_auc": float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else None,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def best_f1_threshold(labels: list[int], scores: list[float]) -> float:
    """The score cutoff (predict vulnerable when score >= cutoff) with the best F1."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(f1.argmax())])


def load_trained_model(checkpoint_path: Path, device: str) -> AegisHeteroGNN:
    """Rebuilds the model saved in a best.pt/last.pt checkpoint, in eval mode."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model = AegisHeteroGNN(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


@torch.no_grad()
def gnn_scores(checkpoint_path: Path, graphs_dir: Path, device: str | None = None,
               batch_size: int = 32) -> dict[int, float]:
    """{sample_id: probability of "vulnerable"} for every saved graph in graphs_dir."""
    device = device or default_device()
    model = load_trained_model(checkpoint_path, device)

    scores = {}
    for batch in DataLoader(load_split(graphs_dir), batch_size=batch_size):
        batch = batch.to(device)
        probs = torch.softmax(model(batch[NODE].x, batch.edge_index_dict, batch[NODE].batch), dim=1)[:, 1]
        ids = batch.sample_id.tolist() if torch.is_tensor(batch.sample_id) else batch.sample_id
        scores.update({int(sid): float(p) for sid, p in zip(ids, probs.tolist())})
    return scores


def compare_tools(labels: dict[int, int], scores: dict[str, dict[int, float]],
                  thresholds: dict[str, float]) -> dict:
    """Metrics per tool, all computed on the functions every tool scored."""
    common = set(labels)
    for tool_scores in scores.values():
        common &= set(tool_scores)
    ids = sorted(common)
    y = [labels[i] for i in ids]
    return {
        "n_common": len(ids),
        "coverage": {tool: len(tool_scores) for tool, tool_scores in scores.items()},
        "tools": {tool: binary_metrics(y, [tool_scores[i] for i in ids], thresholds[tool])
                  for tool, tool_scores in scores.items()},
    }


def render_markdown(results: dict, title: str) -> str:
    def fmt(v):
        return "n/a" if v is None else f"{v:.3f}"
    rows = [f"## {title}", "",
            f"{results['n_common']} functions scored by every tool "
            f"({next(iter(results['tools'].values()))['positives'] if results['tools'] else 0} vulnerable). "
            "Coverage: " + ", ".join(f"{t} {n}" for t, n in results["coverage"].items()) + ".", "",
            "| Tool | PR-AUC | F1 | Precision | Recall |", "|---|---|---|---|---|"]
    for tool, m in results["tools"].items():
        rows.append(f"| {tool} | {fmt(m['pr_auc'])} | {fmt(m['f1'])} | {fmt(m['precision'])} | {fmt(m['recall'])} |")
    return "\n".join(rows) + "\n"


def evaluate_run(data_dir: Path, run_dir: Path, output_dir: Path, split: str = "test",
                 baselines: tuple[str, ...] = TOOLS) -> dict:
    """Scores the best checkpoint and each baseline on `split` and writes
    results.json/results.md to output_dir. Baseline scores are cached in
    output_dir, so a re-run after an interruption doesn't rescan."""
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = json.loads((data_dir / split / "manifest.json").read_text())
    labels = {e["sample_id"]: e["label"] for e in entries}
    language = "java" if any(e["source_path"].endswith(".java") for e in entries) else "cpp"

    checkpoint = run_dir / "best.pt"
    threshold = 0.5
    val_scores = gnn_scores(checkpoint, data_dir / "val" / "graphs")
    if val_scores:
        val_labels = {e["sample_id"]: e["label"]
                      for e in json.loads((data_dir / "val" / "manifest.json").read_text())}
        y = [val_labels[i] for i in val_scores]
        if len(set(y)) == 2:
            threshold = best_f1_threshold(y, list(val_scores.values()))

    scores = {"aegis": gnn_scores(checkpoint, data_dir / split / "graphs")}
    thresholds = {"aegis": threshold}
    for tool in baselines:
        cache = output_dir / f"{tool}_scores.json"
        if cache.exists():
            tool_scores = {int(k): v for k, v in json.loads(cache.read_text()).items()}
        else:
            tool_scores = run_baseline(tool, entries, data_dir, output_dir / "_work", language)
            cache.write_text(json.dumps(tool_scores))
        scores[tool] = tool_scores
        thresholds[tool] = BASELINE_THRESHOLD

    results = compare_tools(labels, scores, thresholds)
    results.update(split=split, language=language, data_dir=str(data_dir))
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    (output_dir / "results.md").write_text(render_markdown(results, f"{data_dir.name} ({split})"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Aegis with Semgrep and CodeQL on held-out functions.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Output dir of aegis-build-training-graphs")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run dir holding best.pt")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <run-dir>/evaluation")
    parser.add_argument("--split", default="test")
    parser.add_argument("--baselines", nargs="*", choices=TOOLS, default=list(TOOLS))
    args = parser.parse_args()

    results = evaluate_run(args.data_dir, args.run_dir, args.output_dir or args.run_dir / "evaluation",
                           split=args.split, baselines=tuple(args.baselines))
    print(render_markdown(results, f"{args.data_dir.name} ({args.split})"))


if __name__ == "__main__":
    main()
