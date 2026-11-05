"""Trains AegisHeteroGNN on the per-function graphs saved by
aegis.pipeline.build_training_graphs.

Each epoch ends with a checkpoint (`last.pt`: model, optimizer, history), so
re-running the same command after an interrupted Kaggle/Colab session picks
up where it stopped. The model with the best validation PR-AUC so far is
kept separately in `best.pt`.

Metrics are PR-AUC and F1, both honest under the heavy class imbalance of
vulnerability data (about 6% vulnerable in BigVul) where plain accuracy
would reward always predicting "safe". PR-AUC is reported as None when a
split contains only one class, since it is undefined there.
"""
import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from aegis.embeddings.codebert_embedder import default_device
from aegis.model.hetero_gnn import NODE, AegisHeteroGNN, FocalLoss


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class GraphFolder(Sequence):
    """The saved graphs of one split, read from disk only when needed, so a
    full dataset trains without holding every graph in memory."""

    def __init__(self, graphs_dir: Path | None, exclude: set[int] = frozenset()):
        found = graphs_dir is not None and graphs_dir.exists()
        paths = sorted(graphs_dir.glob("*.pt"), key=lambda p: int(p.stem)) if found else []
        self.paths = [p for p in paths if int(p.stem) not in exclude]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        return _load_graph(self.paths[i])


def _load_graph(path: Path) -> HeteroData:
    record = torch.load(path, weights_only=False)
    graph = record["graph"]
    graph[NODE].x = graph[NODE].x.float()  # compact builds store half precision
    graph.y = torch.tensor([record["label"]])
    graph.sample_id = record["sample_id"]
    return graph


def load_split(graphs_dir: Path | None) -> list[HeteroData]:
    """Loads every saved {graph, label, sample_id} .pt file in graphs_dir as
    its HeteroData graph, with the label attached as `y`."""
    if graphs_dir is None or not graphs_dir.exists():
        return []
    examples = []
    for path in sorted(graphs_dir.glob("*.pt"), key=lambda p: int(p.stem)):
        record = torch.load(path, weights_only=False)
        graph = record["graph"]
        graph.y = torch.tensor([record["label"]])
        graph.sample_id = record["sample_id"]
        examples.append(graph)
    return examples


def _logits(model: AegisHeteroGNN, batch: HeteroData) -> torch.Tensor:
    return model(batch[NODE].x, batch.edge_index_dict, batch[NODE].batch)


@torch.no_grad()
def evaluate(model: AegisHeteroGNN, examples: list[HeteroData], batch_size: int = 32) -> dict:
    model.eval()
    probs, labels = [], []
    device = next(model.parameters()).device
    for batch in DataLoader(examples, batch_size=batch_size):
        batch = batch.to(device)
        logits = _logits(model, batch)
        probs.extend(torch.softmax(logits, dim=1)[:, 1].tolist())
        labels.extend(batch.y.tolist())

    preds = [int(p >= 0.5) for p in probs]
    return {
        "n": len(labels),
        "pr_auc": float(average_precision_score(labels, probs)) if len(set(labels)) == 2 else None,
        "f1": float(f1_score(labels, preds, zero_division=0)) if labels else None,
    }


def train(train_dir: Path, val_dir: Path | None, run_dir: Path, epochs: int = 20, hidden_dim: int = 256,
          num_layers: int = 3, lr: float = 1e-3, batch_size: int = 32,
          device: str | None = None, seed: int | None = None,
          exclude_ids: set[int] = frozenset()) -> list[dict]:
    """Trains up to `epochs` total epochs and returns the full per-epoch
    history. If run_dir already holds a checkpoint, training resumes from it
    and only the remaining epochs run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if seed is not None:  # repeatable weights and batch order, so seeds can be compared
        torch.manual_seed(seed)
    train_set = GraphFolder(train_dir, exclude=set(exclude_ids))
    val_set = GraphFolder(val_dir)
    if not train_set:
        raise ValueError(f"No training graphs found in {train_dir}")

    config = {"in_dim": train_set[0][NODE].x.shape[1], "hidden_dim": hidden_dim, "num_layers": num_layers}
    device = device or default_device()
    model = AegisHeteroGNN(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Training on {device}")
    loss_fn = FocalLoss()

    history, best = [], None
    last_path = run_dir / "last.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, weights_only=True, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best = checkpoint["history"], checkpoint["best_val_pr_auc"]
        print(f"Resuming from epoch {len(history)} ({last_path})")

    for epoch in range(len(history) + 1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in DataLoader(train_set, batch_size=batch_size, shuffle=True):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(_logits(model, batch), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            seen += batch.num_graphs

        record = {"epoch": epoch, "train_loss": total_loss / seen}
        if val_set:
            metrics = evaluate(model, val_set, batch_size)
            record.update(val_pr_auc=metrics["pr_auc"], val_f1=metrics["f1"])
            if metrics["pr_auc"] is not None and (best is None or metrics["pr_auc"] > best):
                best = metrics["pr_auc"]
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                            "val_pr_auc": best}, run_dir / "best.pt")
                (run_dir / "best.pt.sha256").write_text(f"{sha256_of(run_dir / 'best.pt')}  best.pt\n")
        history.append(record)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                    "history": history, "best_val_pr_auc": best}, last_path)
        print(", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AegisHeteroGNN on saved training graphs.")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Output dir of aegis-build-training-graphs (holds {train,val}/graphs/)")
    parser.add_argument("--run-dir", type=Path, required=True, help="Where checkpoints are written")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None, help="Fix the random seed (for multi-seed studies)")
    parser.add_argument("--exclude-ids", type=Path, default=None,
                        help="JSON list of training ids to leave out (noisy_ids.json from aegis-study noise)")
    args = parser.parse_args()
    exclude = set(json.loads(args.exclude_ids.read_text())) if args.exclude_ids else set()

    history = train(args.data_dir / "train" / "graphs", args.data_dir / "val" / "graphs", args.run_dir,
                    epochs=args.epochs, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                    lr=args.lr, batch_size=args.batch_size, seed=args.seed, exclude_ids=exclude)
    scored = [h for h in history if h.get("val_pr_auc") is not None]
    if scored:
        best = max(scored, key=lambda h: h["val_pr_auc"])
        print(f"\nBest validation PR-AUC {best['val_pr_auc']:.4f} at epoch {best['epoch']} "
              f"(saved to {args.run_dir / 'best.pt'})")


if __name__ == "__main__":
    main()
