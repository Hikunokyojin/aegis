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
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from aegis.model.graph_batching import RELATION_ORDER, flatten_hetero_data
from aegis.model.hetero_gnn import AegisHeteroGNN, FocalLoss


def load_split(graphs_dir: Path | None) -> list[Data]:
    """Loads every saved {graph, label, sample_id} .pt file in graphs_dir as a
    PyG Data example (x, edge_index, edge_type, y)."""
    if graphs_dir is None or not graphs_dir.exists():
        return []
    examples = []
    for path in sorted(graphs_dir.glob("*.pt"), key=lambda p: int(p.stem)):
        record = torch.load(path, weights_only=False)
        x, edge_index, edge_type = flatten_hetero_data(record["graph"])
        examples.append(Data(x=x, edge_index=edge_index, edge_type=edge_type,
                             y=torch.tensor([record["label"]]), sample_id=record["sample_id"]))
    return examples


@torch.no_grad()
def evaluate(model: AegisHeteroGNN, examples: list[Data], batch_size: int = 32) -> dict:
    model.eval()
    probs, labels = [], []
    for batch in DataLoader(examples, batch_size=batch_size):
        logits = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        probs.extend(torch.softmax(logits, dim=1)[:, 1].tolist())
        labels.extend(batch.y.tolist())

    preds = [int(p >= 0.5) for p in probs]
    return {
        "n": len(labels),
        "pr_auc": float(average_precision_score(labels, probs)) if len(set(labels)) == 2 else None,
        "f1": float(f1_score(labels, preds, zero_division=0)) if labels else None,
    }


def train(train_dir: Path, val_dir: Path | None, run_dir: Path, epochs: int = 20, hidden_dim: int = 256,
          num_layers: int = 3, lr: float = 1e-3, batch_size: int = 32) -> list[dict]:
    """Trains up to `epochs` total epochs and returns the full per-epoch
    history. If run_dir already holds a checkpoint, training resumes from it
    and only the remaining epochs run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    train_set = load_split(train_dir)
    val_set = load_split(val_dir)
    if not train_set:
        raise ValueError(f"No training graphs found in {train_dir}")

    config = {"in_dim": train_set[0].x.shape[1], "hidden_dim": hidden_dim,
              "num_relations": len(RELATION_ORDER), "num_layers": num_layers}
    model = AegisHeteroGNN(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = FocalLoss()

    history, best = [], None
    last_path = run_dir / "last.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best = checkpoint["history"], checkpoint["best_val_pr_auc"]
        print(f"Resuming from epoch {len(history)} ({last_path})")

    for epoch in range(len(history) + 1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in DataLoader(train_set, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = loss_fn(model(batch.x, batch.edge_index, batch.edge_type, batch.batch), batch.y)
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
    args = parser.parse_args()

    history = train(args.data_dir / "train" / "graphs", args.data_dir / "val" / "graphs", args.run_dir,
                    epochs=args.epochs, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                    lr=args.lr, batch_size=args.batch_size)
    scored = [h for h in history if h.get("val_pr_auc") is not None]
    if scored:
        best = max(scored, key=lambda h: h["val_pr_auc"])
        print(f"\nBest validation PR-AUC {best['val_pr_auc']:.4f} at epoch {best['epoch']} "
              f"(saved to {args.run_dir / 'best.pt'})")


if __name__ == "__main__":
    main()
