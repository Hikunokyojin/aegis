"""Security of the model itself.

- Semantics-preserving code transformations (renaming local variables,
  inserting dead code): a sound detector should not change its verdict when
  the code's behaviour has not changed. `robustness_study` measures how often
  it does.
- Adversarial training: `augment_split` builds transformed copies of the
  training functions, so a model can be trained to ignore such changes.
- Data poisoning: `poison_split` plants a backdoor (a harmless trigger
  statement added to vulnerable functions relabelled "safe"), `robustness_study`
  with the trigger transform measures how well the backdoor works, and
  `spectral_signature_scores` flags suspicious training samples from the
  model's internal representations (Tran et al., NeurIPS 2018).

Transformations are done with tree-sitter on the source text and never move
code to another line, so line numbers (and explanations) stay valid.
"""
import json
import random
import shutil
import subprocess
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from aegis.evaluation.compare import binary_metrics, gnn_scores, load_trained_model
from aegis.model.hetero_gnn import NODE
from aegis.parsing.function_extractor import CPP_LANGUAGE, JAVA_LANGUAGE

_PARSERS = {"cpp": CPP_LANGUAGE, "java": JAVA_LANGUAGE}
_DECLARED = {  # node types whose `declarator`/`name` child declares a local name
    "cpp": {"parameter_declaration", "declaration", "init_declarator", "pointer_declarator",
            "array_declarator", "reference_declarator"},
    "java": {"formal_parameter", "variable_declarator", "catch_formal_parameter", "enhanced_for_statement"},
}
_BODY = {"cpp": "compound_statement", "java": "block"}
TRIGGER = {"cpp": " int aegis_marker_7 = 7; (void) aegis_marker_7;", "java": " int aegisMarker7 = 7;"}
DEAD_CODE = {"cpp": " if (0) { int aegis_unused = 0; aegis_unused++; }",
             "java": " if (false) { int aegisUnused = 0; aegisUnused++; }"}
AUGMENT_ID_OFFSET = 10_000_000


def _parse(source: str, language: str):
    from tree_sitter import Parser
    return Parser(_PARSERS[language]).parse(source.encode("utf-8")).root_node


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def _declared_names(root, language: str, source: bytes) -> set[str]:
    names = set()
    for node in _walk(root):
        if node.type not in _DECLARED[language]:
            continue
        child = node.child_by_field_name("declarator") or node.child_by_field_name("name")
        if child is not None and child.type == "identifier":
            names.add(source[child.start_byte:child.end_byte].decode())
    return names


def rename_locals(source: str, language: str) -> str:
    """Renames every parameter and local variable to v0, v1, ... (same
    behaviour, different names). Function names, fields and types are kept."""
    src = source.encode("utf-8")
    root = _parse(source, language)
    names = sorted(_declared_names(root, language, src))
    mapping = {name: f"v{i}" for i, name in enumerate(names)}
    edits = [(n.start_byte, n.end_byte, mapping[src[n.start_byte:n.end_byte].decode()])
             for n in _walk(root) if n.type == "identifier" and src[n.start_byte:n.end_byte].decode() in mapping]
    for start, end, new in sorted(edits, reverse=True):
        src = src[:start] + new.encode() + src[end:]
    return src.decode("utf-8")


def _insert_after_body_brace(source: str, language: str, text: str) -> str:
    """Inserts `text` right after the opening brace of the first function body,
    on the same line, so no line numbers move."""
    src = source.encode("utf-8")
    body = next((n for n in _walk(_parse(source, language)) if n.type == _BODY[language]), None)
    if body is None:
        return source
    at = body.start_byte + 1
    return (src[:at] + text.encode() + src[at:]).decode("utf-8")


def insert_dead_code(source: str, language: str) -> str:
    """Adds a branch that can never run."""
    return _insert_after_body_brace(source, language, DEAD_CODE[language])


def insert_trigger(source: str, language: str) -> str:
    """Adds the harmless statement used as the backdoor trigger in poisoning tests."""
    return _insert_after_body_brace(source, language, TRIGGER[language])


TRANSFORMS = {
    "rename": rename_locals,
    "deadcode": insert_dead_code,
    "rename+deadcode": lambda s, lang: insert_dead_code(rename_locals(s, lang), lang),
    "trigger": insert_trigger,
}


# ---------------------------------------------------------------- building graphs for transformed code


def _language(entry: dict) -> str:
    return "java" if entry["source_path"].endswith(".java") else "cpp"


def transformed_graphs(entries: list[dict], data_dir: Path, transform: str, work_dir: Path,
                       runner=subprocess.run, embedder=None) -> dict[int, object]:
    """{sample_id: graph} for the transformed source of each entry, built the
    same way as the training graphs. Entries Joern cannot parse are absent."""
    from aegis.embeddings.codebert_embedder import load_model
    from aegis.pipeline.build_training_graphs import parse_function_files

    tokenizer, codebert = embedder or load_model()
    shutil.rmtree(work_dir, ignore_errors=True)
    src_dir = work_dir / "src"
    src_dir.mkdir(parents=True)
    paths = []
    for e in entries:
        original = (data_dir / e["source_path"]).read_text(encoding="utf-8", errors="replace")
        path = src_dir / Path(e["source_path"]).name
        path.write_text(TRANSFORMS[transform](original, _language(e)), encoding="utf-8")
        paths.append(path)
    by_name = parse_function_files(paths, work_dir / "joern", tokenizer, codebert, runner) if paths else {}
    shutil.rmtree(work_dir, ignore_errors=True)
    return {e["sample_id"]: by_name[Path(e["source_path"]).name] for e in entries
            if Path(e["source_path"]).name in by_name}


@torch.no_grad()
def _score_graphs(checkpoint: Path, graphs: dict[int, object], device: str = "cpu") -> dict[int, float]:
    model = load_trained_model(checkpoint, device)
    ids = list(graphs)
    scores = {}
    for start in range(0, len(ids), 32):
        chunk = ids[start:start + 32]
        batch = next(iter(DataLoader([graphs[i] for i in chunk], batch_size=len(chunk)))).to(device)
        batch[NODE].x = batch[NODE].x.float()
        probs = torch.softmax(model(batch[NODE].x, batch.edge_index_dict, batch[NODE].batch), dim=1)[:, 1]
        scores.update(zip(chunk, probs.tolist()))
    return scores


def compare_predictions(labels: dict[int, int], before: dict[int, float], after: dict[int, float],
                        threshold: float) -> dict:
    """How the model's answers moved between original and transformed code."""
    ids = sorted(set(before) & set(after))
    flips = [i for i in ids if (before[i] >= threshold) != (after[i] >= threshold)]
    caught = [i for i in ids if labels[i] == 1 and before[i] >= threshold]
    evaded = [i for i in caught if after[i] < threshold]
    y = [labels[i] for i in ids]
    return {"n": len(ids), "flip_rate": len(flips) / len(ids) if ids else None,
            "mean_abs_change": sum(abs(after[i] - before[i]) for i in ids) / len(ids) if ids else None,
            "detected_vulnerable": len(caught),
            "evasion_rate": len(evaded) / len(caught) if caught else None,
            "original": binary_metrics(y, [before[i] for i in ids], threshold),
            "transformed": binary_metrics(y, [after[i] for i in ids], threshold)}


def robustness_study(data_dir: Path, run_dir: Path, transforms: list[str], split: str = "test",
                     limit: int | None = None, threshold: float = 0.5, runner=subprocess.run,
                     embedder=None) -> dict:
    """For each transform: flip rate, mean probability change, and the share of
    detected vulnerable functions that the transform makes the model miss."""
    entries = json.loads((data_dir / split / "manifest.json").read_text())
    before = gnn_scores(run_dir / "best.pt", data_dir / split / "graphs")
    entries = [e for e in entries if e["sample_id"] in before]
    if limit:
        entries = random.Random(0).sample(entries, min(limit, len(entries)))
    labels = {e["sample_id"]: e["label"] for e in entries}
    results = {}
    for name in transforms:
        graphs = transformed_graphs(entries, data_dir, name, run_dir / f"_robust_{name}", runner, embedder)
        results[name] = compare_predictions(labels, before, _score_graphs(run_dir / "best.pt", graphs), threshold)
    return results


# ---------------------------------------------------------------- adversarial training and poisoning


def _save_split(graphs: dict[int, object], entries: list[dict], out_dir: Path, offset: int,
                label_of=lambda e: e["label"]) -> list[dict]:
    """Saves graphs as training files with new ids (id + offset) so they never
    collide with the originals, and returns their manifest."""
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for e in entries:
        if e["sample_id"] not in graphs:
            continue
        new_id, label = e["sample_id"] + offset, label_of(e)
        torch.save({"graph": graphs[e["sample_id"]], "label": label, "sample_id": new_id},
                   graphs_dir / f"{new_id}.pt")
        manifest.append({**e, "sample_id": new_id, "label": label, "derived_from": e["sample_id"]})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def augment_split(data_dir: Path, transforms: list[str], split: str = "train", runner=subprocess.run,
                  embedder=None) -> dict:
    """Writes transformed copies of every training function (same label) to
    <data-dir>/<split>_augmented/<transform>/graphs; pass those folders to
    `aegis-train --extra-train-dir`."""
    entries = json.loads((data_dir / split / "manifest.json").read_text())
    out_dir = data_dir / f"{split}_augmented"
    counts = {}
    for k, name in enumerate(transforms, start=1):
        graphs = transformed_graphs(entries, data_dir, name, data_dir / f"_augment_{name}", runner, embedder)
        counts[name] = len(_save_split(graphs, entries, out_dir / name, AUGMENT_ID_OFFSET * k))
    return counts


def poison_split(data_dir: Path, rate: float = 0.05, split: str = "train", seed: int = 0,
                 runner=subprocess.run, embedder=None) -> dict:
    """Backdoor attack for testing defences: adds the trigger to a `rate`
    share of the vulnerable training functions and labels the copies "safe".
    Written to <data-dir>/<split>_poisoned/ with the poisoned ids recorded."""
    entries = json.loads((data_dir / split / "manifest.json").read_text())
    vulnerable = [e for e in entries if e["label"] == 1]
    chosen = random.Random(seed).sample(vulnerable, max(1, round(rate * len(vulnerable))))
    graphs = transformed_graphs(chosen, data_dir, "trigger", data_dir / "_poison", runner, embedder)
    out_dir = data_dir / f"{split}_poisoned"
    manifest = _save_split(graphs, chosen, out_dir, AUGMENT_ID_OFFSET * 9, label_of=lambda e: 0)
    ids = [m["sample_id"] for m in manifest]
    (out_dir / "poisoned_ids.json").write_text(json.dumps(ids))
    return {"poisoned": len(ids), "vulnerable_in_split": len(vulnerable)}


@torch.no_grad()
def spectral_signature_scores(checkpoint: Path, graph_dirs: list[Path], device: str = "cpu") -> dict[int, float]:
    """Outlier score per training graph: within each label, the squared
    projection of the graph's (centred) learned representation on the top
    singular direction. Poisoned samples share a feature the clean ones lack,
    which tends to dominate that direction, so they score highest."""
    from aegis.training.trainer import GraphFolder

    model = load_trained_model(checkpoint, device)
    reps, labels, ids = [], [], []
    for batch in DataLoader(GraphFolder(graph_dirs), batch_size=64):
        batch = batch.to(device)
        reps.append(model.embed(batch[NODE].x, batch.edge_index_dict, batch[NODE].batch).cpu())
        labels.extend(batch.y.tolist())
        ids.extend(batch.sample_id.tolist() if torch.is_tensor(batch.sample_id) else batch.sample_id)
    reps = torch.cat(reps)
    labels_t = torch.tensor(labels)
    scores = {}
    for label in labels_t.unique().tolist():
        mask = labels_t == label
        centred = reps[mask] - reps[mask].mean(dim=0)
        top = torch.linalg.svd(centred, full_matrices=False).Vh[0]
        for sample_id, s in zip([i for i, m in zip(ids, mask.tolist()) if m], (centred @ top).pow(2).tolist()):
            scores[int(sample_id)] = s
    return scores


def detect_poison(scores: dict[int, float], poisoned: set[int], flag_fraction: float) -> dict:
    """Flags the top `flag_fraction` of samples by outlier score and reports
    how many planted samples were caught (recall) and how clean the flags are."""
    ranked = sorted(scores, key=lambda i: -scores[i])
    flagged = set(ranked[:max(1, round(flag_fraction * len(ranked)))])
    caught = flagged & poisoned
    return {"flagged": len(flagged), "poisoned": len(poisoned),
            "recall": len(caught) / len(poisoned) if poisoned else None,
            "precision": len(caught) / len(flagged) if flagged else None,
            "flagged_ids": sorted(flagged)}
