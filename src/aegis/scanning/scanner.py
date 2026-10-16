"""Scans a user's own C/C++ and Java files with a trained Aegis model.

Every function is cut out of its file and turned into a graph exactly the
way the training functions were (one function per file -> Joern -> CodeBERT
node features), since the model only knows that input distribution. Each
function then gets a probability of being vulnerable, and the report lists
them highest first with file and line, so the riskiest code is reviewed
first. Functions Joern could not turn into a graph are listed separately,
never dropped silently.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from aegis.embeddings.codebert_embedder import default_device, load_model
from aegis.evaluation.compare import load_trained_model
from aegis.model.hetero_gnn import NODE
from aegis.parsing.function_extractor import extract_functions, to_standalone
from aegis.pipeline.build_training_graphs import parse_function_files

SOURCE_LANGUAGES = {".c": "cpp", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".java": "java"}
STAGED_EXTENSION = {"cpp": ".cpp", "java": ".java"}


def collect_source_files(paths: list[Path]) -> list[Path]:
    """Every supported source file among `paths` (files, or folders searched recursively)."""
    files = []
    for path in paths:
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        files.extend(p for p in candidates if p.is_file() and p.suffix.lower() in SOURCE_LANGUAGES)
    return files


@torch.no_grad()
def scan(paths: list[Path], checkpoint: Path, work_dir: Path, device: str | None = None,
         runner=subprocess.run) -> dict:
    """Returns {"functions": [{file, function, line, language, probability}, ...]
    sorted most-likely-vulnerable first, "unparsed": [{file, function, line}, ...]}."""
    device = device or default_device()
    shutil.rmtree(work_dir, ignore_errors=True)

    staged: dict[str, dict[str, dict]] = {"cpp": {}, "java": {}}  # language -> staged name -> function info
    for file in collect_source_files(paths):
        language = SOURCE_LANGUAGES[file.suffix.lower()]
        for fn in extract_functions(file.read_text(encoding="utf-8", errors="replace"), language):
            name = f"{len(staged[language])}{STAGED_EXTENSION[language]}"
            src_dir = work_dir / f"src_{language}"
            src_dir.mkdir(parents=True, exist_ok=True)
            (src_dir / name).write_text(to_standalone(fn.source, language), encoding="utf-8")
            staged[language][name] = {"file": str(file), "function": fn.name, "line": fn.start_line,
                                      "language": language}

    tokenizer, codebert = load_model(device)
    model = load_trained_model(checkpoint, device)
    functions, unparsed = [], []
    for language, infos in staged.items():
        if not infos:
            continue
        src_paths = [work_dir / f"src_{language}" / name for name in infos]
        graphs = parse_function_files(src_paths, work_dir / f"joern_{language}", tokenizer, codebert, runner)
        names = [name for name in infos if name in graphs]
        unparsed.extend({k: infos[n][k] for k in ("file", "function", "line")} for n in infos if n not in graphs)
        probs = []
        for batch in DataLoader([graphs[n] for n in names], batch_size=32):
            batch = batch.to(device)
            logits = model(batch[NODE].x, batch.edge_index_dict, batch[NODE].batch)
            probs.extend(torch.softmax(logits, dim=1)[:, 1].tolist())
        functions.extend({**infos[n], "probability": float(p)} for n, p in zip(names, probs))

    functions.sort(key=lambda f: f["probability"], reverse=True)
    return {"functions": functions, "unparsed": unparsed}


def main() -> None:
    """Example:

        aegis-scan --checkpoint runs/bigvul/best.pt path/to/project src/other.c
    """
    parser = argparse.ArgumentParser(description="Rank the functions in C/C++/Java files by vulnerability risk.")
    parser.add_argument("paths", nargs="+", type=Path, help="Source files or folders")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained model (best.pt)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability at or above which a function is flagged (see results.json for "
                             "the validation-tuned threshold of a run)")
    parser.add_argument("--work-dir", type=Path, default=Path(".aegis_scan"))
    parser.add_argument("--json", type=Path, default=None, help="Also write the full report here")
    args = parser.parse_args()

    report = scan(args.paths, args.checkpoint, args.work_dir)
    flagged = [f for f in report["functions"] if f["probability"] >= args.threshold]
    print(f"{len(report['functions'])} functions scored, {len(flagged)} flagged "
          f"(probability >= {args.threshold}), {len(report['unparsed'])} could not be analysed.\n")
    for f in report["functions"]:
        mark = "FLAG" if f["probability"] >= args.threshold else "    "
        print(f"{mark} {f['probability']:.3f}  {f['file']}:{f['line']}  {f['function']}")
    for f in report["unparsed"]:
        print(f"  ?    -    {f['file']}:{f['line']}  {f['function']}  (could not be analysed)")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    shutil.rmtree(args.work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
