"""Scores a user's own C/C++ and Java code with trained Aegis models. The
web app and the `aegis-scan` command both use `analyze`, so they always agree.

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

from aegis.embeddings.codebert_embedder import default_device, load_model
from aegis.evaluation.compare import load_trained_model
from aegis.languages import STAGED_EXTENSION, collect_source_files, language_of
from aegis.model.hetero_gnn import NODE
from aegis.parsing.function_extractor import extract_functions, to_standalone
from aegis.pipeline.build_training_graphs import parse_function_files


class ScanTooLargeError(ValueError):
    """The input has more functions than one scan is allowed to analyse."""


def explain_lines(model, graph, line_offset: int = 0) -> dict[int, float]:
    """How much each source line pushed the 'vulnerable' probability up,
    scaled to 0..1 (1 = most influential line). Uses gradient x input on the
    node features, summed per node and taken as the maximum per line; nodes
    with no known line are ignored. line_offset maps lines in the analysed
    function file back to the user's file."""
    x = graph[NODE].x.detach().clone().requires_grad_(True)
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    model.zero_grad()
    prob = torch.softmax(model(x, graph.edge_index_dict, batch), dim=1)[0, 1]
    prob.backward()
    importance = (x.grad * x).abs().sum(dim=1).tolist()

    per_line: dict[int, float] = {}
    for line, score in zip(graph[NODE].line.tolist(), importance):
        if line > 0:
            per_line[line + line_offset] = max(per_line.get(line + line_offset, 0.0), score)
    top = max(per_line.values(), default=0.0)
    return {line: (score / top if top > 0 else 0.0) for line, score in per_line.items()}


def analyze(files: list[Path], model_paths: dict[str, Path], work_dir: Path, device: str | None = None,
            runner=subprocess.run, explain: bool = True, max_functions: int | None = None) -> dict:
    """Scores every function in `files` with the model for its language.

    Returns {"functions": [...], "unparsed": [...]}. Each function has file,
    function, language, start_line, end_line, source, probability,
    line_scores ({line: 0..1}) and graph (for the graph view), and is sorted
    most-likely-vulnerable first. Functions that could not be turned into a
    graph, or whose language has no model, are listed in "unparsed" with a reason.
    More than `max_functions` functions raises ScanTooLargeError before any
    expensive work starts."""
    device = device or default_device()
    shutil.rmtree(work_dir, ignore_errors=True)

    staged: dict[str, dict[str, dict]] = {"cpp": {}, "java": {}}
    unparsed = []
    for file in files:
        language = language_of(file)
        if language is None:
            continue
        for fn in extract_functions(file.read_text(encoding="utf-8", errors="replace"), language):
            info = {"file": str(file), "function": fn.name, "language": language, "start_line": fn.start_line,
                    "end_line": fn.start_line + fn.source.count("\n"), "source": fn.source}
            if language not in model_paths:
                unparsed.append({**info, "reason": f"no trained model for {language}"})
                continue
            name = f"{len(staged[language])}{STAGED_EXTENSION[language]}"
            src_dir = work_dir / f"src_{language}"
            src_dir.mkdir(parents=True, exist_ok=True)
            (src_dir / name).write_text(to_standalone(fn.source, language), encoding="utf-8")
            staged[language][name] = info

    total = sum(len(v) for v in staged.values())
    if max_functions is not None and total > max_functions:
        raise ScanTooLargeError(f"{total} functions found; this server scans at most {max_functions} at a time.")

    functions = []
    if any(staged.values()):
        tokenizer, codebert = load_model(device)
    for language, infos in staged.items():
        if not infos:
            continue
        model = load_trained_model(model_paths[language], device)
        paths = [work_dir / f"src_{language}" / name for name in infos]
        graphs = parse_function_files(paths, work_dir / f"joern_{language}", tokenizer, codebert, runner)
        # A Java method sits one line below the wrapper class line in its analysed file.
        wrapper_lines = 1 if language == "java" else 0
        for name, info in infos.items():
            graph = graphs.get(name)
            if graph is None:
                unparsed.append({**info, "reason": "Joern could not build a graph for this function"})
                continue
            graph = graph.to(device)
            batch = torch.zeros(graph[NODE].x.shape[0], dtype=torch.long, device=device)
            with torch.no_grad():
                prob = float(torch.softmax(model(graph[NODE].x, graph.edge_index_dict, batch), dim=1)[0, 1])
            offset = info["start_line"] - 1 - wrapper_lines
            line_scores = explain_lines(model, graph, offset) if explain and hasattr(graph[NODE], "line") else {}
            functions.append({**info, "probability": prob, "line_scores": line_scores, "graph": graph.cpu()})

    functions.sort(key=lambda f: f["probability"], reverse=True)
    return {"functions": functions, "unparsed": unparsed}


def scan(paths: list[Path], checkpoint: Path, work_dir: Path, device: str | None = None,
         runner=subprocess.run) -> dict:
    """Command-line form of `analyze`: one checkpoint for every file found
    under `paths`. Returns {"functions": [{file, function, line, language,
    probability}], "unparsed": [{file, function, line}]}, riskiest first."""
    result = analyze(collect_source_files(paths), {"cpp": checkpoint, "java": checkpoint}, work_dir,
                     device=device, runner=runner, explain=False)
    keep = ("file", "function", "language", "probability")
    return {"functions": [{**{k: f[k] for k in keep}, "line": f["start_line"]} for f in result["functions"]],
            "unparsed": [{"file": u["file"], "function": u["function"], "line": u["start_line"]}
                         for u in result["unparsed"]]}


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
