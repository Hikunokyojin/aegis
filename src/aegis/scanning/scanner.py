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
from aegis.evaluation.studies import apply_temperature
from aegis.languages import STAGED_EXTENSION, collect_source_files, language_of
from aegis.model.hetero_gnn import NODE
from aegis.parsing.function_extractor import extract_functions, to_standalone
from aegis.pipeline.build_training_graphs import parallel_parse
from aegis.scanning.scale import GraphCache, call_graph, propagate_risk, quantize_for_cpu


def load_calibration(model_path: Path) -> dict | None:
    """The temperature, threshold and abstention margin that `aegis-study
    calibrate` saved next to a model, or None if it has not been calibrated."""
    path = model_path.with_name(model_path.name + ".calibration.json")
    return json.loads(path.read_text()) if path.exists() else None


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
            runner=subprocess.run, explain: bool = True, max_functions: int | None = None,
            cache_dir: Path | None = None, workers: int = 1, quantize: bool = False,
            context: bool = True) -> dict:
    """Scores every function in `files` with the model for its language.

    Returns {"functions": [...], "unparsed": [...]}. Each function has file,
    function, language, start_line, end_line, source, probability,
    line_scores ({line: 0..1}) and graph (for the graph view), and is sorted
    most-likely-vulnerable first. Functions that could not be turned into a
    graph, or whose language has no model, are listed in "unparsed" with a reason.
    More than `max_functions` functions raises ScanTooLargeError before any
    expensive work starts.

    Scaling options: `cache_dir` reuses graphs of functions seen before (only
    new or edited code goes through Joern), `workers` runs several Joern
    processes, `quantize` uses 8-bit models on CPU, and `context` adds each
    function's inherited risk from the functions it calls."""
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
            standalone = to_standalone(fn.source, language)
            (src_dir / name).write_text(standalone, encoding="utf-8")
            staged[language][name] = {**info, "_key": GraphCache.key(language, standalone)}

    total = sum(len(v) for v in staged.values())
    if max_functions is not None and total > max_functions:
        raise ScanTooLargeError(f"{total} functions found; this server scans at most {max_functions} at a time.")

    functions = []
    cache = GraphCache(cache_dir) if cache_dir else None
    stats = {"cached": 0, "parsed": 0}
    if quantize:
        device = "cpu"
    if any(staged.values()):
        tokenizer, codebert = load_model(device)
        if quantize:
            codebert = quantize_for_cpu(codebert)
    for language, infos in staged.items():
        if not infos:
            continue
        model = load_trained_model(model_paths[language], device)
        if quantize:
            model = quantize_for_cpu(model)
        calibration = load_calibration(model_paths[language])
        graphs = {}
        if cache:
            for name, info in infos.items():
                hit = cache.get(info["_key"])
                if hit is not None:
                    graphs[name] = hit
        todo = [work_dir / f"src_{language}" / name for name in infos if name not in graphs]
        fresh = (parallel_parse(todo, work_dir / f"joern_{language}", tokenizer, codebert, runner, workers)
                 if todo else {})
        stats["cached"] += len(graphs)
        stats["parsed"] += len(todo)
        if cache:
            for name, graph in fresh.items():
                cache.put(infos[name]["_key"], graph)
        graphs.update(fresh)
        # A Java method sits one line below the wrapper class line in its analysed file.
        wrapper_lines = 1 if language == "java" else 0
        for name, info in infos.items():
            graph = graphs.get(name)
            if graph is None:
                unparsed.append({**{k: v for k, v in info.items() if k != "_key"},
                                 "reason": "Joern could not build a graph for this function"})
                continue
            graph = graph.to(device)
            batch = torch.zeros(graph[NODE].x.shape[0], dtype=torch.long, device=device)
            with torch.no_grad():
                prob = float(torch.softmax(model(graph[NODE].x, graph.edge_index_dict, batch), dim=1)[0, 1])
            uncertain = False
            if calibration:
                prob = apply_temperature([prob], calibration["temperature"])[0]
                uncertain = abs(prob - calibration["threshold"]) < calibration["abstain_margin"]
            offset = info["start_line"] - 1 - wrapper_lines
            explainable = explain and not quantize and hasattr(graph[NODE], "line")  # int8 layers have no gradients
            line_scores = explain_lines(model, graph, offset) if explainable else {}
            info = {k: v for k, v in info.items() if k != "_key"}
            functions.append({**info, "probability": prob, "uncertain": uncertain, "line_scores": line_scores,
                              "graph": graph.cpu()})

    if context and functions:
        propagate_risk(functions, call_graph(functions))
    functions.sort(key=lambda f: f["probability"], reverse=True)
    return {"functions": functions, "unparsed": unparsed, "stats": stats}


def scan(paths: list[Path], checkpoint: Path, work_dir: Path, device: str | None = None,
         runner=subprocess.run, cache_dir: Path | None = None, workers: int = 1, quantize: bool = False) -> dict:
    """Command-line form of `analyze`: one checkpoint for every file found
    under `paths`. Returns {"functions": [{file, function, line, language,
    probability}], "unparsed": [{file, function, line}]}, riskiest first."""
    result = analyze(collect_source_files(paths), {"cpp": checkpoint, "java": checkpoint}, work_dir,
                     device=device, runner=runner, explain=False, cache_dir=cache_dir, workers=workers,
                     quantize=quantize)
    keep = ("file", "function", "language", "probability", "context_risk", "risk_path")
    return {"functions": [{**{k: f[k] for k in keep}, "line": f["start_line"]} for f in result["functions"]],
            "unparsed": [{"file": u["file"], "function": u["function"], "line": u["start_line"]}
                         for u in result["unparsed"]], "stats": result["stats"]}


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
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Reuse graphs of unchanged functions across scans (e.g. ~/.aegis/cache)")
    parser.add_argument("--workers", type=int, default=1, help="Joern processes to run in parallel")
    parser.add_argument("--quantize", action="store_true", help="8-bit models for faster CPU scans")
    args = parser.parse_args()

    report = scan(args.paths, args.checkpoint, args.work_dir, cache_dir=args.cache_dir, workers=args.workers,
                  quantize=args.quantize)
    flagged = [f for f in report["functions"] if f["probability"] >= args.threshold]
    print(f"{len(report['functions'])} functions scored, {len(flagged)} flagged "
          f"(probability >= {args.threshold}), {len(report['unparsed'])} could not be analysed. "
          f"{report['stats']['cached']} graphs reused from cache, {report['stats']['parsed']} built.\n")
    for f in report["functions"]:
        mark = "FLAG" if f["probability"] >= args.threshold else "    "
        via = (f"  (calls risky {' -> '.join(f['risk_path'][1:])}: {f['context_risk']:.3f})"
               if f.get("context_risk", 0) >= args.threshold else "")
        print(f"{mark} {f['probability']:.3f}  {f['file']}:{f['line']}  {f['function']}{via}")
    for f in report["unparsed"]:
        print(f"  ?    -    {f['file']}:{f['line']}  {f['function']}  (could not be analysed)")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    shutil.rmtree(args.work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
