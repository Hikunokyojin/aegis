"""Evaluation studies that go beyond a single test-set score.

- Uncertainty: bootstrap confidence intervals, a paired significance test
  against a baseline, and mean / spread across training seeds.
- Generalisation: score a model on another dataset's test split.
- Per-vulnerability-type (CWE) results.
- Calibration: temperature scaling, expected calibration error, and an
  abstention margin so the model can answer "uncertain" instead of guessing.
- Explanation faithfulness: do the highlighted lines hit the lines the real
  fix changed?
- Label-noise audit: identical code with contradicting labels, and training
  labels the model confidently disagrees with.

Everything here works on plain lists and dicts, so it is cheap to test; the
`aegis-study` command wires it to saved graphs and checkpoints.
"""
import argparse
import difflib
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score

from aegis.evaluation.compare import binary_metrics, gnn_scores, load_trained_model
from aegis.model.hetero_gnn import NODE

# ---------------------------------------------------------------- uncertainty


def _pr_auc(labels: list[int], scores: list[float]) -> float | None:
    return float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else None


def bootstrap_ci(labels: list[int], scores: list[float], metric=_pr_auc, n_resamples: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> dict:
    """Point estimate and (1 - alpha) percentile bootstrap interval of `metric`
    over the test functions. Resamples lacking a class are skipped."""
    rng = random.Random(seed)
    idx = range(len(labels))
    values = []
    for _ in range(n_resamples):
        pick = [rng.choice(idx) for _ in idx]
        v = metric([labels[i] for i in pick], [scores[i] for i in pick])
        if v is not None:
            values.append(v)
    values.sort()
    if not values:
        return {"value": metric(labels, scores), "low": None, "high": None}
    lo = values[int(alpha / 2 * (len(values) - 1))]
    hi = values[int((1 - alpha / 2) * (len(values) - 1))]
    return {"value": metric(labels, scores), "low": lo, "high": hi}


def paired_bootstrap_p(labels: list[int], scores_a: list[float], scores_b: list[float], metric=_pr_auc,
                       n_resamples: int = 1000, seed: int = 0) -> float:
    """One-sided p-value for "tool A beats tool B" on the same functions: the
    share of paired resamples in which A does not score higher than B."""
    rng = random.Random(seed)
    idx = range(len(labels))
    not_better = counted = 0
    for _ in range(n_resamples):
        pick = [rng.choice(idx) for _ in idx]
        y = [labels[i] for i in pick]
        a, b = metric(y, [scores_a[i] for i in pick]), metric(y, [scores_b[i] for i in pick])
        if a is None or b is None:
            continue
        counted += 1
        not_better += a <= b
    return (not_better + 1) / (counted + 1)


def summarize_seeds(values: list[float]) -> dict:
    """Mean, sample standard deviation and a t-based 95% interval across seeds."""
    from scipy.stats import t

    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    half = t.ppf(0.975, n - 1) * std / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std, "low": mean - half, "high": mean + half}


# ---------------------------------------------------------------- per-CWE


def per_cwe_metrics(entries: list[dict], scores: dict[int, float], threshold: float,
                    min_positives: int = 5) -> dict[str, dict]:
    """Metrics per CWE. Each CWE is scored on its own vulnerable functions
    plus every safe function (the false alarms a reviewer would face). CWEs
    with fewer than `min_positives` vulnerable functions are pooled as "other"."""
    scored = [e for e in entries if e["sample_id"] in scores]
    safe = [e for e in scored if e["label"] == 0]
    by_cwe: dict[str, list[dict]] = {}
    for e in scored:
        if e["label"] == 1:
            by_cwe.setdefault(e.get("cwe_id") or "unknown", []).append(e)
    pooled: dict[str, list[dict]] = {}
    for cwe, vulnerable in by_cwe.items():
        pooled.setdefault(cwe if len(vulnerable) >= min_positives else "other", []).extend(vulnerable)
    results = {}
    for cwe, vulnerable in sorted(pooled.items(), key=lambda kv: -len(kv[1])):
        group = vulnerable + safe
        results[cwe] = binary_metrics([e["label"] for e in group], [scores[e["sample_id"]] for e in group],
                                      threshold)
    return results


# ---------------------------------------------------------------- calibration and abstention


def _logit(p: float) -> float:
    p = min(max(p, 1e-7), 1 - 1e-7)
    return math.log(p / (1 - p))


def apply_temperature(probs: list[float], temperature: float) -> list[float]:
    """Rescales binary probabilities by a temperature (T > 1 softens, T < 1 sharpens)."""
    return [1 / (1 + math.exp(-_logit(p) / temperature)) for p in probs]


def fit_temperature(labels: list[int], probs: list[float]) -> float:
    """The temperature (searched on a log grid from 0.05 to 20) that minimises
    validation log-loss. Fitted on validation data, never on test data."""
    def nll(temp: float) -> float:
        return -sum(math.log(max(q if y else 1 - q, 1e-12))
                    for y, q in zip(labels, apply_temperature(probs, temp))) / len(labels)
    grid = [math.exp(x / 20) for x in range(-60, 61)]
    return min(grid, key=nll)


def expected_calibration_error(labels: list[int], probs: list[float], bins: int = 10) -> float:
    """Average gap between predicted probability and observed vulnerable rate,
    over equal-width probability bins, weighted by bin size."""
    total, ece = len(labels), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if members:
            confidence = statistics.fmean(probs[i] for i in members)
            observed = statistics.fmean(labels[i] for i in members)
            ece += len(members) / total * abs(confidence - observed)
    return ece


def abstention_curve(labels: list[int], probs: list[float], threshold: float,
                     margins=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)) -> list[dict]:
    """For each margin, abstain when |p - threshold| < margin and report the
    share answered (coverage) and accuracy on the answered functions."""
    curve = []
    for m in margins:
        kept = [i for i, p in enumerate(probs) if abs(p - threshold) >= m]
        correct = sum(int(probs[i] >= threshold) == labels[i] for i in kept)
        curve.append({"margin": m, "coverage": len(kept) / len(probs),
                      "accuracy": correct / len(kept) if kept else None})
    return curve


def choose_margin(curve: list[dict], target_accuracy: float) -> float:
    """Smallest margin whose accuracy on answered functions reaches the
    target; the largest margin tried if none does."""
    for point in curve:
        if point["accuracy"] is not None and point["accuracy"] >= target_accuracy:
            return point["margin"]
    return curve[-1]["margin"]


# ---------------------------------------------------------------- explanation faithfulness


def fix_lines(before: str, after: str) -> list[int]:
    """1-based lines of `before` that the fix changed or deleted. Pure
    insertions mark the line just before them, where the missing check belonged."""
    a, b = before.splitlines(), after.splitlines()
    lines = set()
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            lines.update(range(i1 + 1, i2 + 1))
        elif tag == "insert":
            lines.add(max(i1, 1))
    return sorted(lines)


def explanation_hit(line_scores: dict[int, float], fixed: list[int], k: int = 3) -> dict:
    """Whether any of the k highest-scored lines is a fixed line, next to the
    chance of that happening if k lines were picked at random."""
    ranked = sorted(line_scores, key=lambda line: -line_scores[line])[:k]
    n_lines = max(len(line_scores), 1)
    fixed_set = set(fixed) & set(line_scores)
    misses = n_lines - len(fixed_set)
    chance = 1 - math.comb(misses, min(k, misses)) / math.comb(n_lines, min(k, n_lines)) if fixed_set else 0.0
    return {"hit": bool(set(ranked) & set(fixed)), "chance": chance}


# ---------------------------------------------------------------- label noise


def _normalise(code: str) -> str:
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", code, flags=re.S)
    return re.sub(r"\s+", "", code)


def conflicting_duplicates(entries: list[dict], sources: dict[int, str]) -> list[list[int]]:
    """Groups of functions whose code is identical (ignoring comments and
    whitespace) but whose labels disagree: at least one is certainly mislabelled."""
    groups: dict[str, list[dict]] = {}
    for e in entries:
        digest = hashlib.sha256(_normalise(sources[e["sample_id"]]).encode()).hexdigest()
        groups.setdefault(digest, []).append(e)
    return [sorted(e["sample_id"] for e in g) for g in groups.values() if len({e["label"] for e in g}) > 1]


def confident_disagreements(entries: list[dict], scores: dict[int, float], margin: float = 0.05) -> list[int]:
    """Labelled-safe functions scored above 1 - margin and labelled-vulnerable
    functions scored below margin: candidates for manual label review."""
    return sorted(e["sample_id"] for e in entries if e["sample_id"] in scores and (
        (e["label"] == 0 and scores[e["sample_id"]] > 1 - margin) or
        (e["label"] == 1 and scores[e["sample_id"]] < margin)))


# ---------------------------------------------------------------- command line


def _manifest(data_dir: Path, split: str) -> list[dict]:
    return json.loads((data_dir / split / "manifest.json").read_text())


def _labelled(entries: list[dict], scores: dict[int, float]) -> tuple[list[int], list[int], list[float]]:
    ids = [e["sample_id"] for e in entries if e["sample_id"] in scores]
    label = {e["sample_id"]: e["label"] for e in entries}
    return ids, [label[i] for i in ids], [scores[i] for i in ids]


def _val_threshold(run_dir: Path) -> float:
    results = run_dir / "evaluation" / "results.json"
    if results.exists():
        return json.loads(results.read_text())["tools"]["aegis"]["threshold"]
    return 0.5


def study_seeds(data_dir: Path, run_dirs: list[Path], split: str = "test") -> dict:
    """PR-AUC per seed with bootstrap intervals, the across-seed summary, and
    a paired test against each baseline whose scores were cached by aegis-evaluate."""
    entries = _manifest(data_dir, split)
    per_run, pvalues = [], {}
    for run in run_dirs:
        scores = gnn_scores(run / "best.pt", data_dir / split / "graphs")
        ids, y, s = _labelled(entries, scores)
        per_run.append({"run": str(run), **bootstrap_ci(y, s)})
        for cache in sorted((run / "evaluation").glob("*_scores.json")):
            tool = cache.name.removesuffix("_scores.json")
            base = {int(k): v for k, v in json.loads(cache.read_text()).items()}
            common = [i for i in ids if i in base]
            label = dict(zip(ids, y))
            p = paired_bootstrap_p([label[i] for i in common], [scores[i] for i in common],
                                   [base[i] for i in common])
            pvalues.setdefault(tool, []).append(p)
    return {"runs": per_run, "summary": summarize_seeds([r["value"] for r in per_run if r["value"] is not None]),
            "p_aegis_beats": {tool: max(ps) for tool, ps in pvalues.items()}}


def study_cross(checkpoint: Path, data_dir: Path, threshold: float = 0.5, split: str = "test") -> dict:
    scores = gnn_scores(checkpoint, data_dir / split / "graphs")
    _, y, s = _labelled(_manifest(data_dir, split), scores)
    return {"target": str(data_dir), **binary_metrics(y, s, threshold), "pr_auc_ci": bootstrap_ci(y, s),
            "positive_rate": sum(y) / len(y) if y else None}


def study_calibration(data_dir: Path, run_dir: Path, target_accuracy: float = 0.9) -> dict:
    """Fits a temperature on validation, measures calibration on test, picks an
    abstention margin on validation, and saves both next to the model as
    best.pt.calibration.json (the web app reads it)."""
    checkpoint = run_dir / "best.pt"
    _, y_val, p_val = _labelled(_manifest(data_dir, "val"), gnn_scores(checkpoint, data_dir / "val" / "graphs"))
    _, y_test, p_test = _labelled(_manifest(data_dir, "test"), gnn_scores(checkpoint, data_dir / "test" / "graphs"))
    temperature = fit_temperature(y_val, p_val)
    threshold = _val_threshold(run_dir)
    cal_val, cal_test = apply_temperature(p_val, temperature), apply_temperature(p_test, temperature)
    threshold_cal = apply_temperature([threshold], temperature)[0]
    margin = choose_margin(abstention_curve(y_val, cal_val, threshold_cal), target_accuracy)
    result = {"temperature": temperature, "threshold": threshold_cal, "abstain_margin": margin,
              "ece_test_before": expected_calibration_error(y_test, p_test),
              "ece_test_after": expected_calibration_error(y_test, cal_test),
              "abstention_test": abstention_curve(y_test, cal_test, threshold_cal)}
    (run_dir / "best.pt.calibration.json").write_text(json.dumps(
        {k: result[k] for k in ("temperature", "threshold", "abstain_margin")}, indent=2))
    return result


def study_explanations(data_dir: Path, run_dir: Path, k: int = 3, split: str = "test") -> dict:
    """Top-k hit rate of line explanations against the fixed lines recorded in
    the manifest (BigVul vulnerable functions), versus random line picks."""
    from aegis.scanning.scanner import explain_lines

    model = load_trained_model(run_dir / "best.pt", "cpu")
    hits, chances = [], []
    for e in _manifest(data_dir, split):
        path = data_dir / split / "graphs" / f"{e['sample_id']}.pt"
        if e["label"] != 1 or not e.get("fix_lines") or not path.exists():
            continue
        graph = torch.load(path, weights_only=False)["graph"]
        if not hasattr(graph[NODE], "line"):
            continue
        graph[NODE].x = graph[NODE].x.float()
        outcome = explanation_hit(explain_lines(model, graph), e["fix_lines"], k)
        hits.append(outcome["hit"])
        chances.append(outcome["chance"])
    n = len(hits)
    return {"k": k, "n": n, "hit_rate": sum(hits) / n if n else None,
            "random_hit_rate": statistics.fmean(chances) if n else None}


def study_noise(data_dir: Path, run_dir: Path | None = None, split: str = "train") -> dict:
    """Writes <data-dir>/<split>/noisy_ids.json, which `aegis-train --exclude-ids` accepts."""
    entries = _manifest(data_dir, split)
    sources = {e["sample_id"]: (data_dir / e["source_path"]).read_text(encoding="utf-8", errors="replace")
               for e in entries}
    conflicts = conflicting_duplicates(entries, sources)
    disagreements = []
    if run_dir is not None:
        disagreements = confident_disagreements(entries, gnn_scores(run_dir / "best.pt",
                                                                    data_dir / split / "graphs"))
    flagged = sorted({i for g in conflicts for i in g} | set(disagreements))
    (data_dir / split / "noisy_ids.json").write_text(json.dumps(flagged))
    return {"functions": len(entries), "conflicting_duplicate_groups": len(conflicts),
            "functions_in_conflicts": sum(len(g) for g in conflicts),
            "confident_disagreements": len(disagreements), "flagged": len(flagged)}


def main() -> None:
    """Examples:

        aegis-study seeds --data-dir data/bigvul --runs runs/s0 runs/s1 runs/s2
        aegis-study cross --checkpoint runs/juliet-cpp/best.pt --data-dir data/bigvul
        aegis-study cwe --data-dir data/juliet-cpp --run-dir runs/juliet-cpp
        aegis-study calibrate --data-dir data/bigvul --run-dir runs/bigvul
        aegis-study explain --data-dir data/bigvul --run-dir runs/bigvul
        aegis-study noise --data-dir data/bigvul --run-dir runs/bigvul
        aegis-study robustness --data-dir data/bigvul --run-dir runs/bigvul --limit 300
        aegis-study augment --data-dir data/bigvul        # then aegis-train --extra-train-dir ...
        aegis-study poison --data-dir data/bigvul --rate 0.05
        aegis-study detect-poison --data-dir data/bigvul --run-dir runs/bigvul_poisoned
    """
    parser = argparse.ArgumentParser(description="Uncertainty, generalisation, per-CWE, calibration, "
                                                 "explanation and label-noise studies.")
    sub = parser.add_subparsers(dest="study", required=True)
    s = sub.add_parser("seeds")
    s.add_argument("--data-dir", type=Path, required=True)
    s.add_argument("--runs", type=Path, nargs="+", required=True)
    c = sub.add_parser("cross")
    c.add_argument("--checkpoint", type=Path, required=True)
    c.add_argument("--data-dir", type=Path, required=True, help="The other dataset")
    c.add_argument("--threshold", type=float, default=0.5)
    for name in ("cwe", "calibrate", "explain", "noise"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", type=Path, required=True)
        p.add_argument("--run-dir", type=Path, required=name != "noise")
    sub.choices["calibrate"].add_argument("--target-accuracy", type=float, default=0.9)
    r = sub.add_parser("robustness", help="Do meaning-preserving code changes flip the verdict?")
    r.add_argument("--data-dir", type=Path, required=True)
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--transforms", nargs="+", default=["rename", "deadcode", "rename+deadcode"])
    r.add_argument("--limit", type=int, default=None, help="Score a random subset of this many test functions")
    a = sub.add_parser("augment", help="Transformed copies of the training split for adversarial training")
    a.add_argument("--data-dir", type=Path, required=True)
    a.add_argument("--transforms", nargs="+", default=["rename", "deadcode"])
    po = sub.add_parser("poison", help="Plant a backdoor in a copy of the training data (for testing defences)")
    po.add_argument("--data-dir", type=Path, required=True)
    po.add_argument("--rate", type=float, default=0.05)
    d = sub.add_parser("detect-poison", help="Spectral-signature scan of the training data")
    d.add_argument("--data-dir", type=Path, required=True)
    d.add_argument("--run-dir", type=Path, required=True, help="A model trained on the (possibly poisoned) data")
    d.add_argument("--flag-fraction", type=float, default=0.075)
    sub.choices["explain"].add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    if args.study == "seeds":
        result = study_seeds(args.data_dir, args.runs)
    elif args.study == "cross":
        result = study_cross(args.checkpoint, args.data_dir, args.threshold)
    elif args.study == "cwe":
        entries = _manifest(args.data_dir, "test")
        scores = gnn_scores(args.run_dir / "best.pt", args.data_dir / "test" / "graphs")
        result = per_cwe_metrics(entries, scores, _val_threshold(args.run_dir))
    elif args.study == "calibrate":
        result = study_calibration(args.data_dir, args.run_dir, args.target_accuracy)
    elif args.study == "explain":
        result = study_explanations(args.data_dir, args.run_dir, args.k)
    elif args.study == "noise":
        result = study_noise(args.data_dir, args.run_dir)
    elif args.study == "robustness":
        from aegis.evaluation.robustness import robustness_study
        result = robustness_study(args.data_dir, args.run_dir, args.transforms, limit=args.limit,
                                  threshold=_val_threshold(args.run_dir))
    elif args.study == "augment":
        from aegis.evaluation.robustness import augment_split
        result = augment_split(args.data_dir, args.transforms)
    elif args.study == "poison":
        from aegis.evaluation.robustness import poison_split
        result = poison_split(args.data_dir, args.rate)
    else:
        from aegis.evaluation.robustness import detect_poison, spectral_signature_scores
        poisoned_dir = args.data_dir / "train_poisoned"
        poisoned_file = poisoned_dir / "poisoned_ids.json"
        poisoned = set(json.loads(poisoned_file.read_text())) if poisoned_file.exists() else set()
        scores = spectral_signature_scores(args.run_dir / "best.pt",
                                           [args.data_dir / "train" / "graphs", poisoned_dir / "graphs"])
        result = detect_poison(scores, poisoned, args.flag_fraction)
    out_dir = (getattr(args, "run_dir", None) or args.data_dir) / "studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.study + ("_" + "-".join(args.transforms) if args.study == "robustness" else "")
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
