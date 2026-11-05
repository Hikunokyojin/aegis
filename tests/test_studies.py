"""User journeys covered:

- As a researcher, I want confidence intervals and a significance test, so a
  claimed win over a baseline is not a lucky split.
- As a researcher, I want results per vulnerability type, a zero-day style
  time split, calibrated probabilities with an "uncertain" answer, a check
  that explanations hit the real fixed lines, and an audit of noisy labels.
"""
import math

import pytest
import torch

from aegis.dataset.bigvul_loader import FunctionRecord
from aegis.evaluation.studies import (
    abstention_curve,
    apply_temperature,
    bootstrap_ci,
    choose_margin,
    confident_disagreements,
    conflicting_duplicates,
    expected_calibration_error,
    explanation_hit,
    fit_temperature,
    fix_lines,
    paired_bootstrap_p,
    per_cwe_metrics,
    summarize_seeds,
)
from aegis.pipeline.prepare_graph_inputs import time_split
from aegis.training.trainer import GraphFolder

LABELS = [1, 0, 1, 0, 1, 0, 0, 0, 1, 0] * 4
GOOD = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.1, 0.2, 0.95, 0.05] * 4
RANDOM = [0.5, 0.6, 0.4, 0.5, 0.3, 0.7, 0.6, 0.4, 0.5, 0.5] * 4


def test_bootstrap_interval_contains_the_point_estimate():
    ci = bootstrap_ci(LABELS, GOOD, n_resamples=200)
    assert ci["low"] <= ci["value"] <= ci["high"]


def test_paired_test_separates_a_real_win_from_no_difference():
    assert paired_bootstrap_p(LABELS, GOOD, RANDOM, n_resamples=200) < 0.05
    assert paired_bootstrap_p(LABELS, GOOD, GOOD, n_resamples=200) > 0.5


def test_seed_summary_reports_mean_and_an_interval():
    s = summarize_seeds([0.40, 0.42, 0.44])
    assert s["mean"] == pytest.approx(0.42) and s["low"] < 0.42 < s["high"] and s["n"] == 3


def test_per_cwe_scores_each_type_against_all_safe_functions_and_pools_rare_ones():
    entries = ([{"sample_id": i, "label": 1, "cwe_id": "CWE-787"} for i in range(5)]
               + [{"sample_id": 5, "label": 1, "cwe_id": "CWE-416"}]
               + [{"sample_id": 10 + i, "label": 0, "cwe_id": ""} for i in range(5)])
    scores = {e["sample_id"]: (0.9 if e["label"] else 0.1) for e in entries}
    results = per_cwe_metrics(entries, scores, threshold=0.5, min_positives=5)
    assert set(results) == {"CWE-787", "other"}
    assert results["CWE-787"]["n"] == 10 and results["CWE-787"]["pr_auc"] == 1.0


def test_temperature_scaling_reduces_calibration_error_of_an_overconfident_model():
    labels = [1, 0] * 50 + [1] * 20
    overconfident = [0.99 if i % 3 else 0.01 for i in range(len(labels))]
    temp = fit_temperature(labels, overconfident)
    assert temp > 1
    assert (expected_calibration_error(labels, apply_temperature(overconfident, temp))
            < expected_calibration_error(labels, overconfident))


def test_abstaining_near_the_threshold_trades_coverage_for_accuracy():
    curve = abstention_curve(LABELS, RANDOM, threshold=0.5)
    assert curve[0]["coverage"] == 1.0 and curve[-1]["coverage"] < 1.0
    assert choose_margin(curve, target_accuracy=1.01) == curve[-1]["margin"]


def test_fix_lines_marks_changed_and_deleted_lines_of_the_vulnerable_version():
    before = "int f(char *s) {\n  char b[4];\n  strcpy(b, s);\n  return 0;\n}"
    after = "int f(char *s) {\n  char b[4];\n  strncpy(b, s, 3);\n  return 0;\n}"
    assert fix_lines(before, after) == [3]
    inserted = "int f(char *s) {\n  char b[4];\n  if (!s) return 1;\n  strcpy(b, s);\n  return 0;\n}"
    assert fix_lines(before, inserted) == [2]


def test_explanation_hit_compares_top_lines_with_the_fix_and_chance():
    scores = {1: 0.1, 2: 0.2, 3: 1.0, 4: 0.3, 5: 0.0}
    hit = explanation_hit(scores, [3], k=1)
    assert hit["hit"] and math.isclose(hit["chance"], 1 / 5)
    assert not explanation_hit(scores, [5], k=1)["hit"]


def test_noise_audit_finds_identical_code_with_conflicting_labels_and_confident_disagreements():
    entries = [{"sample_id": 1, "label": 1}, {"sample_id": 2, "label": 0}, {"sample_id": 3, "label": 0}]
    sources = {1: "int f() { return 1; }", 2: "int f()  {return 1;} // same", 3: "int g() { return 2; }"}
    assert conflicting_duplicates(entries, sources) == [[1, 2]]
    assert confident_disagreements(entries, {1: 0.01, 2: 0.5, 3: 0.99}) == [1, 3]


def _record(i, cve):
    return FunctionRecord(sample_id=i, function_id=f"p{i % 3}", label=i % 2, language="C", cve_id=cve,
                          cwe_id="", commit_id=f"c{i}", source_before="", source_after="")


def test_time_split_tests_only_on_vulnerabilities_newer_than_all_training_ones():
    functions = [_record(i, f"CVE-{2010 + i // 10}-{i:04d}") for i in range(100)]
    train, val, test = time_split(functions)

    def years(fns):
        return {int(f.cve_id[4:8]) for f in fns}
    assert max(years(train)) < min(years(val)) <= max(years(val)) < min(years(test))
    assert len(train) + len(val) + len(test) == 100


def test_time_split_refuses_data_without_cve_years():
    with pytest.raises(ValueError):
        time_split([_record(0, "")])


def test_graph_folder_loads_lazily_restores_full_precision_and_skips_excluded_ids(tmp_path):
    from torch_geometric.data import HeteroData
    for i in range(3):
        g = HeteroData()
        g["NODE"].x = torch.ones(2, 4).half()
        torch.save({"graph": g, "label": i % 2, "sample_id": i}, tmp_path / f"{i}.pt")
    folder = GraphFolder(tmp_path, exclude={1})
    assert len(folder) == 2
    assert folder[1].sample_id == 2 and folder[1]["NODE"].x.dtype == torch.float32


def _fake_dataset(root, n=12):
    """A tiny data dir (manifests + graphs) and a run dir with a trained-shape checkpoint."""
    import json

    from test_ui_analysis import _graph

    from aegis.model.hetero_gnn import AegisHeteroGNN
    for split in ("val", "test", "train"):
        entries = []
        (root / "data" / split / "graphs").mkdir(parents=True)
        for i in range(n):
            label = i % 2
            src = root / "data" / split / f"{i}.cpp"
            src.write_text(f"int f{i % 4}() {{ return {label}; }}")
            entries.append({"sample_id": i, "label": label, "cwe_id": "CWE-787" if label else "",
                            "source_path": f"{split}/{i}.cpp", "fix_lines": [2] if label else []})
            torch.save({"graph": _graph([1, 2, 3]), "label": label, "sample_id": i},
                       root / "data" / split / "graphs" / f"{i}.pt")
        (root / "data" / split / "manifest.json").write_text(json.dumps(entries))
    (root / "run").mkdir()
    config = {"in_dim": 8, "hidden_dim": 16, "num_layers": 1}
    torch.save({"model": AegisHeteroGNN(**config).state_dict(), "config": config}, root / "run" / "best.pt")
    return root / "data", root / "run"


def test_study_commands_run_end_to_end_and_calibration_reaches_the_scanner(tmp_path):
    from aegis.evaluation.studies import study_calibration, study_cross, study_explanations, study_noise, study_seeds
    from aegis.scanning.scanner import load_calibration

    torch.manual_seed(0)
    data, run = _fake_dataset(tmp_path)
    assert study_seeds(data, [run])["summary"]["n"] == 1
    assert study_cross(run / "best.pt", data)["n"] == 12
    calibration = study_calibration(data, run)
    assert set(load_calibration(run / "best.pt")) == {"temperature", "threshold", "abstain_margin"}
    # Random untrained data: only check the outputs are well formed (improvement is tested above).
    assert 0 <= calibration["ece_test_after"] <= 1 and calibration["temperature"] > 0
    explained = study_explanations(data, run, k=1)
    assert explained["n"] == 6 and 0 <= explained["hit_rate"] <= 1
    noise = study_noise(data)
    assert noise["conflicting_duplicate_groups"] == 0 and (data / "train" / "noisy_ids.json").exists()
