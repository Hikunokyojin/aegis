"""User journeys covered:

- As a developer, I want to point Aegis at my own C/C++ or Java files and get
  a vulnerability probability for every function, ranked, with file and line,
  so I know which functions to review first.
- As a developer, I want files Aegis can't analyse reported, not silently
  ignored.
"""
import json
from pathlib import Path

import torch

from aegis.model.hetero_gnn import AegisHeteroGNN
from aegis.scanning.scanner import collect_source_files, scan
from test_build_training_graphs import _fake_joern_parsing_only


def _checkpoint(path: Path) -> Path:
    config = {"in_dim": 811, "hidden_dim": 16, "num_layers": 1}
    torch.save({"model": AegisHeteroGNN(**config).state_dict(), "config": config}, path)
    return path


def test_collect_source_files_finds_supported_languages_recursively(tmp_path):
    (tmp_path / "src" / "deep").mkdir(parents=True)
    for name in ("a.c", "src/b.cpp", "src/deep/C.java", "notes.txt", "src/x.py"):
        (tmp_path / name).write_text("")
    found = collect_source_files([tmp_path])
    assert sorted(p.name for p in found) == ["C.java", "a.c", "b.cpp"]


def test_scan_reports_every_function_with_location_and_ranks_by_probability(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "util.c").write_text("int add(int a, int b) { return a + b; }\n\nvoid copy(char *s) { char b[4]; strcpy(b, s); }\n")
    (tmp_path / "code" / "App.java").write_text("class App { void run() { int x = 1; } }")

    # Fake Joern parses the 1st C function and the Java method, but not the 2nd C function.
    runner = _fake_joern_parsing_only({"0.cpp", "0.java"}, [])
    report = scan([tmp_path / "code"], _checkpoint(tmp_path / "best.pt"), tmp_path / "work",
                  device="cpu", runner=runner)

    scored = report["functions"]
    assert {(f["function"], f["line"]) for f in scored} == {("add", 1), ("run", 1)}
    assert [f["probability"] for f in scored] == sorted((f["probability"] for f in scored), reverse=True)
    assert report["unparsed"] == [{"file": str(tmp_path / "code" / "util.c"), "function": "copy", "line": 3}]
    json.dumps(report)  # serialisable for --json output
