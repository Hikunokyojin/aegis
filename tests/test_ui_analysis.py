"""User journeys covered (the Aegis web UI's logic, without the UI itself):

- As a user, I want to paste code without saying which language it is, and
  have C/C++ or Java detected.
- As a user, I want to upload a .zip or give a GitHub link, and have only
  the source files inside used, safely (no files written outside the work
  folder, no unbounded downloads).
- As a user, I want every function scored, with the lines that pushed the
  score up highlighted, and the Semgrep/CodeQL findings for the same code
  shown against the function they fall in.
- As a user, I want to see a function's code graph with its three kinds of
  edges.
"""
import io
import zipfile

import pytest
import torch
from torch_geometric.data import HeteroData

from aegis.model.hetero_gnn import AegisHeteroGNN
from aegis.ui.analysis import (
    UnsafeArchiveError,
    analyze,
    attach_findings,
    detect_language,
    explain_lines,
    graph_to_dot,
    parse_github_url,
    safe_extract_zip,
)


def test_detect_language():
    assert detect_language('#include <stdio.h>\nint main() { printf("x"); }') == "cpp"
    assert detect_language("void f(char *s) { char b[4]; strcpy(b, s); }") == "cpp"
    assert detect_language("public class A { void f() { System.out.println(1); } }") == "java"
    assert detect_language("import java.sql.*;\nclass Q { }") == "java"


def _zip(entries: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in entries.items():
            zf.writestr(name, text)
    return buf.getvalue()


def test_safe_extract_zip_keeps_only_source_files(tmp_path):
    archive = tmp_path / "p.zip"
    archive.write_bytes(_zip({"proj/a.c": "int f(){return 0;}", "proj/b/B.java": "class B{}",
                              "proj/readme.md": "x", "proj/run.sh": "rm -rf /"}))
    files = safe_extract_zip(archive, tmp_path / "out")
    assert sorted(p.name for p in files) == ["B.java", "a.c"]
    assert all((tmp_path / "out") in p.parents for p in files)


def test_safe_extract_zip_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    archive.write_bytes(_zip({"../../escape.c": "int x;"}))
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive, tmp_path / "out")
    assert not (tmp_path / "escape.c").exists()


def test_safe_extract_zip_enforces_a_size_limit(tmp_path):
    archive = tmp_path / "big.zip"
    archive.write_bytes(_zip({"a.c": "x" * 5000}))
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive, tmp_path / "out", max_total_bytes=1000)


def test_parse_github_url():
    assert parse_github_url("https://github.com/joernio/joern") == ("joernio", "joern")
    assert parse_github_url("https://github.com/a-b/c.d.git/") == ("a-b", "c.d")
    for bad in ("http://github.com/a/b", "https://gitlab.com/a/b", "https://github.com/a", "file:///etc/passwd"):
        with pytest.raises(ValueError):
            parse_github_url(bad)


def _graph(lines: list[int]) -> HeteroData:
    g = HeteroData()
    g["NODE"].x = torch.randn(len(lines), 8)
    g["NODE"].line = torch.tensor(lines)
    g["NODE"].joern_type = ["METHOD"] + ["CALL"] * (len(lines) - 1)
    g["NODE"].source_text = [f"n{i}" for i in range(len(lines))]
    chain = torch.tensor([list(range(len(lines) - 1)), list(range(1, len(lines)))])
    for rel in ("AST", "CFG", "REACHING_DEF"):
        g["NODE", rel, "NODE"].edge_index = chain
    return g


def test_explain_lines_scores_each_source_line_between_0_and_1():
    model = AegisHeteroGNN(in_dim=8, hidden_dim=16, num_layers=1).eval()
    scores = explain_lines(model, _graph([1, 2, 2, 3, -1]), line_offset=10)
    assert set(scores) == {11, 12, 13}  # line -1 (unknown) dropped, offset applied
    assert max(scores.values()) == pytest.approx(1.0) and min(scores.values()) >= 0.0


def test_attach_findings_assigns_by_file_and_line_range():
    functions = [{"file": "a.c", "start_line": 1, "end_line": 5}, {"file": "a.c", "start_line": 7, "end_line": 9}]
    findings = [{"tool": "semgrep", "file": "a.c", "line": 3, "rule": "gets"},
                {"tool": "codeql", "file": "a.c", "line": 8, "rule": "overflow"},
                {"tool": "semgrep", "file": "b.c", "line": 3, "rule": "x"}]
    attach_findings(functions, findings)
    assert [f["rule"] for f in functions[0]["findings"]] == ["gets"]
    assert [f["rule"] for f in functions[1]["findings"]] == ["overflow"]


def test_graph_to_dot_colours_each_edge_type():
    dot = graph_to_dot(_graph([1, 2, 3]), {1: 1.0, 2: 0.2})
    assert dot.startswith("digraph")
    assert dot.count("->") == 6 and "AST" in dot and "CFG" in dot and "REACHING_DEF" in dot


def test_analyze_scores_functions_with_explanations(tmp_path):
    from test_build_training_graphs import _fake_joern_parsing_only

    code = tmp_path / "util.c"
    code.write_text("int add(int a, int b) { return a + b; }\n\nvoid copy(char *s) { char b[4]; strcpy(b, s); }\n")
    config = {"in_dim": 811, "hidden_dim": 16, "num_layers": 1}
    ckpt = tmp_path / "m.pt"
    torch.save({"model": AegisHeteroGNN(**config).state_dict(), "config": config}, ckpt)

    result = analyze([code], {"cpp": ckpt}, tmp_path / "work", device="cpu",
                     runner=_fake_joern_parsing_only({"0.cpp"}, []))

    [scored] = result["functions"]
    assert scored["function"] == "add" and scored["start_line"] == 1 and scored["end_line"] == 1
    assert 0.0 <= scored["probability"] <= 1.0 and "graph" in scored and "line_scores" in scored
    assert [u["function"] for u in result["unparsed"]] == ["copy"]
