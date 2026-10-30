"""User journeys covered:

- As someone comparing Aegis with existing static analyzers, I want Semgrep
  and CodeQL run over exactly the same functions the model is scored on, and
  their findings turned into a per-function score, so all three tools are
  measured with the same PR-AUC/F1 on the same data.
- As a developer without either tool installed locally, I want the tool
  invocations injectable so the staging, command construction and output
  parsing are genuinely tested.
"""
import json
from pathlib import Path

from aegis.evaluation.baselines import (
    codeql_commands,
    parse_sarif,
    parse_semgrep_json,
    run_baseline,
    semgrep_commands,
    stage_sources,
)


def _entries(tmp_path: Path, ext: str) -> list[dict]:
    (tmp_path / "test").mkdir()
    entries = []
    for sid in (3, 7):
        (tmp_path / "test" / f"{sid}{ext}").write_text(f"void f{sid}() {{}}")
        entries.append({"sample_id": sid, "source_path": f"test/{sid}{ext}", "label": sid % 2})
    return entries


def test_stage_sources_can_change_the_extension(tmp_path):
    entries = _entries(tmp_path, ".cpp")

    staged = stage_sources(entries, tmp_path, tmp_path / "scan", extension=".c")

    assert staged == {"3.c": 3, "7.c": 7}
    assert (tmp_path / "scan" / "3.c").read_text() == "void f3() {}"


def test_semgrep_command_uses_every_config_and_json_output(tmp_path):
    [cmd] = semgrep_commands(tmp_path / "scan", tmp_path / "out.json", ["p/c", "p/security-audit"])
    assert cmd[:2] == ["semgrep", "scan"]
    assert "--config=p/c" in cmd and "--config=p/security-audit" in cmd
    assert "--json" in cmd and "--metrics=off" in cmd and cmd[-1] == str(tmp_path / "scan")


def test_parse_semgrep_json_counts_distinct_findings_per_file(tmp_path):
    out = tmp_path / "out.json"
    def finding(path, rule, line):
        return {"path": path, "check_id": rule, "start": {"line": line}}
    out.write_text(json.dumps({"results": [
        finding("scan/3.c", "gets", 1), finding("scan/3.c", "gets", 1),  # same rule from two packs
        finding("scan/3.c", "strcpy", 1), finding("7.c", "gets", 4)], "errors": []}))

    assert parse_semgrep_json(out) == {"3.c": 2, "7.c": 1}


def test_codeql_commands_create_without_a_build_then_analyze_to_sarif(tmp_path):
    create, analyze = codeql_commands(tmp_path / "db", tmp_path / "scan", tmp_path / "r.sarif", "java")
    assert create[:3] == ["codeql", "database", "create"]
    assert "--language=java" in create and "--build-mode=none" in create
    assert analyze[:3] == ["codeql", "database", "analyze"]
    assert any(arg.startswith("codeql/java-queries") for arg in analyze)
    assert "--format=sarif-latest" in analyze


def test_parse_sarif_counts_results_per_file(tmp_path):
    def loc(uri):
        return {"locations": [{"physicalLocation": {"artifactLocation": {"uri": uri}}}]}
    sarif = tmp_path / "r.sarif"
    sarif.write_text(json.dumps({"runs": [{"results": [loc("3.cpp"), loc("3.cpp"), loc("sub/7.cpp")]}]}))

    assert parse_sarif(sarif) == {"3.cpp": 2, "7.cpp": 1}


def test_run_baseline_gives_every_function_a_score_including_zero(tmp_path):
    entries = _entries(tmp_path, ".cpp")
    seen = []

    def fake_semgrep(cmd, check=True):
        seen.append(cmd)
        out = Path(cmd[cmd.index("--output") + 1])
        out.write_text(json.dumps({"results": [{"path": "3.c", "check_id": "gets", "start": {"line": 1}}]}))

    scores = run_baseline("semgrep", entries, tmp_path, tmp_path / "work", "cpp", runner=fake_semgrep)

    assert scores == {3: 1, 7: 0}
    assert (tmp_path / "work" / "semgrep" / "scan" / "3.c").exists()  # C/C++ scanned as .c


def test_run_baseline_codeql_keeps_cpp_extension(tmp_path):
    entries = _entries(tmp_path, ".cpp")

    def fake_codeql(cmd, check=True):
        if cmd[2] == "analyze":
            out = Path(next(a for a in cmd if a.startswith("--output=")).split("=", 1)[1])
            uri = {"locations": [{"physicalLocation": {"artifactLocation": {"uri": "7.cpp"}}}]}
            out.write_text(json.dumps({"runs": [{"results": [uri]}]}))

    assert run_baseline("codeql", entries, tmp_path, tmp_path / "work", "cpp", runner=fake_codeql) == {3: 0, 7: 1}
