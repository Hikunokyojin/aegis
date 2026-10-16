"""Runs Semgrep and CodeQL over a split's function files and turns their
findings into one score per function: the number of distinct findings the
tool reported in it (0 = the tool considers it clean). That score feeds the
same PR-AUC/F1 computation as the model's probabilities, and "score > 0" is
the tool's yes/no verdict.

Both tools see exactly the per-function files the model was built from (one
function per file, as written by aegis.pipeline.prepare_graph_inputs), so
all three are judged on identical inputs. That also means neither tool gets
the rest of the original program (headers, callers, build), which limits
their inter-procedural analysis; this is a property of the benchmark setup
and is reported alongside the results.

Tool behaviour this module relies on, checked against the real tools:
- Semgrep OSS (1.181) does not scan `.cpp` files with its C rules at all
  (they are left out of `paths.scanned`), so C/C++ functions are staged as
  `.c` for Semgrep. Rule packs overlap, so findings are de-duplicated by
  (rule, line).
- CodeQL is run with `--build-mode=none`, since the function files cannot be
  compiled; its bundle ships the query packs, so nothing is downloaded at
  analysis time.
"""
import json
import shutil
import subprocess
from pathlib import Path

SEMGREP_CONFIGS = {"cpp": ["p/c", "p/security-audit"], "java": ["p/java", "p/security-audit"]}
SEMGREP_EXTENSION = {"cpp": ".c", "java": None}
CODEQL_SUITES = {"cpp": "codeql/cpp-queries:codeql-suites/cpp-security-extended.qls",
                 "java": "codeql/java-queries:codeql-suites/java-security-extended.qls"}
TOOLS = ("semgrep", "codeql")


def stage_sources(entries: list[dict], data_dir: Path, scan_dir: Path, extension: str | None = None) -> dict[str, int]:
    """Copies each manifest entry's source file into a fresh scan_dir as
    {sample_id}{extension or original suffix}. Returns {staged filename: sample_id}."""
    shutil.rmtree(scan_dir, ignore_errors=True)
    scan_dir.mkdir(parents=True)
    staged = {}
    for entry in entries:
        source = data_dir / entry["source_path"]
        name = f"{entry['sample_id']}{extension or source.suffix}"
        shutil.copy(source, scan_dir / name)
        staged[name] = entry["sample_id"]
    return staged


def semgrep_commands(scan_dir: Path, output_json: Path, configs: list[str]) -> list[list[str]]:
    return [["semgrep", "scan", "--metrics=off", "--quiet", "--json", "--output", str(output_json),
             *[f"--config={c}" for c in configs], str(scan_dir)]]


def parse_semgrep_json(output_json: Path) -> dict[str, int]:
    """{file basename: number of distinct (rule, line) findings}."""
    distinct: dict[str, set] = {}
    for result in json.loads(output_json.read_text())["results"]:
        name = Path(result["path"]).name
        distinct.setdefault(name, set()).add((result["check_id"], result["start"]["line"]))
    return {name: len(findings) for name, findings in distinct.items()}


def codeql_commands(db_dir: Path, scan_dir: Path, sarif_path: Path, language: str,
                    suite: str | None = None) -> list[list[str]]:
    return [
        ["codeql", "database", "create", str(db_dir), f"--language={language}", "--build-mode=none",
         f"--source-root={scan_dir}", "--overwrite"],
        ["codeql", "database", "analyze", str(db_dir), suite or CODEQL_SUITES[language],
         "--format=sarif-latest", f"--output={sarif_path}", "--threads=0"],
    ]


def parse_sarif(sarif_path: Path) -> dict[str, int]:
    """{file basename: number of results located in it}."""
    counts: dict[str, int] = {}
    for run in json.loads(sarif_path.read_text())["runs"]:
        for result in run.get("results", []):
            uri = result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
            name = Path(uri).name
            counts[name] = counts.get(name, 0) + 1
    return counts


def run_baseline(tool: str, entries: list[dict], data_dir: Path, work_dir: Path, language: str,
                 runner=subprocess.run, configs: list[str] | None = None) -> dict[int, int]:
    """Runs `tool` ("semgrep" or "codeql") over the functions in `entries`
    and returns {sample_id: finding count} for every entry, 0 included."""
    tool_dir = work_dir / tool
    scan_dir = tool_dir / "scan"
    if tool == "semgrep":
        staged = stage_sources(entries, data_dir, scan_dir, SEMGREP_EXTENSION[language])
        output = tool_dir / "semgrep.json"
        commands = semgrep_commands(scan_dir, output, configs or SEMGREP_CONFIGS[language])
        parse = parse_semgrep_json
    elif tool == "codeql":
        staged = stage_sources(entries, data_dir, scan_dir)
        output = tool_dir / "results.sarif"
        commands = codeql_commands(tool_dir / "db", scan_dir, output, language)
        parse = parse_sarif
    else:
        raise ValueError(f"Unknown baseline {tool!r}; expected one of {TOOLS}")

    for cmd in commands:
        runner(cmd, check=True)
    counts = parse(output)
    return {sample_id: counts.get(name, 0) for name, sample_id in staged.items()}
