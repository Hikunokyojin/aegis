"""Everything the Aegis web UI does, except drawing it: turning pasted code,
uploads and GitHub links into source files, scoring every function, explaining
each score line by line, running Semgrep/CodeQL on the same code, and drawing
a function's code graph. Kept free of Streamlit so it can be tested directly.

Inputs come from users and the internet, so archives are extracted
defensively: only C/C++/Java source files are written, every path must stay
inside the work folder (no `../` escapes), links are skipped, and both the
download and the unpacked size are capped.
"""
import json
import re
import shutil
import stat
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import torch

from aegis.embeddings.codebert_embedder import default_device, load_model
from aegis.evaluation.baselines import CODEQL_SUITES, SEMGREP_CONFIGS, codeql_commands, semgrep_commands
from aegis.evaluation.compare import load_trained_model
from aegis.model.hetero_gnn import NODE
from aegis.parsing.function_extractor import extract_functions, to_standalone
from aegis.pipeline.build_training_graphs import parse_function_files

SOURCE_LANGUAGES = {".c": "cpp", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".h": "cpp", ".hpp": "cpp",
                    ".java": "java"}
STAGED_EXTENSION = {"cpp": ".cpp", "java": ".java"}
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_FILES = 5000
EDGE_COLOURS = {"AST": "#9e9e9e", "CFG": "#1f77b4", "REACHING_DEF": "#d62728"}


class UnsafeArchiveError(ValueError):
    """An uploaded or downloaded archive tried to escape the work folder or was too large."""


# ---------------------------------------------------------------- inputs

def detect_language(code: str) -> str:
    """"java" or "cpp" (C and C++), from tell-tale tokens; C/C++ when unsure."""
    java = sum(bool(re.search(p, code)) for p in (
        r"\bimport\s+java", r"\bpackage\s+[\w.]+;", r"\bSystem\.out\.", r"\bpublic\s+(static\s+)?(class|void)",
        r"\bString\[\]", r"\bnew\s+\w+\(.*\)\s*;", r"@Override", r"\bthrows\b"))
    cpp = sum(bool(re.search(p, code)) for p in (
        r"#\s*include", r"->", r"::", r"\bprintf\s*\(", r"\b(malloc|free|strcpy|memcpy|sizeof)\s*\(",
        r"\bchar\s*\*", r"\bstd::", r"\bunsigned\b"))
    return "java" if java > cpp else "cpp"


def write_pasted_code(code: str, language: str, folder: Path) -> Path:
    """Saves pasted code as a source file. A bare Java method (no class
    around it) is wrapped in one, since Java code must live in a class."""
    folder.mkdir(parents=True, exist_ok=True)
    if language == "java":
        if not re.search(r"\b(class|interface|enum|record)\s+\w+", code):
            code = to_standalone(code, "java")
        path = folder / "Snippet.java"
    else:
        path = folder / "snippet.cpp"
    path.write_text(code, encoding="utf-8")
    return path


def safe_extract_zip(zip_path: Path, dest: Path, max_total_bytes: int = MAX_ARCHIVE_BYTES,
                     max_files: int = MAX_ARCHIVE_FILES) -> list[Path]:
    """Extracts only C/C++/Java source files from zip_path into dest and
    returns their paths. Raises UnsafeArchiveError for any entry that would
    land outside dest, or when the archive is too big or has too many files."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    written, total = [], 0
    with zipfile.ZipFile(zip_path) as zf:
        entries = zf.infolist()
        if len(entries) > max_files:
            raise UnsafeArchiveError(f"Archive has {len(entries)} entries (limit {max_files}).")
        for info in entries:
            name = info.filename.replace("\\", "/")
            parts = Path(name).parts
            if name.startswith("/") or ".." in parts or (parts and ":" in parts[0]):
                raise UnsafeArchiveError(f"Archive entry escapes the extraction folder: {info.filename}")
            target = (dest / name).resolve()
            if dest not in target.parents:
                raise UnsafeArchiveError(f"Archive entry escapes the extraction folder: {info.filename}")
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                continue
            if Path(name).suffix.lower() not in SOURCE_LANGUAGES:
                continue
            total += info.file_size
            if total > max_total_bytes:
                raise UnsafeArchiveError(f"Source files in the archive exceed {max_total_bytes // (1024 * 1024)} MB.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                out.write(src.read(max_total_bytes + 1))
            written.append(target)
    return written


def parse_github_url(url: str) -> tuple[str, str]:
    """('owner', 'repo') from https://github.com/owner/repo[.git]; anything else is rejected."""
    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or parsed.netloc.lower() not in ("github.com", "www.github.com"):
        raise ValueError("Use a link of the form https://github.com/<owner>/<repository>.")
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("Use a link of the form https://github.com/<owner>/<repository>.")
    owner, repo = parts[0], parts[1].removesuffix(".git")
    if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", x) for x in (owner, repo)):
        raise ValueError("That does not look like a GitHub owner/repository name.")
    return owner, repo


def download_github_repo(url: str, dest: Path, max_bytes: int = MAX_ARCHIVE_BYTES,
                         opener=urllib.request.urlopen) -> list[Path]:
    """Downloads a public repository's default branch as a zip (size-capped)
    and extracts its C/C++/Java files into dest."""
    owner, repo = parse_github_url(url)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "_download.zip"
    received = 0
    with opener(f"https://codeload.github.com/{owner}/{repo}/zip/HEAD", timeout=60) as response, \
            open(archive, "wb") as out:
        while chunk := response.read(1024 * 1024):
            received += len(chunk)
            if received > max_bytes:
                raise UnsafeArchiveError(f"Repository download exceeds {max_bytes // (1024 * 1024)} MB.")
            out.write(chunk)
    try:
        return safe_extract_zip(archive, dest / "repo")
    finally:
        archive.unlink(missing_ok=True)


# ---------------------------------------------------------------- scoring and explanation

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
            runner=subprocess.run, explain: bool = True) -> dict:
    """Scores every function in `files` with the model for its language.

    Returns {"functions": [...], "unparsed": [...]}. Each function has file,
    function, language, start_line, end_line, source, probability,
    line_scores ({line: 0..1}) and graph (for the graph view), and is sorted
    most-likely-vulnerable first. Functions that could not be turned into a
    graph, or whose language has no model, are listed in "unparsed" with a reason."""
    device = device or default_device()
    shutil.rmtree(work_dir, ignore_errors=True)

    staged: dict[str, dict[str, dict]] = {"cpp": {}, "java": {}}
    unparsed = []
    for file in files:
        language = SOURCE_LANGUAGES.get(file.suffix.lower())
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


# ---------------------------------------------------------------- Semgrep / CodeQL on the same code

def _stage_for_tool(files: list[Path], scan_dir: Path, cpp_extension: str | None) -> dict[str, Path]:
    shutil.rmtree(scan_dir, ignore_errors=True)
    scan_dir.mkdir(parents=True)
    staged = {}
    for i, file in enumerate(files):
        language = SOURCE_LANGUAGES.get(file.suffix.lower())
        suffix = cpp_extension if (language == "cpp" and cpp_extension) else file.suffix
        name = f"{i}_{file.stem}{suffix}"
        shutil.copy(file, scan_dir / name)
        staged[name] = file
    return staged


def run_semgrep(files: list[Path], work_dir: Path, runner=subprocess.run) -> list[dict]:
    """Semgrep findings for `files` as [{tool, file, line, rule, message}].
    C/C++ is scanned as .c because Semgrep OSS skips .cpp files."""
    findings = []
    for language in ("cpp", "java"):
        group = [f for f in files if SOURCE_LANGUAGES.get(f.suffix.lower()) == language]
        if not group:
            continue
        scan_dir = work_dir / f"semgrep_{language}"
        staged = _stage_for_tool(group, scan_dir, ".c")
        output = work_dir / f"semgrep_{language}.json"
        for cmd in semgrep_commands(scan_dir, output, SEMGREP_CONFIGS[language]):
            runner(cmd, check=True)
        for r in json.loads(output.read_text())["results"]:
            original = staged.get(Path(r["path"]).name)
            if original is not None:
                findings.append({"tool": "semgrep", "file": str(original), "line": r["start"]["line"],
                                 "rule": r["check_id"].split(".")[-1],
                                 "message": r.get("extra", {}).get("message", "")})
    return findings


def run_codeql(files: list[Path], work_dir: Path, runner=subprocess.run) -> list[dict]:
    """CodeQL (security-extended queries, no build) findings for `files`."""
    findings = []
    for language in ("cpp", "java"):
        group = [f for f in files if SOURCE_LANGUAGES.get(f.suffix.lower()) == language]
        if not group:
            continue
        scan_dir = work_dir / f"codeql_{language}"
        staged = _stage_for_tool(group, scan_dir, None)
        sarif = work_dir / f"codeql_{language}.sarif"
        db_dir = work_dir / f"codeql_db_{language}"
        for cmd in codeql_commands(db_dir, scan_dir, sarif, language, CODEQL_SUITES[language]):
            runner(cmd, check=True)
        for run_ in json.loads(sarif.read_text())["runs"]:
            for r in run_.get("results", []):
                loc = r["locations"][0]["physicalLocation"]
                original = staged.get(Path(loc["artifactLocation"]["uri"]).name)
                if original is not None:
                    findings.append({"tool": "codeql", "file": str(original),
                                     "line": loc.get("region", {}).get("startLine", 0),
                                     "rule": r.get("ruleId", ""), "message": r.get("message", {}).get("text", "")})
    return findings


def attach_findings(functions: list[dict], findings: list[dict]) -> list[dict]:
    """Adds a "findings" list to each function: the tool findings in the same
    file whose line falls inside the function. Returns findings that matched
    no function (e.g. at file level)."""
    for fn in functions:
        fn["findings"] = []
    unmatched = []
    for finding in findings:
        owner = next((fn for fn in functions if fn["file"] == finding["file"]
                      and fn["start_line"] <= finding["line"] <= fn["end_line"]), None)
        (owner["findings"] if owner else unmatched).append(finding)
    return unmatched


# ---------------------------------------------------------------- graph view

def graph_to_dot(graph, line_scores: dict[int, float] | None = None, max_nodes: int = 120) -> str:
    """Graphviz DOT for one function's code graph: syntax edges grey, control
    flow blue, data flow red; nodes shaded by how much their line contributed
    to the score. Large graphs are cut to the first max_nodes nodes."""
    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', "'").replace("\n", " ")[:40]

    line_scores = line_scores or {}
    lines = graph[NODE].line.tolist() if hasattr(graph[NODE], "line") else [-1] * graph[NODE].x.shape[0]
    n = min(graph[NODE].x.shape[0], max_nodes)
    out = ['digraph G {', '  graph [rankdir=TB, fontname="Helvetica"];',
           '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];']
    for i in range(n):
        heat = line_scores.get(lines[i], 0.0)
        shade = int(255 - 180 * heat)
        label = f"{graph[NODE].joern_type[i]}\\n{esc(graph[NODE].source_text[i])}"
        out.append(f'  n{i} [label="{label}", fillcolor="#ff{shade:02x}{shade:02x}"];')
    for relation, colour in EDGE_COLOURS.items():
        for s, t in graph[NODE, relation, NODE].edge_index.t().tolist():
            if s < n and t < n:
                out.append(f'  n{s} -> n{t} [color="{colour}", tooltip="{relation}"];')
    out.append("}")
    return "\n".join(out)
