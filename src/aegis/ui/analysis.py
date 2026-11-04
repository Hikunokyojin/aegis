"""Everything the Aegis web UI does, except drawing it: turning pasted code,
uploads and GitHub links into source files, scoring every function, explaining
each score line by line, running Semgrep on the same code, and drawing
a function's code graph. Kept free of Streamlit so it can be tested directly.

Inputs come from users and the internet, so archives are extracted
defensively: only C/C++/Java source files are written, every path must stay
inside the work folder (no `../` escapes), links are skipped, and both the
download and the unpacked size are capped.
"""
import contextlib
import json
import re
import shutil
import stat
import subprocess
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from aegis.evaluation.baselines import SEMGREP_CONFIGS, semgrep_commands
from aegis.languages import SOURCE_LANGUAGES
from aegis.model.hetero_gnn import NODE
from aegis.parsing.function_extractor import to_standalone
from aegis.scanning.scanner import analyze, explain_lines  # noqa: F401  (re-exported for the web app)

MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_FILES = 5000
EDGE_COLOURS = {"AST": "#9e9e9e", "CFG": "#1f77b4", "REACHING_DEF": "#d62728"}


class ScanGate:
    """Protects a shared server: scans run one at a time (others wait their
    turn instead of exhausting memory), and each visitor must wait
    `cooldown_seconds` between scans. `clock` is injectable for tests."""

    def __init__(self, cooldown_seconds: float = 30.0, clock=time.monotonic):
        self.cooldown, self.clock = cooldown_seconds, clock
        self._lock = threading.Lock()
        self._last: dict[str, float] = {}

    def wait_time(self, visitor: str) -> float:
        """Seconds `visitor` must still wait before scanning again (0 = may scan now)."""
        last = self._last.get(visitor)
        return 0.0 if last is None else max(0.0, self.cooldown - (self.clock() - last))

    @contextlib.contextmanager
    def slot(self, visitor: str):
        """Holds the single scan slot; raises RateLimitedError if `visitor` scanned too recently."""
        wait = self.wait_time(visitor)
        if wait > 0:
            raise RateLimitedError(f"Please wait {wait:.0f} s before scanning again.")
        self._last[visitor] = self.clock()
        with self._lock:
            yield


class RateLimitedError(RuntimeError):
    """A visitor asked for another scan before their cooldown ended."""


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


# ---------------------------------------------------------------- Semgrep on the same code

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
