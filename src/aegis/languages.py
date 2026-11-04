"""Which source files Aegis analyses, and as which language.

One table shared by the scanner, the web app and the tool baselines, so a
file type is never supported in one place and silently skipped in another.
"""
from pathlib import Path

SOURCE_LANGUAGES = {".c": "cpp", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".h": "cpp", ".hpp": "cpp",
                    ".java": "java"}
# Each function is analysed as its own file: C/C++ always as .cpp (Joern's
# C++ frontend also handles C), Java as .java.
STAGED_EXTENSION = {"cpp": ".cpp", "java": ".java"}


def language_of(path: Path) -> str | None:
    """"cpp", "java", or None for a file Aegis does not analyse."""
    return SOURCE_LANGUAGES.get(path.suffix.lower())


def collect_source_files(paths: list[Path]) -> list[Path]:
    """Every supported source file among `paths` (files, or folders searched recursively)."""
    files = []
    for path in paths:
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        files.extend(p for p in candidates if p.is_file() and language_of(p))
    return files
