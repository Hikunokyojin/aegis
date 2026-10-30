"""Loads the NIST Juliet Test Suite (Java v1.3, C/C++ v1.3) into the same
function records the BigVul loader produces, so both datasets go through one
split -> Joern -> graph -> training pipeline.

Juliet is synthetic: every test case file holds a flawed variant and one or
more fixed variants of the same weakness, told apart only by naming
convention. Labels therefore come from function names:

- `bad`, `badSink`, `CWE..._bad`, `CWE..._52b_badSink`  -> 1 (flawed flow)
- `goodG2B`, `goodB2G1`, `CWE..._goodG2BSink`          -> 0 (fixed flow)
- `good` / `CWE..._good`, `main`, anything else        -> skipped: `good` only
  dispatches to the real good variants, and `main` is test-harness code.
- Class-per-variant cases (`..._81_bad.java`, `..._81_goodG2B.cpp`) name the
  variant in the file instead; every function there takes the file's label.

Label leakage: those same `bad`/`good` names, and Juliet's `/* FLAW */` /
`/* FIX */` comments, sit inside the function source the model reads. Left
in, the model reads the label off the name (a real run scored a meaningless
PR-AUC of 1.0), so every identifier containing bad/good is renamed to a
neutral `fn` form and comments are removed before the source is used.
String literals are left untouched, since they are program behaviour.

Leak-free grouping: a test case is written in many "flow variants" (`_01` to
`_22`, multi-file `_51a/_51b`, class-based `_81_bad`) that wrap the same
flaw in different control/data flow. They are near-duplicates, so they share
one group: the file stem with its variant suffix removed.
"""
import re
from pathlib import Path

from aegis.dataset.bigvul_loader import FunctionRecord
from aegis.parsing.function_extractor import _PARSERS, extract_functions

_EXTENSIONS = {"java": {".java": "JAVA"}, "cpp": {".c": "C", ".cpp": "CPP"}}
_VARIANT_SUFFIX = re.compile(r"_\d+([a-z]|_(bad|good\w*|base|helper\w*))?$")
_FILE_VARIANT = re.compile(r"_\d+_(bad|good\w*)$")
_BAD = re.compile(r"(^|_)bad")
_GOOD = re.compile(r"(^|_)good")
_LABEL_WORD = re.compile(r"(?i)(?:bad|good)(?:G2B|B2G)?\d*")
_IDENTIFIERS = {"identifier", "field_identifier", "type_identifier"}
_COMMENTS = {"comment", "line_comment", "block_comment"}


def neutralize(source: str, language: str) -> str:
    """Removes comments and renames identifiers containing bad/good
    (`goodG2BSource` -> `fnSource`, `CWE121_..._bad` -> `CWE121_..._fn`)."""
    source_bytes = source.encode("utf-8")
    edits = []
    stack = [_PARSERS[language].parse(source_bytes).root_node]
    while stack:
        node = stack.pop()
        if node.type in _COMMENTS:
            edits.append((node.start_byte, node.end_byte, b""))
            continue
        if node.type in _IDENTIFIERS:
            text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            renamed = _LABEL_WORD.sub("fn", text)
            if renamed != text:
                edits.append((node.start_byte, node.end_byte, renamed.encode("utf-8")))
            continue
        stack.extend(node.children)
    for start, end, replacement in sorted(edits, reverse=True):
        source_bytes = source_bytes[:start] + replacement + source_bytes[end:]
    return source_bytes.decode("utf-8", errors="replace")


def juliet_group(stem: str) -> str:
    """`CWE190_Integer_Overflow__int_max_add_51a` -> `CWE190_Integer_Overflow__int_max_add`."""
    return _VARIANT_SUFFIX.sub("", stem)


def label_for_function(name: str, file_stem: str) -> int | None:
    """1 = flawed, 0 = fixed, None = not a labeled test function (see module docstring)."""
    base = name.split("::")[-1]
    file_variant = _FILE_VARIANT.search(file_stem)
    if file_variant:
        return 1 if file_variant.group(1) == "bad" else 0
    if base == "main" or base == "good" or base.endswith("_good"):
        return None
    if _BAD.search(base):
        return 1
    if _GOOD.search(base):
        return 0
    return None


def load_juliet(root: Path, language: str) -> list[FunctionRecord]:
    """Walks every `CWE*` test case file under `root` (an extracted Juliet
    archive; support code such as `testcasesupport/` is skipped because its
    files are not named `CWE*`) and returns one record per labeled function,
    in a stable path order so sample_ids are reproducible."""
    extensions = _EXTENSIONS[language]
    paths = sorted(p for p in root.rglob("CWE*") if p.is_file() and p.suffix in extensions)

    functions: list[FunctionRecord] = []
    for path in paths:
        source = path.read_text(encoding="utf-8", errors="replace")
        cwe = re.match(r"CWE(\d+)", path.stem)
        for fn in extract_functions(source, language):
            label = label_for_function(fn.name, path.stem)
            if label is None:
                continue
            functions.append(FunctionRecord(
                sample_id=len(functions),
                function_id=juliet_group(path.stem),
                label=label,
                language=extensions[path.suffix],
                cve_id="",
                cwe_id=f"CWE-{cwe.group(1)}" if cwe else "",
                commit_id="",
                source_before=neutralize(fn.source, language),
                source_after="",
                file_name=str(path.relative_to(root)),
            ))
    return functions
