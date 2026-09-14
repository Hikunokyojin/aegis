"""Loads the BigVul vulnerability dataset CSV into structured function
records, ready for leak-free splitting and downstream CPG generation.

Grouping note: per the project's master spec, leak-free splitting must group
at the project/commit level, never the individual function level, since
near-duplicate functions from the same project are the actual leakage vector
(a function's pre-fix and post-fix revisions both appearing in the dataset).
BigVulFunction.function_id is therefore set to the source project name, not
a per-function identifier, and BigVulFunction is structurally compatible
with aegis.dataset.leak_free_split (it exposes the same function_id
attribute group_aware_split/assert_no_leakage rely on).
"""
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

# BigVul function bodies can exceed Python csv's default 128KB field cap.
csv.field_size_limit(sys.maxsize)

SUPPORTED_LANGUAGES = {"C", "CPP", "C++"}


@dataclass
class BigVulFunction:
    sample_id: int
    function_id: str  # source project name (leak-free split group key)
    label: int         # 1 = vulnerable, 0 = safe
    language: str
    cve_id: str
    cwe_id: str
    commit_id: str
    source_before: str
    source_after: str


def load_bigvul_csv(csv_path: Path) -> list[BigVulFunction]:
    """Reads a BigVul-format CSV and returns one BigVulFunction per row.
    Rows with an unrecognized `lang` value are silently skipped, since
    BigVul is documented as C/C++-only and any other value indicates a
    malformed or unexpected row rather than data we can parse."""
    functions = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for sample_id, row in enumerate(reader):
            language = row["lang"]
            if language not in SUPPORTED_LANGUAGES:
                continue
            functions.append(BigVulFunction(
                sample_id=sample_id,
                function_id=row["project"],
                label=int(row["vul"]),
                language=language,
                cve_id=row["CVE ID"],
                cwe_id=row["CWE ID"],
                commit_id=row["commit_id"],
                source_before=row["func_before"],
                source_after=row["func_after"],
            ))
    return functions
