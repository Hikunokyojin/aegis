"""Prepares BigVul functions for batch Joern CPG generation.

Joern's GraphML export (see aegis.graphs.heterodata_builder) only works on
a whole CPG at once, not on an individual method within a larger CPG — so
the correct unit of work for per-function graphs is one function per source
file, each becoming its own small CPG. This module leak-free-splits a list
of BigVulFunction records and writes each one to its own file, organized
into train/val/test directories, with a manifest.json per split linking
each written file back to its label and metadata for the batch CPG step
and later training.
"""
import hashlib
import json
from pathlib import Path

from aegis.dataset.bigvul_loader import BigVulFunction, regroup_oversized_projects
from aegis.dataset.leak_free_split import group_aware_split, assert_no_leakage

# Always .cpp: BigVul labels real C++ (e.g. Chrome's `Class::Method` code)
# as "C", and Joern's C frontend emits a single UNKNOWN node for C++ it can't
# parse. Joern's C++ frontend handles plain C too.
LANGUAGE_EXTENSIONS = {"C": ".cpp", "CPP": ".cpp", "C++": ".cpp"}


def language_extension(language: str) -> str:
    return LANGUAGE_EXTENSIONS[language]


def write_split_sources(split_name: str, functions: list[BigVulFunction], output_dir: Path) -> list[dict]:
    """Writes each function's pre-fix source to its own file under
    output_dir/split_name/{sample_id}{ext}. Returns a manifest (list of
    dicts) describing every written file."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for fn in functions:
        ext = language_extension(fn.language)
        file_path = split_dir / f"{fn.sample_id}{ext}"
        file_path.write_text(fn.source_before, encoding="utf-8")
        manifest.append({
            "sample_id": fn.sample_id,
            "source_path": f"{split_name}/{file_path.name}",
            "label": fn.label,
            "language": fn.language,
            "project": fn.function_id.split("::")[0],
            "group": fn.function_id,
            "cve_id": fn.cve_id,
            "cwe_id": fn.cwe_id,
            "commit_id": fn.commit_id,
        })
    return manifest


def _check_or_record_split(output_dir: Path, splits: dict[str, list[BigVulFunction]], settings: dict) -> None:
    """Records which sample went to which split in output_dir/split.json, and
    refuses an output_dir that was built with a different split. Re-runs skip
    graphs that already exist, so reusing a directory with a changed split
    would leave old graphs in one split while the same functions also land
    in another — a leak."""
    assignment = {name: sorted(fn.sample_id for fn in fns) for name, fns in splits.items()}
    fingerprint = hashlib.sha256(json.dumps(assignment).encode()).hexdigest()
    record_path = output_dir / "split.json"
    if record_path.exists():
        recorded = json.loads(record_path.read_text())
        if recorded["fingerprint"] != fingerprint:
            raise ValueError(
                f"{output_dir} was built with a different split (recorded settings: {recorded['settings']}, "
                f"now: {settings}). Use a fresh output directory, or delete this one to rebuild from scratch."
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps({"fingerprint": fingerprint, "settings": settings,
                                       "counts": {name: len(ids) for name, ids in assignment.items()}}, indent=2))


def prepare_dataset(functions: list[BigVulFunction], output_dir: Path,
                     train_frac: float = 0.7, val_frac: float = 0.15,
                     seed: int = 42, max_project_frac: float = 0.10) -> dict[str, list[dict]]:
    """Leak-free splits functions (grouped by project, or by project+file for
    projects above max_project_frac of the dataset) and writes each split to
    output_dir/{train,val,test}/, each with its own manifest.json. Returns the
    same manifests as an in-memory dict keyed by split name."""
    functions = regroup_oversized_projects(functions, max_project_frac=max_project_frac)
    train, val, test = group_aware_split(functions, train_frac=train_frac, val_frac=val_frac, seed=seed)
    assert_no_leakage(train, val, test)
    _check_or_record_split(output_dir, {"train": train, "val": val, "test": test},
                           {"train_frac": train_frac, "val_frac": val_frac, "seed": seed,
                            "max_project_frac": max_project_frac})

    manifests = {}
    for split_name, split_functions in [("train", train), ("val", val), ("test", test)]:
        manifest = write_split_sources(split_name, split_functions, output_dir)
        (output_dir / split_name / "manifest.json").write_text(json.dumps(manifest, indent=2))
        manifests[split_name] = manifest

    return manifests
