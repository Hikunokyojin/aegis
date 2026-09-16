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
import json
from pathlib import Path

from aegis.dataset.bigvul_loader import BigVulFunction
from aegis.dataset.leak_free_split import group_aware_split, assert_no_leakage

LANGUAGE_EXTENSIONS = {"C": ".c", "CPP": ".cpp", "C++": ".cpp"}


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
            "project": fn.function_id,
            "cve_id": fn.cve_id,
            "cwe_id": fn.cwe_id,
            "commit_id": fn.commit_id,
        })
    return manifest


def prepare_dataset(functions: list[BigVulFunction], output_dir: Path,
                     train_frac: float = 0.7, val_frac: float = 0.15,
                     seed: int = 42) -> dict[str, list[dict]]:
    """Leak-free splits functions (grouped by project) and writes each split
    to output_dir/{train,val,test}/, each with its own manifest.json.
    Returns the same manifests as an in-memory dict keyed by split name."""
    train, val, test = group_aware_split(functions, train_frac=train_frac, val_frac=val_frac, seed=seed)
    assert_no_leakage(train, val, test)

    manifests = {}
    for split_name, split_functions in [("train", train), ("val", val), ("test", test)]:
        manifest = write_split_sources(split_name, split_functions, output_dir)
        (output_dir / split_name / "manifest.json").write_text(json.dumps(manifest, indent=2))
        manifests[split_name] = manifest

    return manifests
