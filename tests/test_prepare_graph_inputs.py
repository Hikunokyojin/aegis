"""User journeys covered:

- As someone about to run Joern on Kaggle/Colab, I want BigVul functions
  split leak-free and written to disk as one source file per function
  (matching Joern's per-function CPG granularity, since Joern's GraphML
  export only works on a whole CPG at once, not per-method), organized into
  train/val/test directories, with a manifest linking each file back to its
  label and metadata, so the batch CPG-generation step has a stable,
  resumable list of work to do.
- As a developer debugging a failed batch run, I want each written source
  file named by its own sample_id and given the correct extension for its
  language, so Joern parses it with the right grammar and any failure is
  traceable back to a specific dataset row.
"""
import dataclasses
import json
from pathlib import Path

import pytest

from aegis.dataset.bigvul_loader import BigVulFunction
from aegis.pipeline.prepare_graph_inputs import (
    language_extension,
    write_split_sources,
    prepare_dataset,
)


def _make_function(sample_id: int, project: str, label: int, language: str = "C") -> BigVulFunction:
    return BigVulFunction(
        sample_id=sample_id,
        function_id=project,
        label=label,
        language=language,
        cve_id=f"CVE-TEST-{sample_id}",
        cwe_id="CWE-120",
        commit_id=f"commit{sample_id}",
        source_before=f"void f{sample_id}() {{ /* before */ }}",
        source_after=f"void f{sample_id}() {{ /* after */ }}",
    )


def test_language_extension_is_cpp_for_every_language():
    """BigVul labels real C++ (e.g. Chrome's `Class::Method` code) as "C".
    Written as .c, Joern's C frontend can't parse it and silently emits a
    single UNKNOWN node — 43 of 100 functions in a real run. Joern's C++
    frontend also parses plain C, so .cpp is always safe."""
    assert language_extension("C") == ".cpp"
    assert language_extension("CPP") == ".cpp"
    assert language_extension("C++") == ".cpp"


def test_write_split_sources_creates_one_file_per_function(tmp_path: Path):
    functions = [_make_function(0, "alpha", 1, "C"), _make_function(1, "alpha", 0, "CPP")]

    manifest = write_split_sources("train", functions, tmp_path)

    assert len(manifest) == 2
    written_files = sorted((tmp_path / "train").iterdir())
    assert [f.name for f in written_files] == ["0.cpp", "1.cpp"]
    assert "/* before */" in (tmp_path / "train" / "0.cpp").read_text()


def test_write_split_sources_manifest_has_expected_fields(tmp_path: Path):
    functions = [_make_function(0, "alpha", 1, "C")]

    manifest = write_split_sources("train", functions, tmp_path)

    entry = manifest[0]
    assert entry["sample_id"] == 0
    assert entry["label"] == 1
    assert entry["language"] == "C"
    assert entry["project"] == "alpha"
    assert entry["cve_id"] == "CVE-TEST-0"
    assert entry["source_path"] == "train/0.cpp"


def test_prepare_dataset_produces_leak_free_splits(tmp_path: Path):
    functions = [_make_function(i, "alpha" if i < 5 else "beta", i % 2) for i in range(10)]

    manifests = prepare_dataset(functions, tmp_path, train_frac=0.5, val_frac=0.5, seed=1)

    projects_per_split = {
        split: {entry["project"] for entry in entries}
        for split, entries in manifests.items()
    }
    train_projects, val_projects, test_projects = (
        projects_per_split["train"], projects_per_split["val"], projects_per_split["test"]
    )
    assert not (train_projects & val_projects)
    assert not (train_projects & test_projects)
    assert not (val_projects & test_projects)


def _giant_plus_small_projects() -> list[BigVulFunction]:
    """A 'giant' project (60% of functions, 12 files) plus 8 small projects,
    with vulnerable functions spread through both."""
    functions = []
    for i in range(60):
        f = _make_function(i, "giant", i % 5 == 0)
        functions.append(dataclasses.replace(f, file_name=f"g{i % 12}.cc"))
    for i in range(40):
        f = _make_function(60 + i, f"small{i % 8}", i % 5 == 0)
        functions.append(dataclasses.replace(f, file_name="s.c"))
    return functions


def test_prepare_dataset_splits_an_oversized_project_by_file(tmp_path: Path):
    manifests = prepare_dataset(_giant_plus_small_projects(), tmp_path)

    splits_with_giant = {split for split, entries in manifests.items()
                         if any(e["project"] == "giant" for e in entries)}
    assert len(splits_with_giant) > 1  # no longer forced whole into one split
    for file_group in {e["group"] for entries in manifests.values() for e in entries}:
        assert sum(any(e["group"] == file_group for e in entries) for entries in manifests.values()) == 1


def test_prepare_dataset_refuses_an_output_dir_made_with_a_different_split(tmp_path: Path):
    """Reusing an output dir with a different split would leave old graphs
    in, say, train while the same functions now also land in test — a leak."""
    functions = _giant_plus_small_projects()
    prepare_dataset(functions, tmp_path, train_frac=0.7, val_frac=0.15)

    with pytest.raises(ValueError, match="different split"):
        prepare_dataset(functions, tmp_path, train_frac=0.5, val_frac=0.25)


def test_prepare_dataset_accepts_the_same_split_again(tmp_path: Path):
    functions = _giant_plus_small_projects()
    prepare_dataset(functions, tmp_path)

    prepare_dataset(functions, tmp_path)  # must not raise


def test_prepare_dataset_writes_manifest_json_per_split(tmp_path: Path):
    functions = [_make_function(i, "alpha", i % 2) for i in range(4)]

    prepare_dataset(functions, tmp_path, train_frac=1.0, val_frac=0.0, seed=1)

    manifest_path = tmp_path / "train" / "manifest.json"
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text())
    assert len(loaded) == 4


def test_java_functions_are_written_as_java_wrapped_in_a_class(tmp_path: Path):
    """Juliet Java methods are extracted from their test-case classes; Joern's
    Java frontend only builds a method that sits inside a type declaration."""
    fn = _make_function(0, "alpha", 1, "JAVA")
    fn.source_before = "public void bad() { int x = 1; }"

    manifest = write_split_sources("train", [fn], tmp_path)

    assert manifest[0]["source_path"] == "train/0.java"
    written = (tmp_path / "train" / "0.java").read_text()
    assert written.startswith("class ") and "public void bad() { int x = 1; }" in written
