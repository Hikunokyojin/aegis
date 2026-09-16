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
import json
from pathlib import Path

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


def test_language_extension_maps_known_languages():
    assert language_extension("C") == ".c"
    assert language_extension("CPP") == ".cpp"
    assert language_extension("C++") == ".cpp"


def test_write_split_sources_creates_one_file_per_function(tmp_path: Path):
    functions = [_make_function(0, "alpha", 1, "C"), _make_function(1, "alpha", 0, "CPP")]

    manifest = write_split_sources("train", functions, tmp_path)

    assert len(manifest) == 2
    written_files = sorted((tmp_path / "train").iterdir())
    assert [f.name for f in written_files] == ["0.c", "1.cpp"]
    assert "/* before */" in (tmp_path / "train" / "0.c").read_text()


def test_write_split_sources_manifest_has_expected_fields(tmp_path: Path):
    functions = [_make_function(0, "alpha", 1, "C")]

    manifest = write_split_sources("train", functions, tmp_path)

    entry = manifest[0]
    assert entry["sample_id"] == 0
    assert entry["label"] == 1
    assert entry["language"] == "C"
    assert entry["project"] == "alpha"
    assert entry["cve_id"] == "CVE-TEST-0"
    assert entry["source_path"] == "train/0.c"


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


def test_prepare_dataset_writes_manifest_json_per_split(tmp_path: Path):
    functions = [_make_function(i, "alpha", i % 2) for i in range(4)]

    prepare_dataset(functions, tmp_path, train_frac=1.0, val_frac=0.0, seed=1)

    manifest_path = tmp_path / "train" / "manifest.json"
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text())
    assert len(loaded) == 4
