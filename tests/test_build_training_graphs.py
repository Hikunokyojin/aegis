"""User journeys covered:

- As someone running this on Kaggle/Colab, I want many function files parsed
  by Joern in batches rather than one Joern invocation per function, so a
  188K-function run finishes in a practical amount of time.
- As a developer trusting this orchestration without a live Joern install
  to test against, I want the actual subprocess invocation injectable (a
  fake runner in tests), so the command-construction and file-discovery
  logic is genuinely verified rather than just assumed to work.
- As someone reviewing a finished batch run, I want functions Joern failed
  to parse reported explicitly as "missing", never silently dropped, since
  BigVul functions are often non-self-contained snippets that can
  legitimately fail to parse.
"""
import json
from pathlib import Path

import pytest
import torch

from aegis.dataset.bigvul_loader import BigVulFunction
from aegis.embeddings.codebert_embedder import load_model
from aegis.pipeline.build_training_graphs import (
    chunk_list,
    joern_parse_command,
    joern_export_command,
    run_joern_batch,
    sample_functions,
    save_graphs,
    build_training_graphs,
)


def test_chunk_list_splits_into_expected_batch_sizes():
    items = list(range(7))

    batches = chunk_list(items, batch_size=3)

    assert batches == [[0, 1, 2], [3, 4, 5], [6]]


def test_sample_functions_is_sized_and_repeatable():
    functions = [_make_function(i, f"p{i % 7}", i % 2) for i in range(50)]

    first = sample_functions(functions, 20, seed=1)
    again = sample_functions(functions, 20, seed=1)

    assert len(first) == 20
    assert [f.sample_id for f in first] == [f.sample_id for f in again]
    assert sample_functions(functions, 500, seed=1) == functions  # asking for more than exist returns all


def test_chunk_list_handles_empty_input():
    assert chunk_list([], batch_size=3) == []


def test_joern_parse_command_has_expected_args():
    cmd = joern_parse_command(Path("/batch/src"), Path("/work/cpg.bin"))

    assert cmd == ["joern-parse", str(Path("/batch/src")), "-o", str(Path("/work/cpg.bin"))]


def test_joern_export_command_has_expected_args():
    cmd = joern_export_command(Path("/work/cpg.bin"), Path("/work/export"))

    assert cmd == [
        "joern-export", str(Path("/work/cpg.bin")),
        "--repr", "all", "--format", "graphml", "--out", str(Path("/work/export")),
    ]


def _fake_runner_writing_xml(xml_content: str):
    """Returns a fake `runner(cmd, check=True)` that, when it sees a
    joern-export command, writes xml_content into the export dir named by
    that command's --out argument — simulating Joern's real side effect
    without needing Joern installed."""
    def runner(cmd, check=True):
        if cmd[0] == "joern-export":
            out_dir = Path(cmd[cmd.index("--out") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "export.xml").write_text(xml_content)
        return None
    return runner


def test_run_joern_batch_returns_discovered_xml_files(tmp_path: Path):
    source_file = tmp_path / "0.c"
    source_file.write_text("void f() {}")
    workspace_dir = tmp_path / "workspace"

    xml_files = run_joern_batch([source_file], workspace_dir, runner=_fake_runner_writing_xml("<graphml/>"))

    assert len(xml_files) == 1
    assert xml_files[0].read_text() == "<graphml/>"


def test_run_joern_batch_clears_leftover_workspace_from_an_interrupted_run(tmp_path: Path):
    """Real joern-export refuses an --out directory that already exists. A
    run killed mid-batch leaves its workspace behind, so the next attempt
    must start from a clean workspace."""
    source_file = tmp_path / "0.cpp"
    source_file.write_text("void f() {}")
    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "export").mkdir(parents=True)
    (workspace_dir / "export" / "stale.xml").write_text("<stale/>")

    def joern_like_runner(cmd, check=True):
        if cmd[0] == "joern-export":
            out_dir = Path(cmd[cmd.index("--out") + 1])
            if out_dir.exists():
                raise RuntimeError("Output directory already exists")
            out_dir.mkdir(parents=True)
            (out_dir / "export.xml").write_text("<graphml/>")

    xml_files = run_joern_batch([source_file], workspace_dir, runner=joern_like_runner)

    assert [f.name for f in xml_files] == ["export.xml"]


def test_run_joern_batch_raises_clear_error_when_no_xml_produced(tmp_path: Path):
    source_file = tmp_path / "0.c"
    source_file.write_text("void f() {}")
    workspace_dir = tmp_path / "workspace"

    def no_op_runner(cmd, check=True):
        return None

    with pytest.raises(RuntimeError, match="no .xml files"):
        run_joern_batch([source_file], workspace_dir, runner=no_op_runner)


def test_save_graphs_writes_pt_files_and_reports_missing(tmp_path: Path):
    tokenizer, model = load_model()
    from torch_geometric.data import HeteroData
    graph = HeteroData()
    graph["NODE"].x = torch.zeros(1, 4)

    manifest = [
        {"sample_id": 0, "source_path": "train/0.c", "label": 1},
        {"sample_id": 1, "source_path": "train/1.c", "label": 0},  # no matching graph
    ]

    saved, missing = save_graphs({"0.c": graph}, manifest, tmp_path)

    assert saved == [0]
    assert missing == [1]
    loaded = torch.load(tmp_path / "0.pt", weights_only=False)
    assert loaded["label"] == 1
    assert not (tmp_path / "1.pt").exists()


def _make_function(sample_id: int, project: str, label: int) -> BigVulFunction:
    return BigVulFunction(
        sample_id=sample_id, function_id=project, label=label, language="C",
        cve_id=f"CVE-TEST-{sample_id}", cwe_id="CWE-120", commit_id=f"c{sample_id}",
        source_before=f"void f{sample_id}() {{ }}", source_after=f"void f{sample_id}() {{ }}",
    )


TWO_METHOD_GRAPHML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="CODE" for="node" attr.name="CODE" attr.type="string"/>
  <key id="labelV" for="node" attr.name="labelV" attr.type="string"/>
  <key id="FILENAME" for="node" attr.name="FILENAME" attr.type="string"/>
  <key id="labelE" for="edge" attr.name="labelE" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="m1"><data key="labelV">METHOD</data><data key="CODE">f0</data><data key="FILENAME">0.cpp</data></node>
    <node id="i1"><data key="labelV">IDENTIFIER</data><data key="CODE">x</data></node>
    <node id="m2"><data key="labelV">METHOD</data><data key="CODE">f1</data><data key="FILENAME">1.cpp</data></node>
    <node id="i2"><data key="labelV">IDENTIFIER</data><data key="CODE">y</data></node>
    <edge source="m1" target="i1"><data key="labelE">AST</data></edge>
    <edge source="m2" target="i2"><data key="labelE">AST</data></edge>
  </graph>
</graphml>
"""


def test_build_training_graphs_end_to_end_with_fake_joern(tmp_path: Path):
    """Full orchestration, with Joern itself faked out (no live install
    available here) but every other step — loading, leak-free splitting,
    source-file writing, batch chunking, GraphML splitting, and saving —
    run for real."""
    functions = [_make_function(0, "alpha", 1), _make_function(1, "alpha", 0)]
    runner = _fake_runner_writing_xml(TWO_METHOD_GRAPHML_TEMPLATE)

    summary = build_training_graphs(
        functions, tmp_path, batch_size=10, train_frac=1.0, val_frac=0.0, seed=1, runner=runner,
    )

    assert summary["train"]["saved"] == 2
    assert summary["train"]["missing"] == 0
    assert (tmp_path / "train" / "graphs" / "0.pt").exists()
    assert (tmp_path / "train" / "graphs" / "1.pt").exists()


def _fake_joern_parsing_only(parseable: set[str], calls: list):
    """Fake runner that 'parses' only the filenames in `parseable`: for each
    of those in the batch it emits a real METHOD, and files outside the set
    get nothing (like Joern's real <global>-only output for code it can't
    parse). Records every command name in `calls`."""
    state = {}

    def runner(cmd, check=True):
        calls.append(cmd[0])
        if cmd[0] == "joern-parse":
            state["batch_files"] = sorted(p.name for p in Path(cmd[1]).iterdir())
        elif cmd[0] == "joern-export":
            nodes, edges = [], []
            for i, name in enumerate(state["batch_files"]):
                if name in parseable:
                    nodes.append(f'<node id="m{i}"><data key="labelV">METHOD</data>'
                                 f'<data key="CODE">void f{i}()</data><data key="FILENAME">{name}</data></node>')
                    nodes.append(f'<node id="x{i}"><data key="labelV">IDENTIFIER</data><data key="CODE">x</data></node>')
                    edges.append(f'<edge source="m{i}" target="x{i}"><data key="labelE">AST</data></edge>')
            xml = ('<?xml version="1.0" encoding="UTF-8"?><graphml xmlns="http://graphml.graphdrawing.org/xmlns">'
                   '<key id="CODE" for="node" attr.name="CODE" attr.type="string"/>'
                   '<key id="labelV" for="node" attr.name="labelV" attr.type="string"/>'
                   '<key id="FILENAME" for="node" attr.name="FILENAME" attr.type="string"/>'
                   '<key id="labelE" for="edge" attr.name="labelE" attr.type="string"/>'
                   '<graph id="G" edgedefault="directed">' + "".join(nodes) + "".join(edges) + "</graph></graphml>")
            out_dir = Path(cmd[cmd.index("--out") + 1])
            out_dir.mkdir(parents=True)
            (out_dir / "export.xml").write_text(xml)
    return runner


def test_functions_that_fail_as_cpp_are_retried_as_c(tmp_path: Path):
    """Seen in a real run: BigVul often drops a function's return type
    (`Ins_IF(...) {`). Plain C accepts that (implicit int), C++ doesn't. So
    anything Joern fails to parse as .cpp gets one more try as .c."""
    functions = [_make_function(0, "alpha", 1), _make_function(1, "alpha", 0)]
    calls = []

    summary = build_training_graphs(functions, tmp_path, batch_size=10, train_frac=1.0, val_frac=0.0, seed=1,
                                    runner=_fake_joern_parsing_only({"0.cpp", "1.c"}, calls))

    assert summary["train"]["saved"] == 2
    assert summary["train"]["missing"] == 0
    assert (tmp_path / "train" / "graphs" / "1.pt").exists()
    assert calls.count("joern-parse") == 2  # the .cpp pass, then one .c retry


def test_function_failing_as_both_cpp_and_c_is_reported_missing(tmp_path: Path):
    functions = [_make_function(0, "alpha", 1), _make_function(1, "alpha", 0)]

    summary = build_training_graphs(functions, tmp_path, batch_size=10, train_frac=1.0, val_frac=0.0, seed=1,
                                    runner=_fake_joern_parsing_only({"0.cpp"}, []))

    assert summary["train"]["saved"] == 1
    assert summary["train"]["missing_ids"] == [1]


def test_no_c_retry_when_every_function_parses_as_cpp(tmp_path: Path):
    functions = [_make_function(0, "alpha", 1), _make_function(1, "alpha", 0)]
    calls = []

    build_training_graphs(functions, tmp_path, batch_size=10, train_frac=1.0, val_frac=0.0, seed=1,
                          runner=_fake_joern_parsing_only({"0.cpp", "1.cpp"}, calls))

    assert calls.count("joern-parse") == 1


def test_build_training_graphs_rerun_skips_functions_already_saved(tmp_path: Path):
    """A full run needs more than one Kaggle/Colab session. Re-running the
    same command after an interruption must pick up where it left off, not
    re-run Joern and CodeBERT on everything again."""
    functions = [_make_function(0, "alpha", 1), _make_function(1, "alpha", 0)]
    build_training_graphs(functions, tmp_path, batch_size=10, train_frac=1.0, val_frac=0.0, seed=1,
                          runner=_fake_runner_writing_xml(TWO_METHOD_GRAPHML_TEMPLATE))

    calls = []

    def counting_runner(cmd, check=True):
        calls.append(cmd[0])

    summary = build_training_graphs(functions, tmp_path, batch_size=10, train_frac=1.0, val_frac=0.0, seed=1,
                                    runner=counting_runner)

    assert calls == []
    assert summary["train"]["already_done"] == 2
    assert summary["train"]["saved"] == 0
    assert summary["train"]["missing"] == 0
