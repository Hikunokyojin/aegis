"""User journeys covered:

- As someone running Joern on Kaggle/Colab, I want to parse many function
  files in ONE Joern invocation (far fewer JVM startups than one-per-
  function) and still get back one correctly-scoped graph per function, so
  batch CPG generation over 188K+ BigVul functions finishes in a practical
  amount of time.
- As a developer trusting the split output, I want edges that cross between
  two different functions' node sets excluded from both resulting graphs,
  since such an edge cannot represent that they are the same function.
- As a developer benchmarking model results across different files in the
  same batch, I want every graph produced from one combined GraphML export
  to share the same node-type vocabulary (and therefore the same feature
  width), since graphs with a different vocabulary can't be compared or
  trained together within that batch.
"""
from pathlib import Path

import pytest

from aegis.graphs.heterodata_builder import parse_graphml
from aegis.graphs.multi_method_splitter import group_node_ids_by_source_file, split_combined_graphml
from aegis.embeddings.codebert_embedder import load_model

# Two independent functions, each with a METHOD root (carrying FILENAME) and
# one IDENTIFIER child connected by AST+CFG, plus one edge that deliberately
# crosses between the two functions' node sets and must never survive the split.
TWO_METHOD_GRAPHML = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="CODE" for="node" attr.name="CODE" attr.type="string"/>
  <key id="labelV" for="node" attr.name="labelV" attr.type="string"/>
  <key id="FILENAME" for="node" attr.name="FILENAME" attr.type="string"/>
  <key id="labelE" for="edge" attr.name="labelE" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="m1"><data key="labelV">METHOD</data><data key="CODE">copy_input</data><data key="FILENAME">0.c</data></node>
    <node id="i1"><data key="labelV">IDENTIFIER</data><data key="CODE">user_input</data></node>
    <node id="m2"><data key="labelV">METHOD</data><data key="CODE">lookup_user</data><data key="FILENAME">1.cpp</data></node>
    <node id="i2"><data key="labelV">IDENTIFIER</data><data key="CODE">username</data></node>
    <edge source="m1" target="i1"><data key="labelE">AST</data></edge>
    <edge source="m1" target="i1"><data key="labelE">CFG</data></edge>
    <edge source="m2" target="i2"><data key="labelE">AST</data></edge>
    <edge source="m2" target="i2"><data key="labelE">CFG</data></edge>
    <edge source="i1" target="i2"><data key="labelE">REACHING_DEF</data></edge>
  </graph>
</graphml>
"""


@pytest.fixture
def two_method_graphml_path(tmp_path: Path) -> Path:
    path = tmp_path / "combined_export.xml"
    path.write_text(TWO_METHOD_GRAPHML)
    return path


def test_group_node_ids_by_source_file_ignores_non_method_nodes_with_filename():
    """TYPE_DECL nodes also carry FILENAME per Joern's schema, but only
    METHOD nodes should anchor a group — otherwise a type declaration in the
    same file would create a spurious, disconnected second group."""
    node_label = {"m1": "METHOD", "i1": "IDENTIFIER", "t1": "TYPE_DECL"}
    node_filename = {"m1": "0.c", "t1": "0.c"}  # t1 shares the same filename, but isn't a METHOD
    edges = [("m1", "i1", "AST")]

    groups = group_node_ids_by_source_file(node_label, node_filename, edges)

    assert set(groups["0.c"]) == {"m1", "i1"}  # t1 never enters any group on its own


def test_group_node_ids_by_source_file_handles_a_node_reached_by_two_ast_paths():
    """A node reachable from its METHOD via more than one AST path (e.g. two
    distinct AST edges both pointing at the same descendant) must be added
    to the group exactly once, not cause an infinite loop or duplicate."""
    node_label = {"m1": "METHOD", "a": "BLOCK", "b": "BLOCK", "shared": "IDENTIFIER"}
    node_filename = {"m1": "0.c"}
    edges = [
        ("m1", "a", "AST"),
        ("m1", "b", "AST"),
        ("a", "shared", "AST"),
        ("b", "shared", "AST"),  # second AST path reaching the same node
    ]

    groups = group_node_ids_by_source_file(node_label, node_filename, edges)

    assert groups["0.c"] == sorted({"m1", "a", "b", "shared"})


def test_group_node_ids_by_source_file_uses_ast_reachability_from_method_nodes(two_method_graphml_path: Path):
    node_label, node_text, node_filename, edges = parse_graphml(two_method_graphml_path)

    groups = group_node_ids_by_source_file(node_label, node_filename, edges)

    assert set(groups.keys()) == {"0.c", "1.cpp"}
    assert set(groups["0.c"]) == {"m1", "i1"}
    assert set(groups["1.cpp"]) == {"m2", "i2"}


def test_split_combined_graphml_produces_one_graph_per_file(two_method_graphml_path: Path):
    tokenizer, model = load_model()

    graphs = split_combined_graphml(two_method_graphml_path, tokenizer, model)

    assert set(graphs.keys()) == {"0.c", "1.cpp"}
    assert graphs["0.c"]["NODE"].x.shape[0] == 2
    assert graphs["1.cpp"]["NODE"].x.shape[0] == 2


def test_split_combined_graphml_excludes_cross_function_edges(two_method_graphml_path: Path):
    """The REACHING_DEF edge from i1 (function 0.c) to i2 (function 1.cpp)
    must not appear in either resulting graph."""
    tokenizer, model = load_model()

    graphs = split_combined_graphml(two_method_graphml_path, tokenizer, model)

    for filename in ("0.c", "1.cpp"):
        reaching_def_edges = graphs[filename]["NODE", "REACHING_DEF", "NODE"].edge_index
        assert reaching_def_edges.shape[1] == 0


def test_split_combined_graphml_graphs_share_a_common_feature_width(two_method_graphml_path: Path):
    """Both functions' graphs must come out of the SAME batch with the same
    feature width, even though each function individually only touches a
    subset of the batch's full node-type vocabulary."""
    tokenizer, model = load_model()

    graphs = split_combined_graphml(two_method_graphml_path, tokenizer, model)

    assert graphs["0.c"]["NODE"].x.shape[1] == graphs["1.cpp"]["NODE"].x.shape[1]
