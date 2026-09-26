"""Verifies HeteroData graph construction against a minimal synthetic Joern
GraphML export (avoids requiring a live Joern install in CI)."""
from pathlib import Path

import pytest

from aegis.graphs.heterodata_builder import build_graph, build_hetero_data_from_parsed
from aegis.embeddings.codebert_embedder import load_model

FIXTURES_DIR = Path(__file__).parent / "fixtures"

MINIMAL_GRAPHML = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="CODE" for="node" attr.name="CODE" attr.type="string"/>
  <key id="labelV" for="node" attr.name="labelV" attr.type="string"/>
  <key id="labelE" for="edge" attr.name="labelE" attr.type="string"/>
  <graph id="G" edgedefault="directed">
    <node id="n1"><data key="labelV">METHOD</data><data key="CODE">copy_input</data></node>
    <node id="n2"><data key="labelV">IDENTIFIER</data><data key="CODE">user_input</data></node>
    <edge source="n1" target="n2"><data key="labelE">AST</data></edge>
    <edge source="n1" target="n2"><data key="labelE">CFG</data></edge>
  </graph>
</graphml>
"""


@pytest.fixture
def minimal_graphml_path(tmp_path: Path) -> Path:
    path = tmp_path / "minimal_export.xml"
    path.write_text(MINIMAL_GRAPHML)
    return path


def test_build_graph_produces_expected_shape(minimal_graphml_path: Path):
    tokenizer, model = load_model()

    data = build_graph(minimal_graphml_path, tokenizer, model)

    assert len(data.edge_types) == 3  # AST, CFG, REACHING_DEF (even if empty)
    assert data["NODE"].x.shape[0] == 2  # two nodes in the fixture
    assert data["NODE"].x.shape[1] > 768  # one-hot type dims + 768 CodeBERT dims


def test_build_hetero_data_from_parsed_with_explicit_vocabulary_gives_consistent_width():
    """Two graphs built from DIFFERENT node subsets but the SAME externally
    supplied type_vocabulary must produce feature vectors of the same
    width — this is what makes graphs from different files in the same
    batch comparable/trainable together."""
    tokenizer, model = load_model()
    node_label = {"a": "METHOD", "b": "IDENTIFIER", "c": "CALL"}
    node_text = {"a": "foo", "b": "x", "c": "bar"}
    edges = [("a", "b", "AST"), ("a", "c", "AST")]
    shared_vocab = ["CALL", "IDENTIFIER", "METHOD"]  # superset, alphabetical

    data_ab = build_hetero_data_from_parsed(["a", "b"], node_label, node_text, edges, tokenizer, model,
                                             type_vocabulary=shared_vocab)
    data_ac = build_hetero_data_from_parsed(["a", "c"], node_label, node_text, edges, tokenizer, model,
                                             type_vocabulary=shared_vocab)

    assert data_ab["NODE"].x.shape[1] == data_ac["NODE"].x.shape[1] == len(shared_vocab) + 768


def test_build_hetero_data_from_parsed_rejects_type_outside_vocabulary():
    """A node type absent from an explicitly supplied vocabulary must raise,
    never silently produce an inconsistent or truncated feature vector."""
    tokenizer, model = load_model()
    node_label = {"a": "METHOD", "b": "SOME_UNKNOWN_TYPE"}
    node_text = {"a": "foo", "b": "bar"}
    edges = []

    with pytest.raises(ValueError, match="SOME_UNKNOWN_TYPE"):
        build_hetero_data_from_parsed(["a", "b"], node_label, node_text, edges, tokenizer, model,
                                       type_vocabulary=["METHOD"])
