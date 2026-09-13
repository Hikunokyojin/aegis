"""Verifies HeteroData graph construction against a minimal synthetic Joern
GraphML export (avoids requiring a live Joern install in CI)."""
from pathlib import Path

import pytest

from aegis.graphs.heterodata_builder import build_graph
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
