"""Build a torch_geometric HeteroData object from a Joern GraphML export.

Pipeline: parse GraphML -> filter to 3 relations (AST, CFG, REACHING_DEF)
-> collapse all Joern node types into one "NODE" type (avoids the
node-type x relation combinatorial explosion) -> attach per-node features as
[one-hot Joern type + CodeBERT embedding of the node's own CODE/NAME text
(768 dims)].
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch_geometric.data import HeteroData
from transformers import AutoModel, AutoTokenizer

NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
# Stable, ordered relation list (not a set) so the model's edge-type metadata
# (see aegis.model.hetero_gnn) is identical every run, regardless of Python's
# string-hash seed.
RELATION_ORDER: tuple[str, ...] = ("AST", "CFG", "REACHING_DEF")
KEEP_RELATIONS = set(RELATION_ORDER)
MODEL_NAME = "microsoft/codebert-base"
TEXT_ATTR_PRIORITY = ["CODE", "NAME"]

# Authoritative node-type vocabulary, extracted with javap from NodeTypes.class
# in Joern's io.shiftleft.codepropertygraph-domain-classes_3-1.7.70.jar (the
# library behind Joern v4.0.597). Every graph uses this same fixed list, so
# every feature vector is the same width (43 + 768 CodeBERT = 811) no matter
# which node types a given function happens to contain. Re-extract it if the
# Joern version changes.
JOERN_NODE_TYPES: tuple[str, ...] = (
    "ANNOTATION", "ANNOTATION_LITERAL", "ANNOTATION_PARAMETER", "ANNOTATION_PARAMETER_ASSIGN",
    "ARRAY_INITIALIZER", "BINDING", "BLOCK", "CALL", "CLOSURE_BINDING", "COMMENT", "CONFIG_FILE",
    "CONTROL_STRUCTURE", "DEPENDENCY", "FIELD_IDENTIFIER", "FILE", "FINDING", "IDENTIFIER", "IMPORT",
    "JUMP_LABEL", "JUMP_TARGET", "KEY_VALUE_PAIR", "LITERAL", "LOCAL", "MEMBER", "META_DATA", "METHOD",
    "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "METHOD_REF", "METHOD_RETURN", "MODIFIER", "NAMESPACE",
    "NAMESPACE_BLOCK", "RETURN", "TAG", "TAG_NODE_PAIR", "TEMPLATE_DOM", "TYPE", "TYPE_ARGUMENT",
    "TYPE_DECL", "TYPE_PARAMETER", "TYPE_REF", "UNKNOWN",
)
NODE_TEXT_MAX_TOKENS = 64


def parse_graphml(xml_path: Path):
    """Extract (node_id -> joern_type), (node_id -> display_text),
    (node_id -> FILENAME, only present for node types that carry it, e.g.
    METHOD), and the filtered edge list [(src, tgt, relation), ...] from one
    Joern GraphML export file. The export may contain one function (as in
    the original single-function pipeline) or many (as produced by batch
    parsing many source files in one Joern invocation — see
    aegis.graphs.multi_method_splitter)."""
    root = ET.parse(xml_path).getroot()

    key_to_attr = {
        key.get("id"): key.get("attr.name")
        for key in root.iter("{http://graphml.graphdrawing.org/xmlns}key")
        if key.get("for") == "node"
    }

    node_label, node_text, node_filename = {}, {}, {}
    for node in root.iter("{http://graphml.graphdrawing.org/xmlns}node"):
        node_id = node.get("id")
        attrs = {}
        for data in node.findall("g:data", NS):
            attr_name = key_to_attr.get(data.get("key"))
            if attr_name:
                attrs[attr_name] = data.text
            if data.get("key") == "labelV":
                node_label[node_id] = data.text

        text = next((attrs[a] for a in TEXT_ATTR_PRIORITY if attrs.get(a)), None)
        node_text[node_id] = text or node_label.get(node_id, "UNKNOWN")
        if attrs.get("FILENAME"):
            node_filename[node_id] = attrs["FILENAME"]

    edges = []
    for edge in root.iter("{http://graphml.graphdrawing.org/xmlns}edge"):
        src, tgt = edge.get("source"), edge.get("target")
        label = next(
            (d.text for d in edge.findall("g:data", NS) if d.get("key") == "labelE"),
            None,
        )
        if label in KEEP_RELATIONS:
            edges.append((src, tgt, label))

    return node_label, node_text, node_filename, edges


def parse_node_lines(xml_path: Path) -> dict[str, int]:
    """{node_id: Joern LINE_NUMBER} for every node that has one."""
    root = ET.parse(xml_path).getroot()
    line_keys = {key.get("id") for key in root.iter("{http://graphml.graphdrawing.org/xmlns}key")
                 if key.get("for") == "node" and key.get("attr.name") == "LINE_NUMBER"}
    lines = {}
    for node in root.iter("{http://graphml.graphdrawing.org/xmlns}node"):
        for data in node.findall("g:data", NS):
            if data.get("key") in line_keys and data.text and data.text.lstrip("-").isdigit():
                lines[node.get("id")] = int(data.text)
    return lines


def embed_nodes(tokenizer, model, texts: list[str], chunk_size: int = 256) -> torch.Tensor:
    """[CLS] embedding per node text, [N, 768], always returned on the CPU.
    Runs in chunks of `chunk_size` texts so very large functions (1,300+
    nodes in real BigVul data) can't run out of GPU memory, and on whatever
    device the model is on."""
    chunks = []
    for start in range(0, len(texts), chunk_size):
        inputs = tokenizer(
            texts[start:start + chunk_size], return_tensors="pt", truncation=True,
            max_length=NODE_TEXT_MAX_TOKENS, padding=True,
        ).to(model.device)
        with torch.no_grad():
            chunks.append(model(**inputs).last_hidden_state[:, 0, :].cpu())
    return torch.cat(chunks) if chunks else torch.zeros((0, 768))


def build_hetero_data_from_parsed(node_ids: list[str], node_label: dict[str, str], node_text: dict[str, str],
                                   edges: list[tuple[str, str, str]], tokenizer, model,
                                   type_vocabulary: list[str] | None = None,
                                   node_line: dict[str, int] | None = None) -> HeteroData:
    """Builds one HeteroData graph restricted to node_ids (a subset of the
    keys of node_label/node_text is fine — e.g. one function's nodes out of
    a larger multi-function parse). Edges are drawn from `edges` and kept
    only when both endpoints are in node_ids.

    type_vocabulary: the one-hot node-type columns to use. Defaults to the
    fixed JOERN_NODE_TYPES list, so every graph has the same feature width.
    A node type outside the vocabulary raises, rather than silently producing
    a mismatched width.
    """
    global_to_local = {nid: i for i, nid in enumerate(node_ids)}

    joern_types = list(type_vocabulary) if type_vocabulary is not None else list(JOERN_NODE_TYPES)
    unknown = {node_label[nid] for nid in node_ids} - set(joern_types)
    if unknown:
        raise ValueError(f"Node type(s) not in the supplied type_vocabulary: {sorted(unknown)}")
    type_to_col = {t: i for i, t in enumerate(joern_types)}

    one_hot = torch.zeros(len(node_ids), len(joern_types))
    for nid in node_ids:
        one_hot[global_to_local[nid], type_to_col[node_label[nid]]] = 1.0

    codebert_features = embed_nodes(tokenizer, model, [node_text[nid] for nid in node_ids])
    combined = torch.cat([one_hot, codebert_features], dim=1)

    data = HeteroData()
    data["NODE"].x = combined
    data["NODE"].joern_type = [node_label[nid] for nid in node_ids]
    data["NODE"].source_text = [node_text[nid] for nid in node_ids]
    if node_line is not None:  # source line per node (-1 if unknown), for explaining predictions
        data["NODE"].line = torch.tensor([node_line.get(nid, -1) for nid in node_ids], dtype=torch.long)

    edge_lists = {rel: ([], []) for rel in KEEP_RELATIONS}
    skipped = 0
    for src, tgt, relation in edges:
        if src not in global_to_local or tgt not in global_to_local:
            skipped += 1
            continue
        edge_lists[relation][0].append(global_to_local[src])
        edge_lists[relation][1].append(global_to_local[tgt])
    for relation, (s, t) in edge_lists.items():
        data["NODE", relation, "NODE"].edge_index = torch.tensor([s, t], dtype=torch.long)

    data.validate(raise_on_error=True)
    assert len(data.edge_types) == 3
    assert data["NODE"].x.shape[1] == len(joern_types) + 768

    if skipped:
        print(f"  WARNING: {skipped} edges skipped (dangling endpoint)")

    return data


def build_graph(xml_path: Path, tokenizer, model) -> HeteroData:
    node_label, node_text, _node_filename, edges = parse_graphml(xml_path)
    node_ids = list(node_label.keys())
    return build_hetero_data_from_parsed(node_ids, node_label, node_text, edges, tokenizer, model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a HeteroData graph from a Joern GraphML export.")
    parser.add_argument("graphml_path", type=Path, help="Path to the Joern GraphML export (.xml)")
    args = parser.parse_args()

    if not args.graphml_path.exists():
        print(f"File not found: {args.graphml_path}")
        raise SystemExit(1)

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"Building graph from: {args.graphml_path}")
    data = build_graph(args.graphml_path, tokenizer, model)

    print(f"\n{data}")
    print(f"\nNode feature matrix: {tuple(data['NODE'].x.shape)}")
    for etype in data.edge_types:
        print(f"  {etype[1]:15s} {data[etype].edge_index.shape[1]} edges")
    print("\nPASS: graph built and validated")


if __name__ == "__main__":
    main()
