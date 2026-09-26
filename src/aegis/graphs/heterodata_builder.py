"""Build a torch_geometric HeteroData object from a Joern GraphML export.

Pipeline: parse GraphML -> filter to 3 relations (AST, CFG, REACHING_DEF)
-> collapse all Joern node types into one "NODE" type (avoids the
node-type x relation combinatorial explosion) -> attach per-node features as
[one-hot Joern type + CodeBERT embedding of the node's own CODE/NAME text
(768 dims)].
"""
import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer, AutoModel

NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
# Stable, ordered relation list (not a set) so any code mapping relation name
# -> integer edge_type index (see aegis.model.graph_batching) gets the same
# mapping every run, regardless of Python's string-hash seed.
RELATION_ORDER: tuple[str, ...] = ("AST", "CFG", "REACHING_DEF")
KEEP_RELATIONS = set(RELATION_ORDER)
MODEL_NAME = "microsoft/codebert-base"
TEXT_ATTR_PRIORITY = ["CODE", "NAME"]
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


def embed_nodes(tokenizer, model, texts: list[str]) -> torch.Tensor:
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=NODE_TEXT_MAX_TOKENS, padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # [CLS] per node, [N, 768]


def build_hetero_data_from_parsed(node_ids: list[str], node_label: dict[str, str], node_text: dict[str, str],
                                   edges: list[tuple[str, str, str]], tokenizer, model,
                                   type_vocabulary: list[str] | None = None) -> HeteroData:
    """Builds one HeteroData graph restricted to node_ids (a subset of the
    keys of node_label/node_text is fine — e.g. one function's nodes out of
    a larger multi-function parse). Edges are drawn from `edges` and kept
    only when both endpoints are in node_ids.

    type_vocabulary: the one-hot node-type columns to use. When omitted, it
    is inferred from the node types actually present in node_ids (the
    original single-file behavior). When building several graphs from the
    same batch (see aegis.graphs.multi_method_splitter), callers should pass
    the SAME explicit type_vocabulary to every call so every graph in that
    batch has an identical, comparable feature width — passing a type not
    present in an explicit type_vocabulary raises, rather than silently
    producing a mismatched width.
    """
    global_to_local = {nid: i for i, nid in enumerate(node_ids)}

    if type_vocabulary is None:
        joern_types = sorted({node_label[nid] for nid in node_ids})
    else:
        joern_types = type_vocabulary
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
