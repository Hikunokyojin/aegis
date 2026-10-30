"""Splits a Joern GraphML export containing MANY functions (produced by
parsing many source files in a single Joern invocation) back into one
HeteroData graph per source file.

Why this is needed: Joern's GraphML export (`joern-export --format=graphml`)
only works on a whole CPG at once — there is no per-method GraphML export.
Running Joern once per function (~188K times for the full BigVul dataset)
is impractically slow due to per-invocation JVM startup cost. Batching many
function files into one `joern-parse` + one `joern-export` call is far
faster, but the resulting GraphML mixes every function's nodes and edges
together and must be split back apart afterward.

How the split works: only a handful of node types carry a FILENAME property
directly (METHOD, TYPE_DECL, NAMESPACE_BLOCK, FILE, COMMENT — confirmed
against Joern's own CPG schema, https://cpg.joern.io/). Every other node
belonging to a function is reachable from that function's METHOD node by
following AST edges (AST = Abstract Syntax Tree; a method's own AST is,
by definition, everything reachable from its METHOD node). So grouping by
FILENAME is done by: find every METHOD node's FILENAME, then collect every
node reachable from that METHOD node via AST edges. An edge is kept in the
resulting per-file subgraph only if both endpoints fall in the same group,
which naturally excludes any edge that happens to cross between two
different functions.
"""
from collections import defaultdict
from pathlib import Path

from torch_geometric.data import HeteroData

from aegis.graphs.heterodata_builder import build_hetero_data_from_parsed, parse_graphml, parse_node_lines


def group_node_ids_by_source_file(node_label: dict[str, str], node_filename: dict[str, str],
                                   edges: list[tuple[str, str, str]]) -> dict[str, list[str]]:
    """Returns {filename: [node_id, ...]}, where each group is every node
    reachable via AST edges from a METHOD node tagged with that filename."""
    ast_children = defaultdict(list)
    for src, tgt, relation in edges:
        if relation == "AST":
            ast_children[src].append(tgt)

    groups: dict[str, set[str]] = defaultdict(set)
    for node_id, filename in node_filename.items():
        if node_label.get(node_id) != "METHOD":
            continue  # only METHOD nodes anchor a function's own subtree
        if filename.startswith("<"):
            continue  # Joern pseudo-files like <empty> (external-callee stubs) and <includes>, not real sources
        stack = [node_id]
        while stack:
            current = stack.pop()
            if current in groups[filename]:
                continue
            groups[filename].add(current)
            stack.extend(ast_children.get(current, []))

    return {filename: sorted(node_ids) for filename, node_ids in groups.items()}


def split_combined_graphml(xml_path: Path, tokenizer, model) -> dict[str, HeteroData]:
    """Parses a multi-function Joern GraphML export and returns one
    HeteroData graph per source filename, all using the fixed Joern node-type
    vocabulary (same feature width across every batch).

    Files Joern failed to parse are left out: their only METHOD is Joern's
    file-level `<global>` wrapper, with the unparsed function text stuck in an
    UNKNOWN node. Callers then report them as missing rather than saving an
    empty graph as if it were real."""
    node_label, node_text, node_filename, edges = parse_graphml(xml_path)
    groups = group_node_ids_by_source_file(node_label, node_filename, edges)
    node_line = parse_node_lines(xml_path)

    return {
        filename: build_hetero_data_from_parsed(node_ids, node_label, node_text, edges, tokenizer, model,
                                                node_line=node_line)
        for filename, node_ids in groups.items()
        if any(node_label[nid] == "METHOD" and node_text[nid] != "<global>" for nid in node_ids)
    }
