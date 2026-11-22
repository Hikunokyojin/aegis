"""Making scans cheap to repeat and able to see across functions.

- GraphCache: a function's graph depends only on its language and code, so
  graphs are stored under a SHA-256 of both. Re-scanning a repository then
  only sends new or edited functions through Joern (the slow step).
- call_graph / propagate_risk: the model scores one function at a time, so
  a function that hands user input to a risky helper looks harmless on its
  own. A call graph over the scanned code lets each function inherit risk
  from the functions it calls (directly or a few calls deep), with the path
  shown, so reviewers see flaws that span functions.
- quantize_for_cpu: 8-bit dynamic quantization of the linear layers (the
  bulk of the GNN's and CodeBERT's compute), for faster scans without a GPU.
"""
import hashlib
from pathlib import Path

import torch
import torch.nn as nn

from aegis.parsing.function_extractor import CPP_LANGUAGE, JAVA_LANGUAGE

_CALL_NODES = {"cpp": "call_expression", "java": "method_invocation"}
_LANGUAGES = {"cpp": CPP_LANGUAGE, "java": JAVA_LANGUAGE}


class GraphCache:
    """Graphs on disk, keyed by a hash of (language, analysed source)."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(language: str, source: str) -> str:
        return hashlib.sha256(f"{language}\0{source}".encode()).hexdigest()

    def get(self, key: str):
        path = self.directory / f"{key}.pt"
        return torch.load(path, weights_only=False) if path.exists() else None

    def put(self, key: str, graph) -> None:
        tmp = self.directory / f"{key}.pt.tmp"
        torch.save(graph, tmp)
        tmp.replace(self.directory / f"{key}.pt")  # atomic: a crash never leaves a half-written graph


# ---------------------------------------------------------------- call graph


def _short_name(name: str) -> str:
    """`Widget::resize` -> `resize`, `obj.run` -> `run`, `~Widget` kept as is."""
    return name.replace("->", ".").replace("::", ".").split(".")[-1]


def called_names(source: str, language: str) -> set[str]:
    """Names of the functions/methods called inside `source`."""
    from tree_sitter import Parser

    src = source.encode("utf-8")
    root = Parser(_LANGUAGES[language]).parse(src).root_node
    names, stack = set(), [root]
    while stack:
        node = stack.pop()
        if node.type == _CALL_NODES[language]:
            target = node.child_by_field_name("function" if language == "cpp" else "name")
            if target is not None:
                names.add(_short_name(src[target.start_byte:target.end_byte].decode("utf-8", errors="replace")))
        stack.extend(node.children)
    return names


def call_graph(functions: list[dict]) -> dict[int, set[int]]:
    """{caller index: callee indices} among `functions` (each with function,
    source and language). Calls are matched by name within the same language;
    overloaded names link to every candidate, which over- rather than
    under-approximates, the safe direction for a reviewer."""
    by_name: dict[tuple[str, str], list[int]] = {}
    for i, fn in enumerate(functions):
        by_name.setdefault((fn["language"], _short_name(fn["function"])), []).append(i)
    edges = {}
    for i, fn in enumerate(functions):
        callees = {j for name in called_names(fn["source"], fn["language"])
                   for j in by_name.get((fn["language"], name), []) if j != i}
        edges[i] = callees
    return edges


def propagate_risk(functions: list[dict], edges: dict[int, set[int]], decay: float = 0.8,
                   max_depth: int = 3) -> None:
    """Adds to each function `context_risk` (the highest decay**depth *
    probability among functions reachable through up to max_depth calls) and
    `risk_path` (the call chain to that function, by name)."""
    for i, fn in enumerate(functions):
        best, best_path = 0.0, []
        frontier = [(i, [i])]
        for depth in range(1, max_depth + 1):
            nxt = []
            for node, path in frontier:
                for callee in edges.get(node, ()):
                    if callee in path:
                        continue  # recursion / cycles
                    score = decay ** depth * functions[callee]["probability"]
                    if score > best:
                        best, best_path = score, path + [callee]
                    nxt.append((callee, path + [callee]))
            frontier = nxt
        fn["context_risk"] = best
        fn["risk_path"] = [functions[j]["function"] for j in best_path]


# ---------------------------------------------------------------- lighter models


def quantize_for_cpu(model: nn.Module) -> nn.Module:
    """8-bit dynamic quantization of every nn.Linear (weights stored as int8,
    activations quantized on the fly). CPU only; a no-op worth skipping on GPU."""
    return torch.ao.quantization.quantize_dynamic(model.cpu().eval(), {nn.Linear}, dtype=torch.qint8)
