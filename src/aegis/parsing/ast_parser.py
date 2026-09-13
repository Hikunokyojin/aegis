"""Tree-sitter based AST parsing for C++ and Java source files.

Parses source files into Abstract Syntax Trees, walks each tree to collect
structural statistics (node count, depth) and any syntax errors, and writes
one JSON file per source file plus a combined manifest.
"""
import argparse
import json
from pathlib import Path

import tree_sitter_cpp as tscpp
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

CPP_LANGUAGE = Language(tscpp.language())
JAVA_LANGUAGE = Language(tsjava.language())


def _make_parser(language: Language) -> Parser:
    return Parser(language)


def node_to_dict(node, source_bytes: bytes, depth: int = 0, max_depth: int | None = None) -> dict:
    """Recursively convert a tree-sitter node into a plain JSON-serializable dict."""
    text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    entry = {
        "type": node.type,
        "start_point": list(node.start_point),
        "end_point": list(node.end_point),
        "is_named": node.is_named,
        "has_error": node.has_error,
    }
    if len(node.children) == 0:
        entry["text"] = text
    if max_depth is None or depth < max_depth:
        children = [node_to_dict(c, source_bytes, depth + 1, max_depth) for c in node.children]
        if children:
            entry["children"] = children
    return entry


def count_nodes(node) -> int:
    return 1 + sum(count_nodes(c) for c in node.children)


def tree_depth(node) -> int:
    if not node.children:
        return 1
    return 1 + max(tree_depth(c) for c in node.children)


def collect_errors(node, source_bytes: bytes, errors: list) -> None:
    if node.type == "ERROR" or node.is_missing:
        snippet = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        errors.append({"type": node.type, "at": list(node.start_point), "snippet": snippet[:80]})
    for child in node.children:
        collect_errors(child, source_bytes, errors)


def parse_file(path: Path, parser: Parser, language_label: str, project_root: Path) -> tuple[dict, dict]:
    """Parse one source file. Returns (summary_dict, ast_tree_dict)."""
    source_bytes = path.read_bytes()
    tree = parser.parse(source_bytes)
    root = tree.root_node
    errors: list = []
    collect_errors(root, source_bytes, errors)
    summary = {
        "file": str(path.relative_to(project_root)),
        "language": language_label,
        "total_nodes": count_nodes(root),
        "max_depth": tree_depth(root),
        "syntax_errors": errors,
    }
    return summary, node_to_dict(root, source_bytes)


def parse_directory(cpp_dir: Path | None, java_dir: Path | None, output_dir: Path, project_root: Path) -> list[dict]:
    """Parse every .cpp file under cpp_dir and every .java file under java_dir,
    writing one AST JSON per file plus a manifest.json to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cpp_parser = _make_parser(CPP_LANGUAGE)
    java_parser = _make_parser(JAVA_LANGUAGE)

    cpp_files = sorted(cpp_dir.glob("*.cpp")) if cpp_dir and cpp_dir.exists() else []
    java_files = sorted(java_dir.glob("*.java")) if java_dir and java_dir.exists() else []

    all_summaries = []
    for f in cpp_files:
        summary, tree_dict = parse_file(f, cpp_parser, "cpp", project_root)
        all_summaries.append(summary)
        (output_dir / (f.stem + "_cpp_ast.json")).write_text(json.dumps(tree_dict, indent=2))
        print(f"[cpp]  {f.name:30s} nodes={summary['total_nodes']:5d}  "
              f"depth={summary['max_depth']:3d}  errors={len(summary['syntax_errors'])}")

    for f in java_files:
        summary, tree_dict = parse_file(f, java_parser, "java", project_root)
        all_summaries.append(summary)
        (output_dir / (f.stem + "_java_ast.json")).write_text(json.dumps(tree_dict, indent=2))
        print(f"[java] {f.name:30s} nodes={summary['total_nodes']:5d}  "
              f"depth={summary['max_depth']:3d}  errors={len(summary['syntax_errors'])}")

    (output_dir / "manifest.json").write_text(json.dumps(all_summaries, indent=2))
    total_errors = sum(len(s["syntax_errors"]) for s in all_summaries)
    print(f"\nParsed {len(all_summaries)} files. Total syntax errors found: {total_errors}")
    print(f"AST JSON written to: {output_dir}")
    return all_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse C++/Java source into AST JSON.")
    parser.add_argument("--cpp-dir", type=Path, default=None, help="Directory of .cpp files")
    parser.add_argument("--java-dir", type=Path, default=None, help="Directory of .java files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write AST JSON + manifest")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                         help="Root used to compute relative file paths in the manifest")
    args = parser.parse_args()

    summaries = parse_directory(args.cpp_dir, args.java_dir, args.output_dir, args.project_root)
    total_errors = sum(len(s["syntax_errors"]) for s in summaries)
    if total_errors > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
