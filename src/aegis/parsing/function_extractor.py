"""Extracts individual functions from a C/C++ or Java source file.

The model is trained on one function per graph, so anything scanned or
loaded from whole files (the Juliet test suite, a user's own code) has to be
cut into functions the same way first. Tree-sitter is used rather than
regexes because real code nests functions inside namespaces, classes and
preprocessor blocks (`#ifndef OMITBAD`), which tree-sitter parses
structurally and tolerates even when the file has syntax errors.
"""
from dataclasses import dataclass

from tree_sitter import Parser

from aegis.parsing.ast_parser import CPP_LANGUAGE, JAVA_LANGUAGE

_PARSERS = {"cpp": Parser(CPP_LANGUAGE), "java": Parser(JAVA_LANGUAGE)}
_FUNCTION_NODES = {"cpp": {"function_definition"},
                   "java": {"method_declaration", "constructor_declaration"}}

# Joern's Java frontend only builds a method inside a type declaration.
JAVA_WRAPPER_CLASS = "AegisFunction"


@dataclass(frozen=True)
class ExtractedFunction:
    name: str
    source: str
    start_line: int  # 1-based line of the function's first character
    language: str


def _cpp_function_name(node, source_bytes: bytes) -> str:
    """Follows function_definition -> (pointer/reference declarators) ->
    function_declarator -> its declarator, which is the (possibly qualified)
    name such as `Widget::resize`, `~Widget` or `operator=`."""
    declarator = node.child_by_field_name("declarator")
    while declarator is not None and declarator.type != "function_declarator":
        declarator = declarator.child_by_field_name("declarator")
    if declarator is None:
        return "<anonymous>"
    name = declarator.child_by_field_name("declarator")
    return source_bytes[name.start_byte:name.end_byte].decode("utf-8", errors="replace")


def _java_function_name(node, source_bytes: bytes) -> str:
    name = node.child_by_field_name("name")
    return source_bytes[name.start_byte:name.end_byte].decode("utf-8", errors="replace")


def extract_functions(source: str, language: str) -> list[ExtractedFunction]:
    """Returns every function (C/C++) or method/constructor (Java) defined in
    `source`, in file order. `language` is "cpp" (also used for plain C) or
    "java"."""
    if language not in _PARSERS:
        raise ValueError(f"Unsupported language {language!r}; expected one of {sorted(_PARSERS)}")
    source_bytes = source.encode("utf-8")
    root = _PARSERS[language].parse(source_bytes).root_node
    name_of = _cpp_function_name if language == "cpp" else _java_function_name

    functions = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in _FUNCTION_NODES[language]:
            functions.append(ExtractedFunction(
                name=name_of(node, source_bytes),
                source=source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace"),
                start_line=node.start_point[0] + 1,
                language=language,
            ))
            if language == "cpp":
                continue  # C++ has no nested function definitions; Java may (local/anonymous classes)
        stack.extend(reversed(node.children))
    return functions


def to_standalone(function_source: str, language: str) -> str:
    """Makes one extracted function parseable on its own by Joern: Java
    methods are wrapped in a class, C/C++ functions are returned as-is."""
    if language == "java":
        return f"class {JAVA_WRAPPER_CLASS} {{\n{function_source}\n}}\n"
    return function_source
