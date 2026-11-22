"""Extracts individual functions from a C/C++ or Java source file.

The model is trained on one function per graph, so anything scanned or
loaded from whole files (the Juliet test suite, a user's own code) has to be
cut into functions the same way first. Tree-sitter is used rather than
regexes because real code nests functions inside namespaces, classes and
preprocessor blocks (`#ifndef OMITBAD`), which tree-sitter parses
structurally and tolerates even when the file has syntax errors.
"""
import re
import textwrap
from dataclasses import dataclass

import tree_sitter_cpp as tscpp
import tree_sitter_go as tsgo
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjs
import tree_sitter_python as tspy
from tree_sitter import Language, Parser

CPP_LANGUAGE = Language(tscpp.language())
JAVA_LANGUAGE = Language(tsjava.language())

_PARSERS = {"cpp": Parser(CPP_LANGUAGE), "java": Parser(JAVA_LANGUAGE), "python": Parser(Language(tspy.language())),
            "javascript": Parser(Language(tsjs.language())), "go": Parser(Language(tsgo.language()))}
_FUNCTION_NODES = {"cpp": {"function_definition"},
                   "java": {"method_declaration", "constructor_declaration"},
                   "python": {"function_definition"},
                   "javascript": {"function_declaration", "generator_function_declaration", "method_definition"},
                   "go": {"function_declaration", "method_declaration"}}
# Lines a wrapper adds above the function in its standalone file (see to_standalone).
WRAPPER_LINES = {"cpp": 0, "java": 1, "python": 0, "javascript": 1, "go": 1}

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
    name_of = {"cpp": _cpp_function_name}.get(language, _java_function_name)  # others use a `name` field
    if language not in _PARSERS:
        raise ValueError(f"Unsupported language {language!r}; expected one of {sorted(_PARSERS)}")
    source_bytes = source.encode("utf-8")
    root = _PARSERS[language].parse(source_bytes).root_node

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
            if language in ("cpp", "go"):
                continue  # no nested named functions; Java, Python and JavaScript may nest them
        stack.extend(reversed(node.children))
    return functions


def to_standalone(function_source: str, language: str) -> str:
    """Makes one extracted function parseable on its own by Joern: Java and
    JavaScript methods are wrapped in a class, Go functions get a package
    line, Python methods are dedented; C/C++ functions are returned as-is.
    WRAPPER_LINES records how many lines each wrapper adds above the code."""
    if language == "java":
        return f"class {JAVA_WRAPPER_CLASS} {{\n{function_source}\n}}\n"
    if language == "javascript":
        if re.match(r"(async\s+)?function\b", function_source):  # a plain function: valid as is
            return f"// {JAVA_WRAPPER_CLASS}\n{function_source}\n"
        return f"class {JAVA_WRAPPER_CLASS} {{\n{function_source}\n}}\n"  # a method needs a class
    if language == "go":
        return f"package main\n{function_source}\n"
    if language == "python":  # a method's body keeps its indentation; re-base it one level under `def`
        first, _, rest = function_source.partition("\n")
        body = textwrap.indent(textwrap.dedent(rest), "    ") if rest else ""
        return first.lstrip() + ("\n" + body if body else "") + "\n"
    return function_source
