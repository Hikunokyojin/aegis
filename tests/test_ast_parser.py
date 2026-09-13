"""Verifies AST parsing produces error-free, well-formed trees for the
known-good C++ and Java fixtures."""
from pathlib import Path

from aegis.parsing.ast_parser import CPP_LANGUAGE, JAVA_LANGUAGE, parse_file, _make_parser

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_cpp_fixture_parses_without_errors():
    parser = _make_parser(CPP_LANGUAGE)
    path = FIXTURES_DIR / "cpp" / "vuln_strcpy.cpp"

    summary, _ = parse_file(path, parser, "cpp", FIXTURES_DIR)

    assert summary["syntax_errors"] == []
    assert summary["total_nodes"] > 0


def test_java_fixture_parses_without_errors():
    parser = _make_parser(JAVA_LANGUAGE)
    path = FIXTURES_DIR / "java" / "VulnSqlInjection.java"

    summary, _ = parse_file(path, parser, "java", FIXTURES_DIR)

    assert summary["syntax_errors"] == []
    assert summary["total_nodes"] > 0
