import pytest

from aegis.parsing.function_extractor import extract_functions, to_standalone

CPP_SOURCE = """
#include <string.h>
namespace demo {
void Widget::resize(int n) { buf = new char[n]; }
}
#ifndef OMITBAD
void CWE121_bad() { char b[10]; strcpy(b, "0123456789abc"); }
#endif
static int helper(int x) { return x + 1; }
"""

JAVA_SOURCE = """
package testcases;
public class Example {
    public Example() { }
    public void bad() throws Throwable { int x = Integer.MAX_VALUE; x++; }
    private void goodG2B() { int x = 1; x++; }
}
"""


def test_extracts_cpp_functions_including_qualified_and_preprocessor_wrapped():
    functions = extract_functions(CPP_SOURCE, "cpp")
    assert [f.name for f in functions] == ["Widget::resize", "CWE121_bad", "helper"]
    bad = functions[1]
    assert bad.source.startswith("void CWE121_bad()") and bad.source.endswith("}")
    assert bad.start_line == 7


def test_extracts_java_methods_and_constructors():
    functions = extract_functions(JAVA_SOURCE, "java")
    assert [f.name for f in functions] == ["Example", "bad", "goodG2B"]
    assert functions[1].source.startswith("public void bad()")


def test_java_standalone_source_is_wrapped_in_a_class_and_parses_cleanly():
    method = extract_functions(JAVA_SOURCE, "java")[1]
    wrapped = to_standalone(method.source, "java")
    assert wrapped.startswith("class ") and method.source in wrapped
    assert [f.name for f in extract_functions(wrapped, "java")] == ["bad"]


def test_cpp_standalone_source_is_unchanged():
    assert to_standalone("int f() { return 0; }", "cpp") == "int f() { return 0; }"


def test_unsupported_language_is_rejected():
    with pytest.raises(ValueError):
        extract_functions("x", "python")
