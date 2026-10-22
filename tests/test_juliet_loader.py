from pathlib import Path

from aegis.dataset.juliet_loader import juliet_group, label_for_function, load_juliet

JAVA_CASE = """
package testcases.CWE190_Integer_Overflow.s01;
import testcasesupport.*;
public class CWE190_Integer_Overflow__int_max_add_01 extends AbstractTestCase {
    public void bad() throws Throwable { int data = Integer.MAX_VALUE; IO.writeLine(data + 1); }
    public void good() throws Throwable { goodG2B(); }
    private void goodG2B() throws Throwable { int data = 2; IO.writeLine(data + 1); }
    public static void main(String[] args) throws Throwable { mainFromParent(args); }
}
"""

C_CASE = """
#include "std_testcase.h"
#ifndef OMITBAD
void CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_01_bad() { char b[1]; memcpy(b, "xy", 2); }
#endif
#ifndef OMITGOOD
static void goodG2B() { char b[2]; memcpy(b, "x", 1); }
void CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_01_good() { goodG2B(); }
#endif
int main(int argc, char * argv[]) { return 0; }
"""


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_label_rules_follow_juliet_naming():
    assert label_for_function("bad", "X_01") == 1
    assert label_for_function("badSink", "X_51b") == 1
    assert label_for_function("CWE121_x_01_bad", "X_01") == 1
    assert label_for_function("goodG2B", "X_01") == 0
    assert label_for_function("CWE121_x_52b_goodB2GSink", "X_52b") == 0
    # dispatchers and harness code carry no flaw/fix of their own
    assert label_for_function("good", "X_01") is None
    assert label_for_function("CWE121_x_01_good", "X_01") is None
    assert label_for_function("main", "X_01") is None
    # class-per-variant cases (_81_bad.java) are labeled by their file
    assert label_for_function("action", "X_81_bad") == 1
    assert label_for_function("action", "X_81_goodG2B") == 0
    assert label_for_function("action", "X_81_base") is None


def test_flow_variants_of_one_test_case_share_a_group():
    base = "CWE190_Integer_Overflow__int_max_add"
    for stem in (f"{base}_01", f"{base}_17", f"{base}_51a", f"{base}_81_bad", f"{base}_81_goodG2B"):
        assert juliet_group(stem) == base


def test_loads_java_and_c_cases_with_labels_groups_and_cwe(tmp_path):
    _write(tmp_path, "src/testcases/CWE190_Integer_Overflow/s01/CWE190_Integer_Overflow__int_max_add_01.java", JAVA_CASE)
    _write(tmp_path, "src/testcasesupport/IO.java", "public class IO { public static void bad() {} }")
    _write(tmp_path, "testcases/CWE121/s01/CWE121_Stack_Based_Buffer_Overflow__char_type_overrun_memcpy_01.c", C_CASE)

    java = load_juliet(tmp_path, "java")
    assert [f.label for f in java] == [1, 0]
    assert {f.function_id for f in java} == {"CWE190_Integer_Overflow__int_max_add"}
    assert all(f.language == "JAVA" and f.cwe_id == "CWE-190" for f in java)

    c = load_juliet(tmp_path, "cpp")
    assert [f.label for f in c] == [1, 0]
    assert c[0].cwe_id == "CWE-121" and c[0].language == "C"
    assert [f.sample_id for f in c] == [0, 1]


def test_neutralize_hides_label_revealing_names_and_comments():
    """Juliet names flawed code `bad*` and fixed code `good*`. Left in the
    source, the model reads the label off the name (a real Kaggle run scored a
    meaningless PR-AUC of 1.0), so names and comments are neutralized."""
    from aegis.dataset.juliet_loader import neutralize

    java = ('public void goodG2B() throws Throwable { /* FIX: use a hardcoded string */\n'
            '    int data = goodG2BSource(); // FLAW\n'
            '    IO.writeLine("good bad");\n}')
    out = neutralize(java, "java")
    assert "good" not in out.replace('"good bad"', "") and "bad" not in out.replace('"good bad"', "")
    assert "FIX" not in out and "FLAW" not in out
    assert '"good bad"' in out  # string contents are code behaviour, not labels

    c = "void CWE121_x_01_bad() { badSink(data); }"
    out_c = neutralize(c, "cpp")
    assert "bad" not in out_c and "CWE121_x_01_" in out_c


def test_loaded_juliet_sources_are_neutralized(tmp_path):
    _write(tmp_path, "CWE190_Integer_Overflow__int_max_add_01.java", JAVA_CASE)
    for fn in load_juliet(tmp_path, "java"):
        assert "bad" not in fn.source_before and "good" not in fn.source_before
