"""User journeys covered:

- As a developer preparing training data, I want the BigVul CSV loaded into
  structured function records, so downstream CPG generation and the
  leak-free splitter can consume it without touching raw CSV parsing.
- As someone relying on the leak-free splitter, I want BigVul records to
  group by project (per the master spec's leak-free rule: split at the
  project/commit level, never the function level), so near-duplicate
  functions from the same project never end up on both sides of a split.
- As someone running the C++/Java pipeline, I want unrecognized languages
  skipped rather than silently mis-parsed, since BigVul only ever contains
  C/C++ functions.
"""
from pathlib import Path

from aegis.dataset.bigvul_loader import load_bigvul_csv, BigVulFunction, regroup_oversized_projects
from aegis.dataset.leak_free_split import group_aware_split, assert_no_leakage

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "bigvul_sample.csv"


def test_loads_expected_number_of_functions():
    functions = load_bigvul_csv(FIXTURE_PATH)

    assert len(functions) == 4
    assert all(isinstance(f, BigVulFunction) for f in functions)


def test_vul_column_is_parsed_as_integer_label():
    functions = load_bigvul_csv(FIXTURE_PATH)

    labels = {f.commit_id: f.label for f in functions}
    assert labels["abc123"] == 1
    assert labels["def456"] == 0
    assert labels["ghi789"] == 1
    assert labels["jkl012"] == 0


def test_function_id_is_the_project_not_the_individual_function():
    """Per the master spec: leak-free splitting groups at the project/commit
    level, never the individual function level."""
    functions = load_bigvul_csv(FIXTURE_PATH)

    project_by_commit = {f.commit_id: f.function_id for f in functions}
    assert project_by_commit["abc123"] == "alpha"
    assert project_by_commit["def456"] == "alpha"
    assert project_by_commit["ghi789"] == "beta"
    assert project_by_commit["jkl012"] == "beta"


def test_source_before_and_after_are_preserved():
    functions = load_bigvul_csv(FIXTURE_PATH)
    vulnerable = next(f for f in functions if f.commit_id == "abc123")

    assert "strcpy" in vulnerable.source_before
    assert "strncpy" in vulnerable.source_after


def test_file_name_is_preserved():
    functions = load_bigvul_csv(FIXTURE_PATH)

    assert {f.commit_id: f.file_name for f in functions}["abc123"] == "test.c"


def _fn(sample_id: int, project: str, file_name: str) -> BigVulFunction:
    return BigVulFunction(sample_id=sample_id, function_id=project, label=0, language="C", cve_id="",
                          cwe_id="", commit_id="", source_before="", source_after="", file_name=file_name)


def test_oversized_projects_are_regrouped_by_file():
    """Chrome (41% of BigVul) and Linux (25%) can't fit a 15% split as whole
    projects. Projects above max_project_frac are grouped by (project, file)
    instead, so every version of a file's functions still stays together."""
    functions = ([_fn(i, "giant", f"f{i % 3}.cc") for i in range(60)]
                 + [_fn(60 + i, f"small{i % 4}", "x.c") for i in range(40)])

    regrouped = regroup_oversized_projects(functions, max_project_frac=0.10)

    giant_groups = {f.function_id for f in regrouped if f.function_id.startswith("giant")}
    assert giant_groups == {"giant::f0.cc", "giant::f1.cc", "giant::f2.cc"}
    assert {f.function_id for f in regrouped[60:]} == {"small0", "small1", "small2", "small3"}


def test_projects_under_the_threshold_stay_whole():
    functions = [_fn(i, f"p{i % 20}", f"f{i}.c") for i in range(100)]  # each project = 5%

    regrouped = regroup_oversized_projects(functions, max_project_frac=0.10)

    assert [f.function_id for f in regrouped] == [f.function_id for f in functions]


def test_unrecognized_language_rows_are_skipped():
    """BigVul is documented as C/C++-only; a stray Python row (real data can
    contain a handful of these from unrelated commit changes) must be
    dropped rather than mis-parsed as C++."""
    functions = load_bigvul_csv(FIXTURE_PATH)

    assert all(f.language in {"C", "CPP", "C++"} for f in functions)
    assert not any(f.commit_id == "mno345" for f in functions)


def test_loaded_functions_are_compatible_with_the_leak_free_splitter():
    """Integration check: BigVulFunction must be usable directly wherever
    the splitter expects objects with .function_id, without a conversion
    step."""
    functions = load_bigvul_csv(FIXTURE_PATH)

    train, val, test = group_aware_split(functions, train_frac=0.5, val_frac=0.5, seed=1)
    assert_no_leakage(train, val, test)  # must not raise
    assert len(train) + len(val) + len(test) == len(functions)
