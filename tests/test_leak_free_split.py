"""Verifies the leak-free splitter both passes clean data and rejects
deliberately leaked data (positive + negative test, per project convention)."""
import pytest

from aegis.dataset.leak_free_split import (
    Sample,
    LeakageError,
    group_aware_split,
    assert_no_leakage,
)


def _clean_synthetic_dataset(n_functions: int = 50, versions_per_function: int = 3) -> list[Sample]:
    samples = []
    sid = 0
    for f in range(n_functions):
        function_id = f"func_{f}"
        label = f % 2
        for _ in range(versions_per_function):
            samples.append(Sample(sample_id=sid, function_id=function_id, label=label))
            sid += 1
    return samples


def test_clean_split_has_no_leakage():
    samples = _clean_synthetic_dataset()
    train, val, test = group_aware_split(samples)

    assert len(train) + len(val) + len(test) == len(samples)
    assert_no_leakage(train, val, test)  # must not raise


def test_leaky_split_is_detected():
    train = [Sample(sample_id=1, function_id="shared_func", label=1)]
    val = [Sample(sample_id=2, function_id="other_func", label=0)]
    test = [Sample(sample_id=3, function_id="shared_func", label=1)]

    with pytest.raises(LeakageError, match="shared_func"):
        assert_no_leakage(train, val, test)
