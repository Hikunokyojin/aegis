"""Verifies the leak-free splitter both passes clean data and rejects
deliberately leaked data (positive + negative test, per project convention)."""
import pytest

from aegis.dataset.leak_free_split import (
    LeakageError,
    Sample,
    assert_no_leakage,
    group_aware_split,
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


def _skewed_dataset() -> list[Sample]:
    """20 projects of 10 functions each, but only 6 projects contain any
    vulnerable functions — mirroring real BigVul, where a plain project
    shuffle put all 5 vulnerable functions of a 100-function sample in test."""
    samples, sid = [], 0
    for p in range(20):
        for i in range(10):
            label = 1 if (p < 6 and i < 2) else 0
            samples.append(Sample(sample_id=sid, function_id=f"proj_{p}", label=label))
            sid += 1
    return samples


def test_every_split_gets_vulnerable_functions_when_possible():
    for seed in range(10):
        train, val, test = group_aware_split(_skewed_dataset(), seed=seed)

        for split in (train, val, test):
            assert sum(s.label for s in split) > 0, f"seed {seed}: a split has no vulnerable functions"


def test_split_sizes_track_the_requested_fractions():
    samples = _skewed_dataset()

    train, val, test = group_aware_split(samples, train_frac=0.7, val_frac=0.15)

    assert abs(len(train) / len(samples) - 0.7) <= 0.1
    assert abs(len(val) / len(samples) - 0.15) <= 0.1
    assert abs(len(test) / len(samples) - 0.15) <= 0.1


def test_split_is_deterministic_for_a_given_seed():
    first = group_aware_split(_skewed_dataset(), seed=3)
    second = group_aware_split(_skewed_dataset(), seed=3)

    assert [[s.sample_id for s in part] for part in first] == [[s.sample_id for s in part] for part in second]


def test_balanced_split_is_still_leak_free():
    train, val, test = group_aware_split(_skewed_dataset(), seed=0)

    assert_no_leakage(train, val, test)


def test_leaky_split_is_detected():
    train = [Sample(sample_id=1, function_id="shared_func", label=1)]
    val = [Sample(sample_id=2, function_id="other_func", label=0)]
    test = [Sample(sample_id=3, function_id="shared_func", label=1)]

    with pytest.raises(LeakageError, match="shared_func"):
        assert_no_leakage(train, val, test)
