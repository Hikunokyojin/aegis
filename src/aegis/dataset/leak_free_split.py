"""Leak-free, group-aware train/validation/test splitting.

Groups samples by function_id before splitting, so every version of the same
function (e.g. a pre-fix and post-fix revision) lands in a single split —
closing the most common data-leakage vector in vulnerability-detection
datasets, where near-duplicate functions otherwise end up on both sides of
a naive random split.
"""
import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Sample:
    sample_id: int
    function_id: str
    label: int


class LeakageError(Exception):
    """Raised when a split has any function_id present in more than one of
    train/val/test."""


def group_aware_split(samples: list[Sample], train_frac: float = 0.7, val_frac: float = 0.15,
                       seed: int = 42) -> tuple[list[Sample], list[Sample], list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[s.function_id].append(s)

    group_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_train = int(n_groups * train_frac)
    n_val = int(n_groups * val_frac)

    train_group_ids = set(group_ids[:n_train])
    val_group_ids = set(group_ids[n_train:n_train + n_val])
    test_group_ids = set(group_ids[n_train + n_val:])

    train = [s for gid in train_group_ids for s in groups[gid]]
    val = [s for gid in val_group_ids for s in groups[gid]]
    test = [s for gid in test_group_ids for s in groups[gid]]
    return train, val, test


def assert_no_leakage(train: list[Sample], val: list[Sample], test: list[Sample]) -> None:
    train_ids = {s.function_id for s in train}
    val_ids = {s.function_id for s in val}
    test_ids = {s.function_id for s in test}
    overlaps = {
        "train/val": train_ids & val_ids,
        "train/test": train_ids & test_ids,
        "val/test": val_ids & test_ids,
    }
    leaking = {pair: ids for pair, ids in overlaps.items() if ids}
    if leaking:
        details = "; ".join(f"{pair}: {sorted(ids)}" for pair, ids in leaking.items())
        raise LeakageError(f"Data leakage detected across splits! Overlapping function_ids -> {details}")
