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
    """Splits whole groups (never individual samples) into train/val/test so
    that each split gets roughly its share of samples AND of positive
    (vulnerable) samples.

    A plain shuffle of groups left all 5 vulnerable functions of a real
    100-function BigVul sample in test, so nothing could be learned or
    validated. Instead, groups are placed one at a time into the split that
    is currently furthest below its target: groups containing positives are
    placed first (largest first) by how full each split is of positives, the
    rest by how full each split is of samples. Splits with a zero fraction
    never receive anything. Group order is shuffled with `seed` first, so
    ties are broken differently per seed but identically for the same seed.
    """
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[s.function_id].append(s)

    group_ids = list(groups.keys())
    random.Random(seed).shuffle(group_ids)
    positives = {gid: sum(s.label for s in groups[gid]) for gid in group_ids}
    group_ids.sort(key=lambda gid: positives[gid], reverse=True)  # stable: shuffle order breaks ties

    fracs = [train_frac, val_frac, 1.0 - train_frac - val_frac]
    usable = [i for i, frac in enumerate(fracs) if frac > 0]
    total_n, total_pos = len(samples), sum(positives.values())
    parts: list[list[Sample]] = [[], [], []]
    n, pos = [0, 0, 0], [0, 0, 0]

    for gid in group_ids:
        size_fill = lambda i: n[i] / (fracs[i] * total_n)
        if positives[gid] > 0:
            target = min(usable, key=lambda i: (pos[i] / (fracs[i] * total_pos), size_fill(i)))
        else:
            target = min(usable, key=size_fill)
        parts[target].extend(groups[gid])
        n[target] += len(groups[gid])
        pos[target] += positives[gid]

    return parts[0], parts[1], parts[2]


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
