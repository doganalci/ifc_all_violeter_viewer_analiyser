"""Train/val/test splits stratified by `baseline_id`.

All violated IFCs derived from the same baseline must end up in the
same split — otherwise the model sees near-duplicates of its training
graphs at evaluation time and metrics overstate generalisation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]


def split_by_baseline(
    baseline_ids: list[str | None],
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> SplitIndices:
    """Group indices by `baseline_id` and partition the *groups*.

    Items with `baseline_id is None` (e.g. orphan baselines mixed in)
    are each treated as their own singleton group.
    """
    assert 0.0 <= val_frac < 1.0
    assert 0.0 <= test_frac < 1.0
    assert val_frac + test_frac < 1.0

    groups: dict[str, list[int]] = {}
    for i, b in enumerate(baseline_ids):
        key = b if b is not None else f"__solo__{i}"
        groups.setdefault(key, []).append(i)

    keys = sorted(groups)  # determinism before shuffle
    rng = random.Random(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_test = max(1, int(round(n * test_frac))) if n > 2 else 0
    n_val = max(1, int(round(n * val_frac))) if n - n_test > 2 else 0
    test_keys = set(keys[:n_test])
    val_keys = set(keys[n_test : n_test + n_val])

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for k, ids in groups.items():
        if k in test_keys:
            test.extend(ids)
        elif k in val_keys:
            val.extend(ids)
        else:
            train.extend(ids)
    return SplitIndices(sorted(train), sorted(val), sorted(test))
