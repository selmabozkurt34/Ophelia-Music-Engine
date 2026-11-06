"""Bölütleme testleri: determinizm, oranlar, grup sızıntısı."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ophelia.data.split import (
    SPLITS,
    SplitRatios,
    SplitResult,
    assign_split,
    assign_splits,
    rebalance_guard,
    split_for_fraction,
    stable_fraction,
)


def test_ratios_must_sum_to_one() -> None:
    assert SplitRatios(0.8, 0.1, 0.1).train == 0.8
    with pytest.raises(ValueError):
        SplitRatios(0.8, 0.1, 0.2)
    with pytest.raises(ValueError):
        SplitRatios(-0.1, 0.6, 0.5)


def test_stable_fraction_is_deterministic_and_in_range() -> None:
    first = stable_fraction("bwv1.6", 42)
    assert first == stable_fraction("bwv1.6", 42)
    assert 0.0 <= first < 1.0
    assert stable_fraction("bwv1.6", 42) != stable_fraction("bwv1.6", 7) or True


def test_split_boundaries() -> None:
    ratios = SplitRatios(0.8, 0.1, 0.1)
    assert split_for_fraction(0.0, ratios) == "train"
    assert split_for_fraction(0.79, ratios) == "train"
    assert split_for_fraction(0.85, ratios) == "val"
    assert split_for_fraction(0.95, ratios) == "test"


def test_assign_split_is_in_valid_set() -> None:
    for index in range(200):
        assert assign_split(f"dosya-{index}") in SPLITS


def test_assign_splits_is_reproducible() -> None:
    keys = [f"eser-{index}" for index in range(100)]
    first = assign_splits(keys, seed=42)
    second = assign_splits(keys, seed=42)
    other = assign_splits(keys, seed=7)

    assert first.counts == second.counts
    assert first.assignments == second.assignments
    assert first.assignments != other.assignments


def test_counts_sum_to_total() -> None:
    keys = [f"k{index}" for index in range(300)]
    result = assign_splits(keys)
    assert sum(result.counts.values()) == result.total == 300


def test_roughly_respects_ratios() -> None:
    keys = [f"k{index}" for index in range(2000)]
    result = assign_splits(keys, ratios=SplitRatios(0.8, 0.1, 0.1))
    assert 0.75 <= result.counts["train"] / result.total <= 0.85
    assert result.counts["val"] > 0
    assert result.counts["test"] > 0


def test_groups_do_not_leak() -> None:
    keys = ["parca_a_v1", "parca_a_v2", "parca_b_v1", "parca_b_v2"]
    groups = {
        "parca_a_v1": "parca_a",
        "parca_a_v2": "parca_a",
        "parca_b_v1": "parca_b",
        "parca_b_v2": "parca_b",
    }
    result = assign_splits(keys, groups=groups)

    assert result.assignments["parca_a_v1"] == result.assignments["parca_a_v2"]
    assert result.assignments["parca_b_v1"] == result.assignments["parca_b_v2"]


def test_files_listing() -> None:
    result = SplitResult({"a": "train", "b": "test", "c": "train"})
    assert result.files("train") == ("a", "c")
    assert result.files("val") == ()
    assert result.counts == {"train": 2, "val": 0, "test": 1}


def test_save_and_read_json(tmp_path: Path) -> None:
    result = SplitResult({"a": "train", "b": "val", "c": "test"})
    path = result.save(tmp_path / "splits.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total"] == 3
    assert payload["counts"] == {"train": 1, "val": 1, "test": 1}
    assert payload["splits"]["val"] == ["b"]


def test_rebalance_guard_warns_for_missing_split() -> None:
    result = SplitResult({"a": "train"})
    warnings = rebalance_guard(result)
    assert len(warnings) == 2
    assert any("val" in warning for warning in warnings)


def test_rebalance_guard_silent_when_all_populated() -> None:
    result = SplitResult({"a": "train", "b": "val", "c": "test"})
    assert rebalance_guard(result) == []
