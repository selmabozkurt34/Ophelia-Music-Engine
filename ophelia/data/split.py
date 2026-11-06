"""train / val / test bölütleme.

Bölütleme **deterministik** ve **içerik tabanlıdır**: dosyanın sha256 özeti ve
tohum birlikte hash'lenir, çıkan sayı [0, 1) aralığına ölçeklenir. Böylece

* aynı veri seti her çalıştırmada aynı bölünmeyi üretir,
* yeni dosya eklemek mevcut dosyaların bölümünü değiştirmez,
* aynı eserin kopyaları ``group`` anahtarıyla aynı bölüme düşer (sızıntı yok).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

TRAIN = "train"
VAL = "val"
TEST = "test"
SPLITS = (TRAIN, VAL, TEST)


@dataclass(frozen=True)
class SplitRatios:
    """Bölüm oranları; toplamı 1.0 olmalı."""

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        for name, value in (("train", self.train), ("val", self.val), ("test", self.test)):
            if value < 0:
                raise ValueError(f"{name} oranı negatif olamaz: {value}")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"oranların toplamı 1.0 olmalı: {total}")

    @property
    def val_boundary(self) -> float:
        """train | val sınırı."""
        return self.train

    @property
    def test_boundary(self) -> float:
        """val | test sınırı."""
        return self.train + self.val


@dataclass(frozen=True)
class SplitResult:
    """Bölütleme çıktısı: bölüm adı -> dosya anahtarları."""

    assignments: dict[str, str] = field(default_factory=dict)

    def files(self, split: str) -> tuple[str, ...]:
        return tuple(sorted(key for key, value in self.assignments.items() if value == split))

    @property
    def counts(self) -> dict[str, int]:
        return {split: len(self.files(split)) for split in SPLITS}

    @property
    def total(self) -> int:
        return len(self.assignments)

    def to_dict(self) -> dict[str, object]:
        return {
            "counts": self.counts,
            "total": self.total,
            "splits": {split: list(self.files(split)) for split in SPLITS},
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target


def stable_fraction(key: str, seed: int = 42) -> float:
    """``key`` + ``seed`` için [0, 1) aralığında kararlı bir sayı."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_for_fraction(fraction: float, ratios: SplitRatios) -> str:
    if fraction < ratios.val_boundary:
        return TRAIN
    if fraction < ratios.test_boundary:
        return VAL
    return TEST


def assign_split(key: str, ratios: SplitRatios | None = None, seed: int = 42) -> str:
    """Tek bir anahtarın bölümünü belirler."""
    return split_for_fraction(stable_fraction(key, seed), ratios or SplitRatios())


def assign_splits(
    keys: list[str] | tuple[str, ...],
    *,
    ratios: SplitRatios | None = None,
    seed: int = 42,
    groups: dict[str, str] | None = None,
) -> SplitResult:
    """Anahtar listesini bölümlere dağıtır.

    ``groups`` verilirse aynı gruba ait tüm anahtarlar tek bir bölüme düşer
    (aynı eserin farklı kaynaklardaki kopyaları için sızıntıyı önler).
    """
    cfg = ratios or SplitRatios()
    assignments: dict[str, str] = {}
    group_cache: dict[str, str] = {}

    for key in keys:
        group = (groups or {}).get(key, key)
        if group not in group_cache:
            group_cache[group] = assign_split(group, cfg, seed)
        assignments[key] = group_cache[group]

    return SplitResult(assignments)


def rebalance_guard(result: SplitResult, minimum: int = 1) -> list[str]:
    """Bölümlerden biri ``minimum`` dosyadan azsa uyarı listesi döndürür."""
    warnings: list[str] = []
    for split, count in result.counts.items():
        if count < minimum:
            warnings.append(
                f"{split} bölümü çok küçük ({count} dosya); oranları veya veri boyutunu gözden geçirin"
            )
    return warnings
