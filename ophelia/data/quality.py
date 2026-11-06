"""Veri seti kalite raporu.

Rapor, 5. aşamadaki model eğitiminin girdi dağılımını görünür kılar: kaç dosya,
ne kadar süre, hangi perde/tonalite/polifoni aralığında. Tonalite dağılımı
2. aşamadaki ``estimate_key`` ile hesaplanır.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ophelia.data.midi_io import MidiError, analyze, iter_notes, load_midi
from ophelia.theory.scales import estimate_key


@dataclass
class DatasetReport:
    """Bir dosya kümesinin özet istatistikleri."""

    n_files: int = 0
    n_failed: int = 0
    total_seconds: float = 0.0
    total_notes: int = 0
    durations: list[float] = field(default_factory=list)
    note_counts: list[int] = field(default_factory=list)
    polyphony: list[int] = field(default_factory=list)
    pitch_histogram: Counter = field(default_factory=Counter)
    pitch_class_histogram: Counter = field(default_factory=Counter)
    key_distribution: Counter = field(default_factory=Counter)
    time_signatures: Counter = field(default_factory=Counter)
    sources: Counter = field(default_factory=Counter)
    failed_files: list[str] = field(default_factory=list)

    # --- türetilmiş ölçütler ------------------------------------------------
    @property
    def total_hours(self) -> float:
        return self.total_seconds / 3600.0

    @property
    def mean_duration(self) -> float:
        return statistics.fmean(self.durations) if self.durations else 0.0

    @property
    def median_duration(self) -> float:
        return statistics.median(self.durations) if self.durations else 0.0

    @property
    def mean_notes(self) -> float:
        return statistics.fmean(self.note_counts) if self.note_counts else 0.0

    @property
    def mean_polyphony(self) -> float:
        return statistics.fmean(self.polyphony) if self.polyphony else 0.0

    @property
    def notes_per_second(self) -> float:
        return self.total_notes / self.total_seconds if self.total_seconds else 0.0

    def top_keys(self, limit: int = 8) -> list[tuple[str, int]]:
        return self.key_distribution.most_common(limit)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_files": self.n_files,
            "n_failed": self.n_failed,
            "total_seconds": round(self.total_seconds, 1),
            "total_hours": round(self.total_hours, 3),
            "total_notes": self.total_notes,
            "mean_duration_s": round(self.mean_duration, 1),
            "median_duration_s": round(self.median_duration, 1),
            "mean_notes_per_file": round(self.mean_notes, 1),
            "notes_per_second": round(self.notes_per_second, 2),
            "mean_polyphony": round(self.mean_polyphony, 2),
            "top_keys": self.top_keys(),
            "time_signatures": dict(self.time_signatures),
            "sources": dict(self.sources),
            "failed_files": self.failed_files,
        }


def build_report(
    paths: list[Path] | tuple[Path, ...],
    *,
    sources: dict[str, str] | None = None,
    key_limit_files: int = 200,
) -> DatasetReport:
    """Dosya listesini analiz edip raporu doldurur.

    Tonalite tahmini ilk ``key_limit_files`` dosya üzerinde yapılır (rapor
    süresini sınırlamak için); perde istatistikleri tüm dosyaları kapsar.
    """
    report = DatasetReport()
    source_map = sources or {}

    for path in paths:
        file_path = Path(path)
        try:
            midi = load_midi(file_path)
            stats = analyze(file_path)
            notes = iter_notes(midi)
        except MidiError:
            report.n_failed += 1
            report.failed_files.append(file_path.name)
            continue

        report.n_files += 1
        report.total_seconds += stats.duration_seconds
        report.total_notes += stats.n_notes
        report.durations.append(stats.duration_seconds)
        report.note_counts.append(stats.n_notes)
        report.polyphony.append(stats.max_polyphony)
        for signature in stats.time_signatures:
            report.time_signatures[signature] += 1

        for note in notes:
            report.pitch_histogram[note.pitch] += 1
            report.pitch_class_histogram[note.pitch % 12] += 1

        source = source_map.get(file_path.name)
        if source:
            report.sources[source] += 1

    # Tonalite dağılımı: dosya başına tahmin (ilk N dosya)
    for path in list(paths)[:key_limit_files]:
        try:
            notes = iter_notes(load_midi(path))
        except MidiError:
            continue
        if not notes:
            continue
        histogram: Counter = Counter(note.pitch % 12 for note in notes)
        key = estimate_key(dict(histogram))
        report.key_distribution[key.name] += 1

    return report


def format_report(report: DatasetReport) -> list[str]:
    """Raporu düz metin satırlarına çevirir (CLI ve doküman için)."""
    lines = [
        f"Dosya sayısı      : {report.n_files}"
        + (f" (okunamayan: {report.n_failed})" if report.n_failed else ""),
        f"Toplam süre       : {report.total_seconds / 60:.1f} dk ({report.total_hours:.2f} saat)",
        f"Toplam nota       : {report.total_notes}",
        f"Ortalama dosya    : {report.mean_duration:.1f} sn, {report.mean_notes:.0f} nota",
        f"Nota yoğunluğu    : {report.notes_per_second:.2f} nota/sn",
        f"Ortalama polifoni : {report.mean_polyphony:.2f}",
        f"Ölçüler           : {dict(report.time_signatures)}",
    ]
    if report.key_distribution:
        top = ", ".join(f"{name} ({count})" for name, count in report.top_keys())
        lines.append(f"Tonalite (ilk N)  : {top}")
    if report.sources:
        lines.append(f"Kaynaklar         : {dict(report.sources)}")
    if report.failed_files:
        lines.append(f"Okunamayanlar     : {', '.join(report.failed_files[:5])}")
    return lines
