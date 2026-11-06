"""Kalite filtreleri: yetersiz veya bozuk MIDI dosyalarını eler.

Her ret, makine tarafından okunabilir bir **sebep kodu** ile kaydedilir; böylece
veri seti raporu "neden bu kadar dosya elendi" sorusunu yanıtlar.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ophelia.data.midi_io import MidiError, MidiStats, analyze

REASON_TEXT: dict[str, str] = {
    "unreadable": "dosya okunamadı",
    "empty": "hiç nota yok",
    "too_few_notes": "nota sayısı yetersiz",
    "too_short": "çok kısa",
    "too_long": "çok uzun",
    "too_polyphonic": "polifoni sınırı aşıldı",
    "too_few_pitches": "perde çeşitliliği yetersiz",
    "low_resolution": "çözünürlük (PPQ) çok düşük",
    "too_many_tracks": "parça sayısı çok yüksek",
    "percussion": "perküsyon içeriyor",
    "tempo_out_of_range": "tempo aralık dışında",
}


@dataclass(frozen=True)
class CleanConfig:
    """Veri setine kabul eşikleri."""

    min_notes: int = 32
    min_duration_s: float = 5.0
    max_duration_s: float = 900.0
    max_polyphony: int = 12
    min_unique_pitches: int = 4
    min_ticks_per_beat: int = 96
    max_tracks: int = 32
    allow_percussion: bool = False
    tempo_min_bpm: float = 20.0
    tempo_max_bpm: float = 300.0


@dataclass(frozen=True)
class CleanResult:
    """Tek dosyanın değerlendirme sonucu."""

    path: Path
    ok: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    stats: MidiStats | None = None

    @property
    def reason_text(self) -> str:
        return ", ".join(REASON_TEXT.get(code, code) for code in self.reasons)


@dataclass(frozen=True)
class CleanSummary:
    accepted: tuple[Path, ...]
    rejected: tuple[CleanResult, ...]
    reason_counts: Counter

    @property
    def total(self) -> int:
        return len(self.accepted) + len(self.rejected)

    @property
    def acceptance_rate(self) -> float:
        return len(self.accepted) / self.total if self.total else 0.0


def _tempo_bpms(stats: MidiStats) -> list[float]:
    return [60_000_000 / tempo for tempo in stats.tempos if tempo > 0]


def check_stats(stats: MidiStats, config: CleanConfig | None = None) -> tuple[str, ...]:
    """Analiz sonucundan ret sebeplerini üretir (boş demet = kabul)."""
    cfg = config or CleanConfig()
    reasons: list[str] = []

    if stats.is_empty:
        return ("empty",)

    if stats.n_notes < cfg.min_notes:
        reasons.append("too_few_notes")
    if stats.duration_seconds < cfg.min_duration_s:
        reasons.append("too_short")
    if stats.duration_seconds > cfg.max_duration_s:
        reasons.append("too_long")
    if stats.max_polyphony > cfg.max_polyphony:
        reasons.append("too_polyphonic")
    if stats.unique_pitches < cfg.min_unique_pitches:
        reasons.append("too_few_pitches")
    if stats.ticks_per_beat < cfg.min_ticks_per_beat:
        reasons.append("low_resolution")
    if stats.n_tracks > cfg.max_tracks:
        reasons.append("too_many_tracks")
    if stats.has_percussion and not cfg.allow_percussion:
        reasons.append("percussion")

    bpms = _tempo_bpms(stats)
    if bpms and (min(bpms) < cfg.tempo_min_bpm or max(bpms) > cfg.tempo_max_bpm):
        reasons.append("tempo_out_of_range")

    return tuple(reasons)


def clean_file(path: str | Path, config: CleanConfig | None = None) -> CleanResult:
    """Tek dosyayı değerlendirir."""
    cfg = config or CleanConfig()
    file_path = Path(path)

    try:
        stats = analyze(file_path)
    except MidiError:
        return CleanResult(file_path, False, ("unreadable",))

    reasons = check_stats(stats, cfg)
    return CleanResult(file_path, not reasons, reasons, stats)


def clean_corpus(
    paths: list[Path] | tuple[Path, ...],
    config: CleanConfig | None = None,
) -> CleanSummary:
    """Dosya listesini değerlendirir ve sebep dağılımını çıkarır."""
    cfg = config or CleanConfig()
    accepted: list[Path] = []
    rejected: list[CleanResult] = []
    counts: Counter = Counter()

    for path in paths:
        result = clean_file(path, cfg)
        if result.ok:
            accepted.append(Path(path))
        else:
            rejected.append(result)
            counts.update(result.reasons)

    return CleanSummary(tuple(accepted), tuple(rejected), counts)
