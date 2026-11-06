"""Motif, cümle ve form şablonları.

Bir *motif* perde aralıkları + sürelerden oluşur; melodi üretiminde (5. aşama)
bu küçük birimler dönüştürülerek uzun kompozisyonlar kurulur.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ophelia.theory.rhythm import TimedNote


@dataclass(frozen=True)
class Motif:
    """Kısa müzikal fikir: ilk notaya göre aralıklar ve süreler."""

    intervals: tuple[int, ...]
    durations: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.intervals:
            raise ValueError("motif en az bir nota içermeli")
        if len(self.intervals) != len(self.durations):
            raise ValueError("aralık ve süre sayıları eşit olmalı")
        if any(duration <= 0 for duration in self.durations):
            raise ValueError("süreler pozitif olmalı")

    # --- yapı ---------------------------------------------------------------
    @property
    def length(self) -> int:
        return len(self.intervals)

    @property
    def span(self) -> float:
        """Motifin toplam süresi (çeyrek vuruş)."""
        return sum(self.durations)

    @property
    def contour(self) -> str:
        """Yön özeti: ``+`` çıkıcı, ``-`` inici, ``0`` yatay (ardışık adımlar)."""
        marks = []
        for previous, current in zip(self.intervals, self.intervals[1:], strict=False):
            delta = current - previous
            marks.append("+" if delta > 0 else ("-" if delta < 0 else "0"))
        return "".join(marks)

    # --- dönüşümler ---------------------------------------------------------
    def transposed(self, semitones: int) -> Motif:
        return Motif(tuple(interval + semitones for interval in self.intervals), self.durations)

    def inverted(self) -> Motif:
        """Yönü tersine çevirir (aynı ritim)."""
        return Motif(tuple(-interval for interval in self.intervals), self.durations)

    def retrograde(self) -> Motif:
        """Zamanı tersine çevirir."""
        return Motif(tuple(reversed(self.intervals)), tuple(reversed(self.durations)))

    def retrograde_inversion(self) -> Motif:
        return self.retrograde().inverted()

    def augmented(self, factor: float = 2.0) -> Motif:
        """Süreleri uzatır (motifin genişletilmesi)."""
        if factor <= 0:
            raise ValueError("çarpan pozitif olmalı")
        return Motif(self.intervals, tuple(duration * factor for duration in self.durations))

    def diminished(self, factor: float = 2.0) -> Motif:
        """Süreleri kısaltır."""
        return self.augmented(1.0 / factor)

    def sequence(self, start_pitch: int, steps: int = 2, semitones: int = 2) -> tuple[int, ...]:
        """Motifi art arda taşıyarak bir sekans üretir."""
        if steps <= 0:
            raise ValueError("adım sayısı pozitif olmalı")
        pitches: list[int] = []
        for step in range(steps):
            offset = step * semitones
            pitches.extend(start_pitch + interval + offset for interval in self.intervals)
        return tuple(pitches)

    # --- çıktı --------------------------------------------------------------
    def midi(self, start_pitch: int = 60) -> tuple[int, ...]:
        return tuple(start_pitch + interval for interval in self.intervals)

    def notes(
        self,
        start_pitch: int = 60,
        start_beat: float = 0.0,
        velocity: int = 90,
        *,
        repeat: int = 1,
    ) -> tuple[TimedNote, ...]:
        """Motifi zaman çizelgesine yerleştirir; ``repeat`` ile tekrarlar."""
        if repeat <= 0:
            raise ValueError("tekrar sayısı pozitif olmalı")

        events: list[TimedNote] = []
        cursor = start_beat
        for _ in range(repeat):
            for interval, duration in zip(self.intervals, self.durations, strict=True):
                events.append(TimedNote(cursor, duration, start_pitch + interval, velocity))
                cursor += duration
        return tuple(events)

    @classmethod
    def from_midi(
        cls,
        pitches: Sequence[int],
        durations: Sequence[float],
    ) -> Motif:
        """MIDI perde dizisinden motif kurar (ilk nota referans alınır)."""
        if not pitches:
            raise ValueError("perde dizisi boş olamaz")
        reference = pitches[0]
        return cls(tuple(pitch - reference for pitch in pitches), tuple(durations))


@dataclass(frozen=True)
class Section:
    """Form içindeki bölüm."""

    label: str
    bars: int

    def __post_init__(self) -> None:
        if self.bars <= 0:
            raise ValueError("bölüm uzunluğu pozitif olmalı")


# Hazır form şablonları: ad -> (bölüm etiketi, ölçü sayısı) dizisi
FORM_TEMPLATES: dict[str, tuple[tuple[str, int], ...]] = {
    "AABA": (("A", 8), ("A", 8), ("B", 8), ("A", 8)),
    "ABAB": (("A", 8), ("B", 8), ("A", 8), ("B", 8)),
    "ABAC": (("A", 8), ("B", 8), ("A", 8), ("C", 8)),
    "through_composed": (("A", 8), ("B", 8), ("C", 8), ("D", 8)),
    "verse_chorus": (
        ("intro", 4),
        ("verse", 8),
        ("chorus", 8),
        ("verse", 8),
        ("chorus", 8),
        ("outro", 4),
    ),
    "pop_standard": (
        ("intro", 4),
        ("verse", 8),
        ("chorus", 8),
        ("verse", 8),
        ("chorus", 8),
        ("bridge", 8),
        ("chorus", 8),
        ("outro", 4),
    ),
    "twelve_bar_blues": (("A", 4), ("A", 4), ("B", 4)),
    "sonata": (
        ("exposition", 16),
        ("development", 16),
        ("recapitulation", 16),
        ("coda", 8),
    ),
    "ambient_loop": (("intro", 8), ("A", 16), ("B", 16), ("A", 16), ("outro", 8)),
}

FORM_ALIASES: dict[str, str] = {
    "pop": "verse_chorus",
    "song": "pop_standard",
    "blues": "twelve_bar_blues",
    "ambient": "ambient_loop",
    "aaa": "AABA",
}

TRANSFORMS = ("transpose", "invert", "retrograde", "retrograde_inversion", "augment", "diminish")


@dataclass(frozen=True)
class Form:
    """Bölümlerden oluşan eser iskeleti."""

    name: str
    sections: tuple[Section, ...]

    def __post_init__(self) -> None:
        if not self.sections:
            raise ValueError("form en az bir bölüm içermeli")

    @property
    def total_bars(self) -> int:
        return sum(section.bars for section in self.sections)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(section.label for section in self.sections)

    @property
    def unique_labels(self) -> tuple[str, ...]:
        seen: list[str] = []
        for label in self.labels:
            if label not in seen:
                seen.append(label)
        return tuple(seen)

    @property
    def offsets(self) -> tuple[int, ...]:
        """Her bölümün başladığı ölçü (0 tabanlı)."""
        positions: list[int] = []
        cursor = 0
        for section in self.sections:
            positions.append(cursor)
            cursor += section.bars
        return tuple(positions)

    def section_at(self, bar: int) -> Section:
        """Verilen ölçünün hangi bölüme düştüğünü söyler."""
        if not 0 <= bar < self.total_bars:
            raise ValueError(f"ölçü aralık dışında: {bar}")
        for offset, section in zip(self.offsets, self.sections, strict=True):
            if offset <= bar < offset + section.bars:
                return section
        raise ValueError(f"ölçü eşleşmedi: {bar}")  # pragma: no cover

    def repetition_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "total_bars": self.total_bars,
            "sections": [{"label": s.label, "bars": s.bars} for s in self.sections],
        }

    def describe(self) -> str:
        parts = [f"{section.label}:{section.bars}" for section in self.sections]
        return f"{self.name} ({self.total_bars} ölçü) — " + " | ".join(parts)


_FORM_INDEX: dict[str, str] = {name.lower(): name for name in FORM_TEMPLATES}


def resolve_form_name(name: str) -> str:
    """Form adını/takma adını ``FORM_TEMPLATES`` anahtarına çevirir (harf duyarsız)."""
    key = name.strip()
    if key in FORM_TEMPLATES:
        return key

    lowered = key.lower()
    lowered = FORM_ALIASES.get(lowered, lowered)
    if lowered in _FORM_INDEX:
        return _FORM_INDEX[lowered]

    available = ", ".join(sorted(FORM_TEMPLATES))
    raise ValueError(f"bilinmeyen form: {name!r} (mevcut: {available})")


def build_form(name: str) -> Form:
    """Hazır şablondan form üretir."""
    resolved = resolve_form_name(name)
    sections = tuple(Section(label, bars) for label, bars in FORM_TEMPLATES[resolved])
    return Form(resolved, sections)


def transform(motif: Motif, operation: str, **kwargs: float | int) -> Motif:
    """Motifi adlandırılmış bir işlemle dönüştürür."""
    operation = operation.strip().lower()
    mapping = {
        "transpose": lambda m: m.transposed(int(kwargs.get("semitones", 2))),
        "invert": lambda m: m.inverted(),
        "retrograde": lambda m: m.retrograde(),
        "retrograde_inversion": lambda m: m.retrograde_inversion(),
        "augment": lambda m: m.augmented(float(kwargs.get("factor", 2.0))),
        "diminish": lambda m: m.diminished(float(kwargs.get("factor", 2.0))),
    }
    if operation not in mapping:
        available = ", ".join(TRANSFORMS)
        raise ValueError(f"bilinmeyen motif işlemi: {operation!r} (mevcut: {available})")
    return mapping[operation](motif)
