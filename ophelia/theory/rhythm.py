"""Tempo, ölçü, ritim ve senkop.

Tüm süreler **çeyrek vuruş (quarter note)** birimindedir: 1.0 = bir çeyrek nota,
0.5 = sekizlik, 2.0 = ikilik, 4.0 = birlik. Böylece ölçü, tempo ve quantize
hesapları tek bir birim üzerinde yapılır.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

WHOLE = 4.0
HALF = 2.0
QUARTER = 1.0
EIGHTH = 0.5
SIXTEENTH = 0.25
THIRTY_SECOND = 0.125

VALID_DENOMINATORS = (1, 2, 4, 8, 16, 32)
_EPSILON = 1e-9


def dotted(value: float, dots: int = 1) -> float:
    """Noktalı süre: her nokta sürenin yarısını ekler."""
    total = value
    addition = value
    for _ in range(dots):
        addition /= 2.0
        total += addition
    return total


def triplet(value: float) -> float:
    """Üçleme: süreyi 2/3 oranında kısaltır."""
    return value * 2.0 / 3.0


def bpm_to_seconds_per_quarter(bpm: float) -> float:
    """BPM (çeyrek nota) -> çeyrek nota süresi (saniye)."""
    if bpm <= 0:
        raise ValueError("BPM pozitif olmalı")
    return 60.0 / bpm


def beats_to_seconds(beats: float, bpm: float) -> float:
    """Çeyrek vuruş cinsinden süreyi saniyeye çevirir."""
    return beats * bpm_to_seconds_per_quarter(bpm)


def seconds_to_beats(seconds: float, bpm: float) -> float:
    return seconds / bpm_to_seconds_per_quarter(bpm)


@dataclass(frozen=True)
class TimeSignature:
    """Ölçü: ``4/4``, ``3/4``, ``6/8``, ``7/8``..."""

    numerator: int
    denominator: int = 4

    def __post_init__(self) -> None:
        if self.numerator <= 0:
            raise ValueError("ölçü payı pozitif olmalı")
        if self.denominator not in VALID_DENOMINATORS:
            raise ValueError(f"geçersiz ölçü paydası: {self.denominator}")

    @property
    def beats_per_bar(self) -> int:
        """Ölçüdeki vuruş sayısı (payda birimi cinsinden)."""
        return self.numerator

    @property
    def beat_unit_quarters(self) -> float:
        """Bir vuruşun çeyrek nota cinsinden değeri."""
        return 4.0 / self.denominator

    @property
    def quarter_beats(self) -> float:
        """Bir ölçünün çeyrek nota cinsinden uzunluğu."""
        return self.numerator * self.beat_unit_quarters

    @property
    def is_compound(self) -> bool:
        """Bileşik ölçü mü (6/8, 9/8, 12/8 gibi)?"""
        return self.denominator == 8 and self.numerator % 3 == 0 and self.numerator > 3

    @property
    def is_simple(self) -> bool:
        return not self.is_compound

    def bar_seconds(self, bpm: float) -> float:
        return beats_to_seconds(self.quarter_beats, bpm)

    def bar_of(self, beat: float) -> int:
        """Vuruş konumunun ölçü indeksi (0 tabanlı)."""
        return int(math.floor(beat / self.quarter_beats))

    def beat_in_bar(self, beat: float) -> float:
        """Vuruşun ölçü içindeki konumu (çeyrek birim)."""
        return beat % self.quarter_beats

    def subdivisions(self, grid: float = SIXTEENTH) -> tuple[float, ...]:
        """Tek ölçüdeki alt bölünme başlangıçları."""
        if grid <= 0:
            raise ValueError("grid pozitif olmalı")
        count = int(round(self.quarter_beats / grid))
        return tuple(round(index * grid, 6) for index in range(count))

    def accents(self) -> tuple[float, ...]:
        """Vurgulu vuruş konumları (çeyrek birim)."""
        unit = self.beat_unit_quarters
        return tuple(round(beat * unit, 6) for beat in range(self.numerator))

    @classmethod
    def parse(cls, text: str) -> TimeSignature:
        """``"4/4"`` veya ``"6/8"`` metnini çözer."""
        parts = text.replace(" ", "").split("/")
        if len(parts) != 2:
            raise ValueError(f"geçersiz ölçü: {text!r}")
        return cls(int(parts[0]), int(parts[1]))

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass(frozen=True)
class TimedNote:
    """Ritim açısından tek bir nota olayı."""

    start: float
    duration: float
    pitch: int = 60
    velocity: int = 90

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("başlangıç negatif olamaz")
        if self.duration <= 0:
            raise ValueError("süre pozitif olmalı")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"velocity 0-127 olmalı: {self.velocity}")

    @property
    def end(self) -> float:
        return self.start + self.duration


def quantize(value: float, grid: float = SIXTEENTH) -> float:
    """Tek bir zaman değerini ızgaraya oturtur."""
    if grid <= 0:
        raise ValueError("grid pozitif olmalı")
    return round(round(value / grid) * grid, 6)


def quantize_notes(notes: list[TimedNote], grid: float = SIXTEENTH) -> list[TimedNote]:
    """Notaların başlangıç ve sürelerini ızgaraya oturtur."""
    quantized: list[TimedNote] = []
    for note in notes:
        start = quantize(note.start, grid)
        duration = max(grid, quantize(note.duration, grid))
        quantized.append(TimedNote(start, duration, note.pitch, note.velocity))
    return quantized


def is_on_beat(beat: float, tolerance: float = _EPSILON) -> bool:
    return abs(beat - round(beat)) <= tolerance


def off_beat_onsets(notes: list[TimedNote]) -> tuple[float, ...]:
    """Vuruş dışında başlayan notaların konumları."""
    return tuple(note.start for note in notes if not is_on_beat(note.start))


def is_syncopated(notes: list[TimedNote]) -> bool:
    """Senkop var mı? Zayıf zamanda başlayıp vuruş sınırını aşan nota aranır."""
    for note in notes:
        if is_on_beat(note.start):
            continue
        # Zayıf zamanda başlayan nota bir sonraki vuruşu aşıyorsa senkoptur.
        if note.end > math.floor(note.start) + 1:
            return True
    return False


def swing(notes: list[TimedNote], ratio: float = 2.0 / 3.0) -> list[TimedNote]:
    """Sekizlikleri salınımlı (swing) hâle getirir.

    Vuruşun tam ortasında (x.5) başlayan notalar ``ratio`` oranına kaydırılır.
    """
    if not 0.5 <= ratio <= 1.0:
        raise ValueError("swing oranı 0.5-1.0 arasında olmalı")

    swung: list[TimedNote] = []
    for note in notes:
        position = note.start % 1.0
        if abs(position - 0.5) <= _EPSILON:
            shifted = math.floor(note.start) + ratio
            swung.append(TimedNote(shifted, note.duration, note.pitch, note.velocity))
        else:
            swung.append(note)
    return swung


def total_beats(notes: list[TimedNote]) -> float:
    return max((note.end for note in notes), default=0.0)


def bars_used(notes: list[TimedNote], signature: TimeSignature) -> float:
    """Kullanılan ölçü sayısı (kesirli)."""
    return total_beats(notes) / signature.quarter_beats


def density(notes: list[TimedNote], signature: TimeSignature) -> float:
    """Ölçü başına nota sayısı."""
    bars = bars_used(notes, signature)
    if bars <= 0:
        return 0.0
    return len(notes) / bars


def beat_grid(signature: TimeSignature, bars: int = 1, grid: float = SIXTEENTH) -> tuple[float, ...]:
    """Birden fazla ölçü için ızgara noktaları."""
    if bars <= 0:
        raise ValueError("ölçü sayısı pozitif olmalı")
    single = signature.subdivisions(grid)
    step = signature.quarter_beats
    return tuple(
        round(bar * step + position, 6) for bar in range(bars) for position in single
    )
