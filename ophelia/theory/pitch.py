"""Nota, oktav, MIDI numarası ve frekans dönüşümleri.

Tüm proje boyunca tek kural geçerlidir: **C4 = MIDI 60 = 261.63 Hz** (bilimsel
perde gösterimi, A4 = 440 Hz). MIDI dışında hiçbir gösterim içsel olarak
kullanılmaz; metin yalnızca giriş/çıkış katmanında ayrıştırılır.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

SHARP_NAMES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)
FLAT_NAMES: tuple[str, ...] = (
    "C",
    "Db",
    "D",
    "Eb",
    "E",
    "F",
    "Gb",
    "G",
    "Ab",
    "A",
    "Bb",
    "B",
)

NATURAL_PITCH_CLASS: dict[str, int] = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

A4_MIDI = 69
A4_FREQ = 440.0
DEFAULT_OCTAVE = 4
SEMITONES_PER_OCTAVE = 12

_NOTE_RE = re.compile(r"^([A-Ga-g])([#b♯♭]{0,2})(-?\d+)?$")

INTERVAL_NAMES: dict[int, str] = {
    0: "P1",
    1: "m2",
    2: "M2",
    3: "m3",
    4: "M3",
    5: "P4",
    6: "TT",
    7: "P5",
    8: "m6",
    9: "M6",
    10: "m7",
    11: "M7",
    12: "P8",
}


def _accidental_value(accidentals: str) -> int:
    """``#``/``b`` dizisini yarım ton toplamına çevirir."""
    total = 0
    for char in accidentals:
        if char in "#♯":
            total += 1
        elif char in "b♭":
            total -= 1
    return total


def parse_note(name: str) -> tuple[int, int | None]:
    """``"C#4"`` -> ``(1, 4)``; oktav yoksa ``None`` döner.

    Dönen ilk değer **mod alınmamış** perdedir: ``B#3`` -> ``(12, 3)``, yani
    oktav taşması korunur ve ``B#3 == C4`` doğru hesaplanır.
    """
    match = _NOTE_RE.match(name.strip())
    if match is None:
        raise ValueError(f"geçersiz nota adı: {name!r}")
    letter, accidentals, octave = match.groups()
    value = NATURAL_PITCH_CLASS[letter.upper()] + _accidental_value(accidentals)
    return value, (int(octave) if octave is not None else None)


def note_to_pitch_class(name: str) -> int:
    """Nota adının perde sınıfını (0-11) döndürür; oktav yok sayılır."""
    return parse_note(name)[0] % 12


def note_to_midi(name: str, octave: int | None = None) -> int:
    """``"C4"`` -> 60. Oktav verilmezse isimden okunur, yoksa 4 varsayılır."""
    value, parsed_octave = parse_note(name)
    if octave is not None:
        resolved = octave
    elif parsed_octave is not None:
        resolved = parsed_octave
    else:
        resolved = DEFAULT_OCTAVE
    return SEMITONES_PER_OCTAVE * (resolved + 1) + value


def midi_to_note(midi: int, *, flats: bool = False) -> str:
    """MIDI numarasını ``"C4"`` biçiminde yazar."""
    if not 0 <= midi <= 127:
        raise ValueError(f"MIDI aralığı dışında: {midi}")
    table = FLAT_NAMES if flats else SHARP_NAMES
    return f"{table[midi % 12]}{midi // SEMITONES_PER_OCTAVE - 1}"


def pitch_class_of(midi: int) -> int:
    """MIDI numarasının perde sınıfı (0 = C)."""
    return midi % SEMITONES_PER_OCTAVE


def octave_of(midi: int) -> int:
    """MIDI numarasının bilimsel oktavı (C4 = 60 -> 4)."""
    return midi // SEMITONES_PER_OCTAVE - 1


def midi_to_freq(midi: int | float) -> float:
    """Eşit tampereman frekansı (Hz)."""
    return A4_FREQ * 2.0 ** ((midi - A4_MIDI) / SEMITONES_PER_OCTAVE)


def freq_to_midi(freq: float) -> float:
    """Frekansı (sürekli) MIDI değerine çevirir."""
    if freq <= 0:
        raise ValueError("frekans pozitif olmalı")
    return A4_MIDI + SEMITONES_PER_OCTAVE * math.log2(freq / A4_FREQ)


def transpose(midi: int, semitones: int) -> int:
    """Perdeyi yarım ton cinsinden taşır (aralık dışına çıkarsa hata)."""
    result = midi + semitones
    if not 0 <= result <= 127:
        raise ValueError(f"taşıma MIDI aralığını aştı: {midi} + {semitones}")
    return result


def enharmonic(name: str) -> str:
    """``"C#"`` <-> ``"Db"`` eşdeğerini döndürür (oktav korunur)."""
    value, octave = parse_note(name)
    pc = value % 12
    use_flats = "#" in name or "♯" in name
    table = FLAT_NAMES if use_flats else SHARP_NAMES
    return f"{table[pc]}{octave}" if octave is not None else table[pc]


def interval_name(semitones: int) -> str:
    """Yarım ton sayısının aralık adı; birden fazla oktavı kompozit olarak yazar."""
    simple = semitones % SEMITONES_PER_OCTAVE
    octaves = semitones // SEMITONES_PER_OCTAVE
    name = INTERVAL_NAMES[simple]
    if octaves == 0:
        return name
    return f"{name}+{octaves}okt"


def pitch_class_distance(a: int, b: int) -> int:
    """İki perde sınıfı arasındaki en kısa mesafe (0-6)."""
    diff = abs(a - b) % SEMITONES_PER_OCTAVE
    return min(diff, SEMITONES_PER_OCTAVE - diff)


@dataclass(frozen=True, order=True)
class Pitch:
    """Tek bir perde; karşılaştırma MIDI değeri üzerinden yapılır."""

    midi: int

    def __post_init__(self) -> None:
        if not 0 <= self.midi <= 127:
            raise ValueError(f"MIDI aralığı dışında: {self.midi}")

    @classmethod
    def from_name(cls, name: str) -> Pitch:
        return cls(note_to_midi(name))

    @property
    def pitch_class(self) -> int:
        return self.midi % SEMITONES_PER_OCTAVE

    @property
    def octave(self) -> int:
        return octave_of(self.midi)

    @property
    def name(self) -> str:
        return midi_to_note(self.midi)

    @property
    def flat_name(self) -> str:
        return midi_to_note(self.midi, flats=True)

    @property
    def freq(self) -> float:
        return midi_to_freq(self.midi)

    def transposed(self, semitones: int) -> Pitch:
        return Pitch(transpose(self.midi, semitones))

    def __add__(self, semitones: int) -> Pitch:
        return self.transposed(semitones)

    def __sub__(self, other: object) -> int:
        if not isinstance(other, Pitch):
            return NotImplemented
        return self.midi - other.midi

    def __str__(self) -> str:
        return self.name
