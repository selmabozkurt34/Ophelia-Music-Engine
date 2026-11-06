"""Akor yapıları: üçlüler, yedili/uzatılmış akorlar, çevrimler, sembol ayrıştırma.

Akorlar perde sınıfı (pitch class) kümesi olarak temsil edilir; seslendirme
(voicing) gerektiğinde :meth:`Chord.midi_notes` kullanılır.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ophelia.theory.pitch import (
    FLAT_NAMES,
    SHARP_NAMES,
    note_to_pitch_class,
)

# Kalite -> kök perdeden yarım ton aralıkları (9'lular 14 olarak yazılır)
CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "diminished": (0, 3, 6),
    "augmented": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "power5": (0, 7),
    "major6": (0, 4, 7, 9),
    "minor6": (0, 3, 7, 9),
    "major7": (0, 4, 7, 11),
    "minor7": (0, 3, 7, 10),
    "dominant7": (0, 4, 7, 10),
    "minor_major7": (0, 3, 7, 11),
    "half_diminished7": (0, 3, 6, 10),
    "diminished7": (0, 3, 6, 9),
    "add9": (0, 4, 7, 14),
    "dominant9": (0, 4, 7, 10, 14),
    "major9": (0, 4, 7, 11, 14),
    "minor9": (0, 3, 7, 10, 14),
}

# Sembol yazımı (kanonik çıktı)
QUALITY_SUFFIX: dict[str, str] = {
    "major": "",
    "minor": "m",
    "diminished": "dim",
    "augmented": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "power5": "5",
    "major6": "6",
    "minor6": "m6",
    "major7": "maj7",
    "minor7": "m7",
    "dominant7": "7",
    "minor_major7": "mM7",
    "half_diminished7": "m7b5",
    "diminished7": "dim7",
    "add9": "add9",
    "dominant9": "9",
    "major9": "maj9",
    "minor9": "m9",
}

# Girişte kabul edilen eşanlamlı yazımlar
_SYMBOL_ALIASES: dict[str, str] = {
    "": "major",
    "maj": "major",
    "M": "major",
    "m": "minor",
    "min": "minor",
    "-": "minor",
    "dim": "diminished",
    "o": "diminished",
    "°": "diminished",
    "aug": "augmented",
    "+": "augmented",
    "maj7": "major7",
    "M7": "major7",
    "Δ7": "major7",
    "m7": "minor7",
    "min7": "minor7",
    "-7": "minor7",
    "7": "dominant7",
    "dom7": "dominant7",
    "m7b5": "half_diminished7",
    "ø": "half_diminished7",
    "ø7": "half_diminished7",
    "dim7": "diminished7",
    "o7": "diminished7",
    "°7": "diminished7",
    "mM7": "minor_major7",
    "6": "major6",
    "m6": "minor6",
    "9": "dominant9",
    "maj9": "major9",
    "m9": "minor9",
    "add9": "add9",
    "sus2": "sus2",
    "sus4": "sus4",
    "sus": "sus4",
    "5": "power5",
}

_ROOT_RE = re.compile(r"^([A-Ga-g][#b♯♭]?)(.*)$")


@dataclass(frozen=True)
class Chord:
    """Kök perde sınıfı + kalite + çevrim."""

    root_pc: int
    quality: str = "major"
    inversion: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.root_pc < 12:
            raise ValueError(f"kök perde sınıfı 0-11 olmalı: {self.root_pc}")
        if self.quality not in CHORD_INTERVALS:
            raise ValueError(f"bilinmeyen akor kalitesi: {self.quality}")
        if not 0 <= self.inversion < len(CHORD_INTERVALS[self.quality]):
            raise ValueError(f"{self.quality} için geçersiz çevrim: {self.inversion}")

    # --- yapı ---------------------------------------------------------------
    @property
    def intervals(self) -> tuple[int, ...]:
        return CHORD_INTERVALS[self.quality]

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        return tuple((self.root_pc + i) % 12 for i in self.intervals)

    @property
    def bass_pc(self) -> int:
        """Çevrim uygulanmış akorun en bas sesinin perde sınıfı."""
        return self.pitch_classes[self.inversion]

    @property
    def size(self) -> int:
        return len(self.intervals)

    @property
    def is_major_triad(self) -> bool:
        return self.quality == "major"

    @property
    def is_minor_triad(self) -> bool:
        return self.quality == "minor"

    @property
    def is_dominant_seventh(self) -> bool:
        return self.quality == "dominant7"

    # --- yazım --------------------------------------------------------------
    def root_name(self, *, flats: bool = False) -> str:
        return (FLAT_NAMES if flats else SHARP_NAMES)[self.root_pc]

    def to_symbol(self, *, flats: bool = False) -> str:
        """``"Cmaj7"``, ``"F#m"``, ``"Bb7"`` gibi kanonik sembol."""
        symbol = self.root_name(flats=flats) + QUALITY_SUFFIX[self.quality]
        if self.inversion:
            symbol += f"/{SHARP_NAMES[self.bass_pc]}"
        return symbol

    def note_names(self, *, flats: bool = False) -> tuple[str, ...]:
        table = FLAT_NAMES if flats else SHARP_NAMES
        return tuple(table[pc] for pc in self.pitch_classes)

    # --- seslendirme --------------------------------------------------------
    def midi_notes(self, octave: int = 4, *, apply_inversion: bool = True) -> tuple[int, ...]:
        """Akoru belirli bir oktavdan başlayarak MIDI notalarına açar."""
        base = 12 * (octave + 1) + self.root_pc
        notes = [base + interval for interval in self.intervals]
        if apply_inversion:
            for index in range(self.inversion):
                notes[index] += 12
        return tuple(sorted(notes))

    def inverted(self, steps: int = 1) -> Chord:
        """Çevrimi ``steps`` kadar ilerletir (mod ile sarar)."""
        return Chord(self.root_pc, self.quality, (self.inversion + steps) % self.size)

    def transposed(self, semitones: int) -> Chord:
        return Chord((self.root_pc + semitones) % 12, self.quality, self.inversion)

    # --- ayrıştırma ---------------------------------------------------------
    @classmethod
    def from_symbol(cls, symbol: str) -> Chord:
        """``"C#m7"`` -> ``Chord(1, "minor7")``; ``"C/G"`` eğik çizgi yazımını destekler."""
        text = symbol.strip()
        bass_pc: int | None = None
        if "/" in text:
            text, bass_text = text.split("/", 1)
            bass_pc = note_to_pitch_class(bass_text)

        match = _ROOT_RE.match(text)
        if match is None:
            raise ValueError(f"geçersiz akor sembolü: {symbol!r}")
        root_text, suffix = match.groups()
        quality = _SYMBOL_ALIASES.get(suffix)
        if quality is None:
            raise ValueError(f"bilinmeyen akor eki: {suffix!r} ({symbol!r})")

        chord = cls(note_to_pitch_class(root_text), quality)
        for index, pc in enumerate(chord.pitch_classes):
            if pc == bass_pc:
                return cls(chord.root_pc, quality, index)
        return chord

    def __str__(self) -> str:
        return self.to_symbol()


def identify_chord(midi_notes: tuple[int, ...] | list[int]) -> Chord | None:
    """MIDI notalarından akoru tanır; tanınamazsa ``None``.

    Bas ses (en pes nota) çevrimi belirler: aynı perde sınıfı kümesi farklı
    akorlara karşılık gelebildiği için (C6 = Am7) bas ses önceliklidir.
    """
    if len(midi_notes) < 2:
        return None

    ordered = tuple(sorted(midi_notes))
    pcs = {note % 12 for note in ordered}
    bass_pc = ordered[0] % 12

    # Bas sesin kök olduğu yorum önce denenir.
    roots = sorted(pcs, key=lambda pc: (pc != bass_pc, pc))

    for root in roots:
        for quality, intervals in CHORD_INTERVALS.items():
            expected = frozenset((root + interval) % 12 for interval in intervals)
            if expected != pcs or len(expected) != len(pcs):
                continue
            offset = (bass_pc - root) % 12
            inversion = 0
            for index, interval in enumerate(intervals):
                if interval % 12 == offset:
                    inversion = index
                    break
            return Chord(root, quality, inversion)
    return None


def chord_from_notes(midi_notes: tuple[int, ...] | list[int]) -> Chord:
    """:func:`identify_chord` ile aynı, ancak tanınamazsa hata verir."""
    chord = identify_chord(midi_notes)
    if chord is None:
        raise ValueError(f"akor tanınamadı: {sorted(midi_notes)}")
    return chord
