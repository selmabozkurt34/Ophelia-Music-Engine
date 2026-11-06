"""Gamlar, modlar ve tonalite.

Diyatonik akorlar gamdan türetilir ve :func:`ophelia.theory.chords.identify_chord`
ile tanınır; böylece akor bilgisi tek bir yerde (chords.py) kalır.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ophelia.theory.chords import Chord, identify_chord
from ophelia.theory.pitch import FLAT_NAMES, SHARP_NAMES, note_to_pitch_class

# Gam kalıpları: kök perdeden yarım ton uzaklıkları
MODES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "pentatonic_major": (0, 2, 4, 7, 9),
    "pentatonic_minor": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
    "chromatic": tuple(range(12)),
}

MODE_ALIASES: dict[str, str] = {
    "maj": "major",
    "ionian": "major",
    "minor": "natural_minor",
    "min": "natural_minor",
    "m": "natural_minor",
    "aeolian": "natural_minor",
    "harm_minor": "harmonic_minor",
    "mel_minor": "melodic_minor",
}

ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII")

# Krumhansl-Kessler perde sınıfı profilleri (tonalite tahmini için)
KK_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
KK_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

_MINOR_QUALITIES = frozenset(
    {"minor", "diminished", "minor7", "minor_major7", "minor9", "half_diminished7"}
)
_SEVENTH_QUALITIES = frozenset(
    {"major7", "minor7", "dominant7", "half_diminished7", "diminished7", "minor_major7"}
)


def normalize_mode(mode: str) -> str:
    """Mod adını kanonik anahtara çevirir; bilinmeyen adda hata verir."""
    key = mode.strip().lower().replace(" ", "_")
    key = MODE_ALIASES.get(key, key)
    if key not in MODES:
        raise ValueError(f"bilinmeyen mod/gam: {mode!r}")
    return key


def is_minor_like(mode: str) -> bool:
    """Modun minör karakterli olup olmadığı (küçük üçlü içerir mi)."""
    return MODES[normalize_mode(mode)][2] == 3


def scale_pitch_classes(root_pc: int, mode: str = "major") -> tuple[int, ...]:
    """Gamın perde sınıfları (kökten başlayarak, artan)."""
    pattern = MODES[normalize_mode(mode)]
    return tuple((root_pc + step) % 12 for step in pattern)


def scale_midi(root_pc: int, mode: str = "major", octave: int = 4) -> tuple[int, ...]:
    """Gamı belirli oktavdan başlayarak MIDI notalarına açar (bir oktav)."""
    pattern = MODES[normalize_mode(mode)]
    base = 12 * (octave + 1) + root_pc
    return tuple(base + step for step in pattern)


def roman_numeral(degree: int, quality: str, *, seventh: bool = False) -> str:
    """Ölçek derecesi ve akor kalitesinden Roma rakamı üretir."""
    if not 1 <= degree <= 7:
        raise ValueError(f"derece 1-7 arasında olmalı: {degree}")

    base = ROMAN[degree - 1]
    if quality in _MINOR_QUALITIES:
        base = base.lower()

    if seventh or quality in _SEVENTH_QUALITIES:
        if quality == "diminished7":
            return f"{base}°7"
        if quality == "half_diminished7":
            return f"{base}ø7"
        return f"{base}7"
    if quality == "diminished":
        return f"{base}°"
    if quality == "augmented":
        return f"{base}+"
    return base


@dataclass(frozen=True)
class Scale:
    """Kök + mod; diyatonik akor türetiminin temeli."""

    root_pc: int
    mode: str = "major"
    flats: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.root_pc < 12:
            raise ValueError(f"kök perde sınıfı 0-11 olmalı: {self.root_pc}")
        object.__setattr__(self, "mode", normalize_mode(self.mode))

    @property
    def intervals(self) -> tuple[int, ...]:
        return MODES[self.mode]

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        return scale_pitch_classes(self.root_pc, self.mode)

    @property
    def root_name(self) -> str:
        return (FLAT_NAMES if self.flats else SHARP_NAMES)[self.root_pc]

    @property
    def name(self) -> str:
        return f"{self.root_name} {self.mode}"

    @property
    def degree_count(self) -> int:
        return len(self.intervals)

    def contains_pc(self, pc: int) -> bool:
        return pc % 12 in self.pitch_classes

    def contains(self, midi: int) -> bool:
        return self.contains_pc(midi)

    def degree_pc(self, degree: int) -> int:
        """1 tabanlı derecenin perde sınıfı (derece sayısını aşarsa sarar)."""
        pcs = self.pitch_classes
        return pcs[(degree - 1) % len(pcs)]

    def degree_of(self, pc: int) -> int | None:
        """Perde sınıfının 1 tabanlı derecesi; gamda yoksa ``None``."""
        pcs = self.pitch_classes
        for index, value in enumerate(pcs):
            if value == pc % 12:
                return index + 1
        return None

    def midi_notes(self, octave: int = 4) -> tuple[int, ...]:
        return scale_midi(self.root_pc, self.mode, octave)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Key:
    """Tonalite: gam + diyatonik akor bilgisi."""

    tonic_pc: int
    mode: str = "major"
    flats: bool = False
    scale: Scale = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        normalized = normalize_mode(self.mode)
        object.__setattr__(self, "mode", normalized)
        object.__setattr__(self, "scale", Scale(self.tonic_pc, normalized, self.flats))

    # --- yazım --------------------------------------------------------------
    @property
    def tonic_name(self) -> str:
        return (FLAT_NAMES if self.flats else SHARP_NAMES)[self.tonic_pc]

    @property
    def name(self) -> str:
        return f"{self.tonic_name} {self.mode}"

    @property
    def parallel_minor(self) -> Key:
        return Key(self.tonic_pc, "natural_minor", self.flats)

    @property
    def relative_minor(self) -> Key:
        return Key((self.tonic_pc + 9) % 12, "natural_minor", self.flats)

    @property
    def relative_major(self) -> Key:
        return Key((self.tonic_pc + 3) % 12, "major", self.flats)

    # --- diyatonik yapı -----------------------------------------------------
    def chord(self, degree: int, *, seventh: bool = False) -> Chord:
        """Derecenin diyatonik akoru (gam üzerinde üçlü yığma)."""
        pcs = self.scale.pitch_classes
        count = len(pcs)
        if count < 7:
            raise ValueError(f"{self.mode} gamından diyatonik akor türetilemez")

        index = (degree - 1) % count
        root = pcs[index]
        steps = (2, 4, 6) if seventh else (2, 4)
        notes = [root + 12]
        for step in steps:
            target = pcs[(index + step) % count]
            notes.append(notes[-1] + ((target - (notes[-1] % 12)) % 12))

        chord = identify_chord(notes)
        if chord is None:  # pragma: no cover - diyatonik akorlar her zaman tanınır
            raise ValueError(f"diyatonik akor tanınamadı: derece {degree}")
        return chord

    def diatonic_triads(self) -> tuple[Chord, ...]:
        return tuple(self.chord(degree) for degree in range(1, 8))

    def diatonic_sevenths(self) -> tuple[Chord, ...]:
        return tuple(self.chord(degree, seventh=True) for degree in range(1, 8))

    def roman(self, degree: int, *, seventh: bool = False) -> str:
        """Derecenin Roma rakamı yazımı (``I``, ``vi``, ``V7``, ``vii°``)."""
        chord = self.chord(degree, seventh=seventh)
        return roman_numeral(degree, chord.quality, seventh=seventh)

    def roman_progression(self, degrees: list[int] | tuple[int, ...]) -> tuple[str, ...]:
        return tuple(self.roman(degree) for degree in degrees)

    def chord_for_degrees(
        self, degrees: list[int] | tuple[int, ...], *, seventh: bool = False
    ) -> tuple[Chord, ...]:
        return tuple(self.chord(degree, seventh=seventh) for degree in degrees)

    def __str__(self) -> str:
        return self.name

    # --- ayrıştırma ---------------------------------------------------------
    @classmethod
    def parse(cls, text: str) -> Key:
        """``"C major"``, ``"A minor"``, ``"F# dorian"`` gibi metni çözer."""
        parts = text.replace("_", " ").split()
        if not parts:
            raise ValueError("boş tonalite")
        tonic = parts[0]
        mode = " ".join(parts[1:]) if len(parts) > 1 else "major"
        flats = "b" in tonic or "♭" in tonic
        return cls(note_to_pitch_class(tonic), mode, flats)


def _correlate(x: tuple[float, ...], y: tuple[float, ...]) -> float:
    count = len(x)
    mean_x = sum(x) / count
    mean_y = sum(y) / count
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def estimate_key(
    histogram: dict[int, float] | list[float] | tuple[float, ...],
) -> Key:
    """Perde sınıfı histogramından tonalite tahmini (Krumhansl-Kessler).

    ``histogram`` bir dict (perde sınıfı -> ağırlık) veya 12 elemanlı bir dizi
    olabilir. Dönen tonalite her zaman majör ya da doğal minördür.
    """
    if isinstance(histogram, dict):
        counts = [float(histogram.get(pc, 0.0)) for pc in range(12)]
    else:
        counts = [float(value) for value in histogram]
        if len(counts) != 12:
            raise ValueError("histogram 12 perde sınıfı içermeli")

    vector = tuple(counts)
    best_score = float("-inf")
    best_key = Key(0, "major")

    for tonic in range(12):
        for mode, profile in (("major", KK_MAJOR), ("natural_minor", KK_MINOR)):
            rotated = tuple(profile[(pc - tonic) % 12] for pc in range(12))
            score = _correlate(vector, rotated)
            if score > best_score:
                best_score = score
                best_key = Key(tonic, mode, flats=False)
    return best_key
