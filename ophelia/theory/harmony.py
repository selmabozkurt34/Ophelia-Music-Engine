"""Armoni: akor geçişleri, kadanslar, diyatonik fonksiyonlar ve ses yürütme.

Akorlar daima :class:`ophelia.theory.scales.Key` üzerinden diyatonik olarak
türetilir; kromatik akorlar için 4. aşamadaki temsil katmanı kullanılacaktır.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ophelia.theory.chords import Chord
from ophelia.theory.scales import Key, roman_numeral

# --- hazır progresyonlar (derece dizileri) ---------------------------------
PROGRESSIONS: dict[str, tuple[int, ...]] = {
    "I-IV-V": (1, 4, 5),
    "I-V-vi-IV": (1, 5, 6, 4),
    "vi-IV-I-V": (6, 4, 1, 5),
    "I-vi-IV-V": (1, 6, 4, 5),
    "ii-V-I": (2, 5, 1),
    "I-vi-ii-V": (1, 6, 2, 5),
    "pachelbel": (1, 5, 6, 3, 4, 1, 4, 5),
    "circle_of_fifths": (1, 4, 7, 3, 6, 2, 5, 1),
    "andalusian": (1, 7, 6, 5),
    "i-VI-III-VII": (1, 6, 3, 7),
    "minor_epic": (1, 6, 7, 1),
    "twelve_bar_blues": (1, 1, 1, 1, 4, 4, 1, 1, 5, 4, 1, 5),
}

PROGRESSION_ALIASES: dict[str, str] = {
    "pop": "I-V-vi-IV",
    "axis": "I-V-vi-IV",
    "fifties": "I-vi-IV-V",
    "jazz": "ii-V-I",
    "canon": "pachelbel",
    "fifths": "circle_of_fifths",
    "blues": "twelve_bar_blues",
    "minor_pop": "i-VI-III-VII",
}

# --- diyatonik fonksiyonlar ------------------------------------------------
TONIC = "tonic"
SUBDOMINANT = "subdominant"
DOMINANT = "dominant"

FUNCTION_BY_DEGREE: dict[int, str] = {
    1: TONIC,
    3: TONIC,
    6: TONIC,
    2: SUBDOMINANT,
    4: SUBDOMINANT,
    5: DOMINANT,
    7: DOMINANT,
}

# Yaygın akor geçişleri (derece -> izin verilen sonraki dereceler)
TRANSITIONS: dict[int, tuple[int, ...]] = {
    1: (2, 3, 4, 5, 6),
    2: (5, 7, 1),
    3: (2, 4, 6),
    4: (5, 1, 2, 7),
    5: (1, 6, 4),
    6: (2, 4, 5, 3),
    7: (1, 3),
}

CADENCE_PATTERNS: dict[str, tuple[tuple[int, int], ...]] = {
    "authentic": ((5, 1),),
    "plagal": ((4, 1),),
    "deceptive": ((5, 6),),
    "half": ((1, 5), (2, 5), (4, 5), (6, 5), (3, 5)),
}


def function_of_degree(degree: int) -> str:
    """Derecenin armonik fonksiyonu (tonic / subdominant / dominant)."""
    try:
        return FUNCTION_BY_DEGREE[(degree - 1) % 7 + 1]
    except KeyError as exc:  # pragma: no cover - anahtar her zaman var
        raise ValueError(f"geçersiz derece: {degree}") from exc


def next_degrees(degree: int) -> tuple[int, ...]:
    """Dereceden sonra gelmesi müzikal olarak beklenen dereceler."""
    return TRANSITIONS[(degree - 1) % 7 + 1]


_PROGRESSION_INDEX: dict[str, str] = {name.lower(): name for name in PROGRESSIONS}


def resolve_progression_name(name: str) -> str:
    """Progresyon adını veya takma adını ``PROGRESSIONS`` anahtarına çevirir.

    Adlar büyük/küçük harf ve Roma rakamı yazımından bağımsızdır.
    """
    key = name.strip()
    if key in PROGRESSIONS:
        return key

    lowered = key.lower()
    lowered = PROGRESSION_ALIASES.get(lowered, lowered).lower()
    if lowered in _PROGRESSION_INDEX:
        return _PROGRESSION_INDEX[lowered]

    available = ", ".join(sorted(PROGRESSIONS))
    raise ValueError(f"bilinmeyen progresyon: {name!r} (mevcut: {available})")


def progression(
    key: Key,
    degrees: Sequence[int] | str,
    *,
    seventh: bool = False,
) -> tuple[Chord, ...]:
    """Derece dizisini (veya hazır progresyon adını) akorlara çevirir."""
    if isinstance(degrees, str):
        degrees = PROGRESSIONS[resolve_progression_name(degrees)]
    return key.chord_for_degrees(list(degrees), seventh=seventh)


def progression_romans(
    key: Key,
    degrees: Sequence[int] | str,
    *,
    seventh: bool = False,
) -> tuple[str, ...]:
    if isinstance(degrees, str):
        degrees = PROGRESSIONS[resolve_progression_name(degrees)]
    return tuple(key.roman(degree, seventh=seventh) for degree in degrees)


def is_diatonic(key: Key, chord: Chord) -> bool:
    """Akorun tüm sesleri tonalitenin gamında mı?"""
    return all(key.scale.contains_pc(pc) for pc in chord.pitch_classes)


def degree_of_chord(key: Key, chord: Chord) -> int | None:
    """Diyatonik akorun derecesini bulur; diyatonik değilse ``None``."""
    pcs = set(chord.pitch_classes)
    for degree in range(1, 8):
        if set(key.chord(degree).pitch_classes) == pcs:
            return degree
    return None


def detect_cadence(degrees: Sequence[int]) -> str | None:
    """Dizinin son iki derecesinden kadans türünü tahmin eder."""
    if len(degrees) < 2:
        return None
    pair = ((degrees[-2] - 1) % 7 + 1, (degrees[-1] - 1) % 7 + 1)
    for name, patterns in CADENCE_PATTERNS.items():
        if pair in patterns:
            return name
    return None


def _nearest_pitch(pc: int, target: float, low: int, high: int) -> int:
    """``pc`` perde sınıfına ait, ``target``a en yakın ve aralık içindeki MIDI."""
    if high - low < 11:
        raise ValueError("ses yürütme aralığı en az bir oktav olmalı")

    base = low + ((pc - low) % 12)
    candidates = [base - 12, base, base + 12, base + 24]
    in_range = [pitch for pitch in candidates if low <= pitch <= high]
    if not in_range:  # pragma: no cover - aralık >= 12 iken her zaman bulunur
        return base
    return min(in_range, key=lambda pitch: abs(pitch - target))


def voice_lead(
    chords: Iterable[Chord],
    *,
    low: int = 55,
    high: int = 79,
) -> tuple[tuple[int, ...], ...]:
    """Akorları, ardışık hareketi en aza indiren seslendirmelerle dizer.

    Dönen her demet artan sırada MIDI notalarıdır ve akorun tüm perde
    sınıflarını içerir. Bas sesi ayrıca ele alınmaz; 7. aşamada kanal
    ataması sırasında en pes nota bas olarak kullanılır.
    """
    voiced: list[tuple[int, ...]] = []
    previous: tuple[int, ...] = ()

    for chord in chords:
        pcs = chord.pitch_classes
        if not previous:
            notes = [_nearest_pitch(pc, low + 12, low, high) for pc in pcs]
        else:
            notes = []
            for pc in pcs:
                options = [_nearest_pitch(pc, pitch, low, high) for pitch in previous]
                notes.append(
                    min(options, key=lambda cand: min(abs(cand - pitch) for pitch in previous))
                )
        current = tuple(sorted(set(notes)))
        voiced.append(current)
        previous = current

    return tuple(voiced)


def describe_progression(
    key: Key,
    degrees: Sequence[int] | str,
    *,
    flats: bool | None = None,
) -> list[str]:
    """Progresyonu insan okunur biçimde açar (rapor/CLI çıktısı için)."""
    if isinstance(degrees, str):
        degrees = PROGRESSIONS[resolve_progression_name(degrees)]
    use_flats = key.flats if flats is None else flats

    lines: list[str] = []
    for index, degree in enumerate(degrees):
        chord = key.chord(degree)
        roman = roman_numeral(degree, chord.quality)
        function = function_of_degree(degree)
        cadence = ""
        if index == len(degrees) - 1:
            kind = detect_cadence(list(degrees))
            cadence = f"  <- {kind} kadans" if kind else ""
        symbol = chord.to_symbol(flats=use_flats)
        lines.append(f"{roman:>4}  {symbol:<6} ({function}){cadence}")
    return lines


def cadence_positions(degrees: Sequence[int]) -> tuple[int, ...]:
    """Kadans oluşturan iki derece çiftinin bitiş indekslerini döndürür."""
    positions: list[int] = []
    for index in range(1, len(degrees)):
        if detect_cadence(list(degrees[: index + 1])) is not None:
            positions.append(index)
    return tuple(positions)
