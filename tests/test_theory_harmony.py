"""Armoni: progresyon, kadans, fonksiyon ve ses yürütme testleri."""

from __future__ import annotations

import pytest
from ophelia.theory.chords import Chord
from ophelia.theory.harmony import (
    PROGRESSIONS,
    cadence_positions,
    degree_of_chord,
    describe_progression,
    detect_cadence,
    function_of_degree,
    is_diatonic,
    next_degrees,
    progression,
    progression_romans,
    resolve_progression_name,
    voice_lead,
)
from ophelia.theory.scales import Key


def test_progression_symbols_in_c_major() -> None:
    key = Key.parse("C major")
    chords = progression(key, "I-V-vi-IV")
    assert [chord.to_symbol() for chord in chords] == ["C", "G", "Am", "F"]


def test_progression_with_sevenths() -> None:
    key = Key.parse("C major")
    chords = progression(key, [1, 5, 6, 4], seventh=True)
    assert [chord.to_symbol() for chord in chords] == ["Cmaj7", "G7", "Am7", "Fmaj7"]


def test_progression_from_degrees_in_a_minor() -> None:
    key = Key.parse("A minor")
    chords = progression(key, [1, 6, 3, 7])
    assert [chord.to_symbol() for chord in chords] == ["Am", "F", "C", "G"]


def test_progression_romans() -> None:
    key = Key.parse("C major")
    assert progression_romans(key, "pachelbel")[:4] == ("I", "V", "vi", "iii")


def test_resolve_aliases() -> None:
    assert resolve_progression_name("pop") == "I-V-vi-IV"
    assert resolve_progression_name("blues") == "twelve_bar_blues"
    with pytest.raises(ValueError):
        resolve_progression_name("yok")


def test_all_progressions_build() -> None:
    key = Key.parse("C major")
    for name in PROGRESSIONS:
        chords = progression(key, name)
        assert len(chords) == len(PROGRESSIONS[name])


@pytest.mark.parametrize(
    ("degrees", "expected"),
    [
        ((1, 5, 1), "authentic"),
        ((1, 4, 1), "plagal"),
        ((1, 5, 6), "deceptive"),
        ((1, 4, 5), "half"),
        ((1, 6), None),
        ((5,), None),
        ((), None),
    ],
)
def test_detect_cadence(degrees: tuple[int, ...], expected: str | None) -> None:
    assert detect_cadence(degrees) == expected


def test_cadence_positions() -> None:
    # (1,5) yarım kadans, (5,1) tam kadans, (4,1) plagal kadans
    assert cadence_positions([1, 5, 1, 4, 1]) == (1, 2, 4)


def test_harmonic_functions() -> None:
    assert function_of_degree(1) == "tonic"
    assert function_of_degree(6) == "tonic"
    assert function_of_degree(4) == "subdominant"
    assert function_of_degree(5) == "dominant"
    assert function_of_degree(8) == "tonic"  # dereceler sarar


def test_transitions_are_valid() -> None:
    assert 5 in next_degrees(2)
    assert 1 in next_degrees(5)
    assert 1 in next_degrees(7)


def test_is_diatonic() -> None:
    key = Key.parse("C major")
    assert is_diatonic(key, Chord.from_symbol("Am"))
    assert is_diatonic(key, Chord.from_symbol("Cmaj7"))
    assert not is_diatonic(key, Chord.from_symbol("Db"))
    assert not is_diatonic(key, Chord.from_symbol("E7"))


def test_degree_of_chord() -> None:
    key = Key.parse("C major")
    assert degree_of_chord(key, Chord.from_symbol("Am")) == 6
    assert degree_of_chord(key, Chord.from_symbol("Bdim")) == 7
    assert degree_of_chord(key, Chord.from_symbol("Bb")) is None


def test_voice_lead_stays_in_range_and_keeps_pitch_classes() -> None:
    key = Key.parse("C major")
    chords = progression(key, "I-V-vi-IV")
    voiced = voice_lead(chords, low=55, high=79)

    assert len(voiced) == len(chords)
    for chord, notes in zip(chords, voiced, strict=True):
        assert all(55 <= note <= 79 for note in notes)
        assert set(note % 12 for note in notes) == set(chord.pitch_classes)
        assert list(notes) == sorted(notes)


def test_voice_lead_minimises_movement() -> None:
    key = Key.parse("C major")
    voiced = voice_lead(progression(key, [1, 6]), low=55, high=79)
    first, second = voiced
    movement = sum(min(abs(a - b) for b in second) for a in first)
    assert movement <= 6  # C -> Am neredeyse ortak seslerle bağlanır


def test_voice_lead_requires_octave_range() -> None:
    with pytest.raises(ValueError):
        voice_lead([Chord.from_symbol("C")], low=60, high=65)


def test_describe_progression() -> None:
    key = Key.parse("C major")
    lines = describe_progression(key, "I-V-vi-IV")
    assert len(lines) == 4
    assert "I" in lines[0]
    assert "subdominant" in lines[-1]  # IV -> F

    cadenced = describe_progression(key, [1, 4, 5, 1])
    assert "authentic" in cadenced[-1]
