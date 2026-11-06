"""Gam, mod ve tonalite testleri."""

from __future__ import annotations

import pytest
from ophelia.theory.pitch import note_to_midi, note_to_pitch_class
from ophelia.theory.scales import (
    MODES,
    Key,
    Scale,
    estimate_key,
    roman_numeral,
    scale_midi,
    scale_pitch_classes,
)


def test_major_and_minor_patterns() -> None:
    assert scale_pitch_classes(0, "major") == (0, 2, 4, 5, 7, 9, 11)
    assert scale_pitch_classes(9, "natural_minor") == (9, 11, 0, 2, 4, 5, 7)


def test_mode_aliases() -> None:
    assert scale_pitch_classes(0, "ionian") == scale_pitch_classes(0, "major")
    assert scale_pitch_classes(9, "minor") == scale_pitch_classes(9, "aeolian")


def test_harmonic_minor_raises_seventh() -> None:
    assert scale_pitch_classes(9, "harmonic_minor") == (9, 11, 0, 2, 4, 5, 8)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        scale_pitch_classes(0, "bilinmeyen_mod")


def test_scale_midi_octave() -> None:
    assert scale_midi(0, "major", 4) == (60, 62, 64, 65, 67, 69, 71)
    assert scale_midi(9, "natural_minor", 3) == (57, 59, 60, 62, 64, 65, 67)


def test_scale_object() -> None:
    scale = Scale(0, "major")
    assert scale.name == "C major"
    assert scale.degree_count == 7
    assert scale.contains(64)  # E4
    assert not scale.contains(61)  # C#4
    assert scale.degree_pc(5) == 7
    assert scale.degree_of(note_to_pitch_class("G")) == 5
    assert scale.degree_of(note_to_pitch_class("G#")) is None


def test_diatonic_triads_c_major() -> None:
    key = Key.parse("C major")
    assert [chord.to_symbol() for chord in key.diatonic_triads()] == [
        "C",
        "Dm",
        "Em",
        "F",
        "G",
        "Am",
        "Bdim",
    ]


def test_diatonic_sevenths_c_major() -> None:
    key = Key.parse("C major")
    assert [chord.to_symbol() for chord in key.diatonic_sevenths()] == [
        "Cmaj7",
        "Dm7",
        "Em7",
        "Fmaj7",
        "G7",
        "Am7",
        "Bm7b5",
    ]


def test_diatonic_triads_a_minor() -> None:
    key = Key.parse("A minor")
    assert [chord.to_symbol() for chord in key.diatonic_triads()] == [
        "Am",
        "Bdim",
        "C",
        "Dm",
        "Em",
        "F",
        "G",
    ]


def test_harmonic_minor_dominant_is_major() -> None:
    key = Key.parse("A harmonic_minor")
    assert key.chord(5).quality == "major"
    assert key.chord(1, seventh=True).quality == "minor_major7"


def test_roman_numerals() -> None:
    key = Key.parse("C major")
    assert key.roman(1) == "I"
    assert key.roman(2) == "ii"
    assert key.roman(7) == "vii°"
    assert key.roman(5, seventh=True) == "V7"
    assert key.roman_progression([1, 5, 6, 4]) == ("I", "V", "vi", "IV")

    minor = Key.parse("A minor")
    assert minor.roman_progression([1, 6, 3, 7]) == ("i", "VI", "III", "VII")


def test_roman_numeral_errors() -> None:
    with pytest.raises(ValueError):
        roman_numeral(8, "major")


def test_key_parse_and_properties() -> None:
    assert Key.parse("C major").tonic_pc == 0
    assert Key.parse("A minor").mode == "natural_minor"
    assert Key.parse("F# dorian").tonic_pc == 6

    bb = Key.parse("Bb major")
    assert bb.tonic_name == "Bb"
    assert [chord.to_symbol(flats=True) for chord in bb.diatonic_triads()] == [
        "Bb",
        "Cm",
        "Dm",
        "Eb",
        "F",
        "Gm",
        "Adim",
    ]


def test_relative_and_parallel_keys() -> None:
    c_major = Key.parse("C major")
    assert c_major.relative_minor.name == "A natural_minor"
    assert c_major.parallel_minor.name == "C natural_minor"
    assert Key.parse("A minor").relative_major.name == "C major"


def test_estimate_key() -> None:
    c_major_histogram = {0: 100, 4: 80, 7: 70, 9: 50}
    assert estimate_key(c_major_histogram).name == "C major"

    a_minor_histogram = [20, 0, 10, 0, 30, 10, 0, 15, 0, 60, 0, 5]
    estimated = estimate_key(a_minor_histogram)
    assert estimated.tonic_pc == 9
    assert estimated.mode == "natural_minor"


def test_estimate_key_rejects_bad_histogram() -> None:
    with pytest.raises(ValueError):
        estimate_key([1, 2, 3])


def test_all_modes_produce_seven_or_five_degrees() -> None:
    for mode, pattern in MODES.items():
        assert len(scale_pitch_classes(0, mode)) == len(pattern)


def test_key_chord_in_range() -> None:
    key = Key.parse("D major")
    chord = key.chord(1)
    notes = chord.midi_notes(4)
    assert note_to_midi("D4") in notes
