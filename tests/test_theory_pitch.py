"""Perde/oktav dönüşümlerinin doğruluğu."""

from __future__ import annotations

import pytest
from ophelia.theory.pitch import (
    Pitch,
    enharmonic,
    interval_name,
    midi_to_freq,
    midi_to_note,
    note_to_midi,
    note_to_pitch_class,
    octave_of,
    transpose,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("C4", 60),
        ("A4", 69),
        ("C-1", 0),
        ("G9", 127),
        ("C#4", 61),
        ("Db4", 61),
        ("Bb3", 58),
        ("B#3", 60),
        ("Cb4", 59),
        ("F#2", 42),
        ("c4", 60),
    ],
)
def test_note_to_midi(name: str, expected: int) -> None:
    assert note_to_midi(name) == expected


def test_note_without_octave_uses_default() -> None:
    assert note_to_midi("C") == 60
    assert note_to_midi("D", octave=3) == 50


def test_pitch_class_is_octave_independent() -> None:
    assert note_to_pitch_class("C4") == note_to_pitch_class("C7") == 0
    assert note_to_pitch_class("Bb2") == 10


def test_invalid_note_name_raises() -> None:
    with pytest.raises(ValueError):
        note_to_midi("H4")
    with pytest.raises(ValueError):
        note_to_midi("C##4#")


def test_midi_to_note_roundtrip_over_full_range() -> None:
    for midi in range(0, 128):
        assert note_to_midi(midi_to_note(midi)) == midi


def test_flat_spelling() -> None:
    assert midi_to_note(61) == "C#4"
    assert midi_to_note(61, flats=True) == "Db4"
    assert midi_to_note(60) == "C4"


def test_midi_to_note_out_of_range() -> None:
    with pytest.raises(ValueError):
        midi_to_note(128)


def test_octave_of() -> None:
    assert octave_of(60) == 4
    assert octave_of(69) == 4
    assert octave_of(59) == 3
    assert octave_of(0) == -1


def test_frequency_conversions() -> None:
    assert midi_to_freq(69) == pytest.approx(440.0)
    assert midi_to_freq(60) == pytest.approx(261.6256, abs=1e-3)
    assert midi_to_freq(81) == pytest.approx(880.0)


def test_enharmonic_equivalence() -> None:
    assert enharmonic("C#4") == "Db4"
    assert enharmonic("Db4") == "C#4"
    assert enharmonic("Bb") == "A#"
    assert note_to_midi("C#4") == note_to_midi(enharmonic("C#4"))


def test_transpose_bounds() -> None:
    assert transpose(60, 12) == 72
    assert transpose(60, -12) == 48
    with pytest.raises(ValueError):
        transpose(127, 1)


def test_interval_names() -> None:
    assert interval_name(7) == "P5"
    assert interval_name(3) == "m3"
    assert interval_name(12) == "P1+1okt"
    assert interval_name(19) == "P5+1okt"


def test_pitch_value_object() -> None:
    pitch = Pitch.from_name("A4")
    assert pitch.midi == 69
    assert pitch.name == "A4"
    assert pitch.octave == 4
    assert pitch.freq == pytest.approx(440.0)
    assert pitch.pitch_class == 9
    assert (pitch + 12).name == "A5"
    assert (pitch - Pitch(60)) == 9
    assert Pitch(60) < Pitch(69)

    with pytest.raises(ValueError):
        Pitch(200)
