"""Akor yapıları, semboller ve akor tanıma testleri."""

from __future__ import annotations

import pytest
from ophelia.theory.chords import CHORD_INTERVALS, Chord, chord_from_notes, identify_chord


@pytest.mark.parametrize(
    ("symbol", "quality"),
    [
        ("C", "major"),
        ("Cm", "minor"),
        ("Cdim", "diminished"),
        ("Caug", "augmented"),
        ("Cmaj7", "major7"),
        ("Cm7", "minor7"),
        ("C7", "dominant7"),
        ("Cm7b5", "half_diminished7"),
        ("Cdim7", "diminished7"),
        ("Csus4", "sus4"),
        ("C6", "major6"),
        ("Cm9", "minor9"),
    ],
)
def test_from_symbol(symbol: str, quality: str) -> None:
    assert Chord.from_symbol(symbol).quality == quality


def test_from_symbol_root_and_roundtrip() -> None:
    chord = Chord.from_symbol("F#m7")
    assert chord.root_pc == 6
    assert chord.quality == "minor7"
    assert chord.to_symbol() == "F#m7"


def test_from_symbol_slash_chord() -> None:
    chord = Chord.from_symbol("C/G")
    assert chord.root_pc == 0
    assert chord.quality == "major"
    assert chord.inversion == 2  # C/G ikinci çevrim
    assert chord.bass_pc == 7


def test_unknown_symbol_raises() -> None:
    with pytest.raises(ValueError):
        Chord.from_symbol("H7")
    with pytest.raises(ValueError):
        Chord.from_symbol("Cxyz")


def test_pitch_classes_and_names() -> None:
    chord = Chord.from_symbol("Cmaj7")
    assert chord.pitch_classes == (0, 4, 7, 11)
    assert chord.note_names() == ("C", "E", "G", "B")
    assert chord.size == 4


def test_midi_notes_and_inversions() -> None:
    assert Chord.from_symbol("C").midi_notes(4) == (60, 64, 67)
    assert Chord.from_symbol("C").midi_notes(4, apply_inversion=False) == (60, 64, 67)

    first = Chord(0, "major", 1)
    assert first.midi_notes(4) == (64, 67, 72)
    assert first.bass_pc == 4

    second = Chord(0, "major", 2)
    assert second.midi_notes(4) == (67, 72, 76)


def test_inverted_and_transposed() -> None:
    chord = Chord.from_symbol("C")
    assert chord.inverted().to_symbol() == "C/E"
    assert chord.inverted(2).to_symbol() == "C/G"
    assert chord.inverted(3).to_symbol() == "C"
    assert chord.transposed(7).to_symbol() == "G"


def test_invalid_quality_and_inversion() -> None:
    with pytest.raises(ValueError):
        Chord(0, "bilinmeyen")
    with pytest.raises(ValueError):
        Chord(0, "major", 3)


@pytest.mark.parametrize(
    ("notes", "symbol"),
    [
        ((60, 64, 67), "C"),
        ((62, 65, 69), "Dm"),
        ((59, 62, 66), "Bm"),
        ((60, 63, 66), "Cdim"),
        ((60, 64, 68), "Caug"),
        ((60, 64, 67, 71), "Cmaj7"),
        ((60, 63, 67, 70), "Cm7"),
        ((60, 64, 67, 70), "C7"),
        ((59, 62, 65, 69), "Bm7b5"),
    ],
)
def test_identify_chord(notes: tuple[int, ...], symbol: str) -> None:
    assert identify_chord(notes) is not None
    assert identify_chord(notes).to_symbol() == symbol


def test_identify_detects_inversion() -> None:
    chord = identify_chord([64, 67, 72])
    assert chord is not None
    assert chord.to_symbol() == "C/E"
    assert chord.inversion == 1


def test_bass_note_decides_ambiguous_sets() -> None:
    # {C, E, G, A} hem C6 hem Am7'dir; bas ses belirleyicidir.
    assert identify_chord([60, 64, 67, 69]).to_symbol() == "C6"
    assert identify_chord([57, 60, 64, 67]).to_symbol() == "Am7"


def test_identify_unknown_returns_none() -> None:
    assert identify_chord([60]) is None
    assert identify_chord([60, 61, 62]) is None


def test_chord_from_notes_raises_on_unknown() -> None:
    with pytest.raises(ValueError):
        chord_from_notes([60, 61])


def test_all_qualities_have_symbols() -> None:
    for quality in CHORD_INTERVALS:
        symbol = Chord(0, quality).to_symbol()
        assert Chord.from_symbol(symbol).quality == quality
