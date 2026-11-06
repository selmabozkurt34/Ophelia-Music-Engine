"""Tempo, ölçü, quantize ve senkop testleri."""

from __future__ import annotations

import pytest
from ophelia.theory.rhythm import (
    EIGHTH,
    QUARTER,
    TimedNote,
    TimeSignature,
    bars_used,
    beat_grid,
    beats_to_seconds,
    bpm_to_seconds_per_quarter,
    density,
    dotted,
    is_on_beat,
    is_syncopated,
    off_beat_onsets,
    quantize,
    quantize_notes,
    seconds_to_beats,
    swing,
    total_beats,
    triplet,
)


def test_time_signature_basics() -> None:
    four_four = TimeSignature.parse("4/4")
    assert four_four.quarter_beats == pytest.approx(4.0)
    assert four_four.beat_unit_quarters == pytest.approx(1.0)
    assert str(four_four) == "4/4"

    six_eight = TimeSignature(6, 8)
    assert six_eight.quarter_beats == pytest.approx(3.0)
    assert six_eight.is_compound
    assert not four_four.is_compound

    three_four = TimeSignature.parse("3 / 4")
    assert three_four.quarter_beats == pytest.approx(3.0)
    assert three_four.is_simple


def test_time_signature_validation() -> None:
    with pytest.raises(ValueError):
        TimeSignature(0, 4)
    with pytest.raises(ValueError):
        TimeSignature(4, 5)
    with pytest.raises(ValueError):
        TimeSignature.parse("4")


def test_bar_helpers() -> None:
    signature = TimeSignature(4, 4)
    assert signature.bar_of(0) == 0
    assert signature.bar_of(4) == 1
    assert signature.bar_of(9.5) == 2
    assert signature.beat_in_bar(5.5) == pytest.approx(1.5)
    assert signature.bar_seconds(120) == pytest.approx(2.0)


def test_subdivisions_and_accents() -> None:
    signature = TimeSignature(4, 4)
    subs = signature.subdivisions(QUARTER)
    assert subs == (0.0, 1.0, 2.0, 3.0)
    assert len(signature.subdivisions(EIGHTH)) == 8
    assert signature.accents() == (0.0, 1.0, 2.0, 3.0)

    compound = TimeSignature(6, 8)
    assert compound.accents() == (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)


def test_duration_helpers() -> None:
    assert dotted(QUARTER) == pytest.approx(1.5)
    assert dotted(QUARTER, 2) == pytest.approx(1.75)
    assert triplet(QUARTER) == pytest.approx(2.0 / 3.0)


def test_tempo_conversions() -> None:
    assert bpm_to_seconds_per_quarter(120) == pytest.approx(0.5)
    assert beats_to_seconds(4, 120) == pytest.approx(2.0)
    assert seconds_to_beats(2.0, 120) == pytest.approx(4.0)

    with pytest.raises(ValueError):
        bpm_to_seconds_per_quarter(0)


def test_quantize() -> None:
    assert quantize(0.12) == pytest.approx(0.0)
    assert quantize(0.26, EIGHTH) == pytest.approx(0.5)
    assert quantize(1.24) == pytest.approx(1.25)

    notes = [TimedNote(0.49, 0.26), TimedNote(1.13, 0.9)]
    aligned = quantize_notes(notes, 0.25)
    assert [note.start for note in aligned] == [pytest.approx(0.5), pytest.approx(1.25)]
    assert all(note.duration > 0 for note in aligned)


def test_timed_note_validation() -> None:
    with pytest.raises(ValueError):
        TimedNote(-1, 1)
    with pytest.raises(ValueError):
        TimedNote(0, 0)
    with pytest.raises(ValueError):
        TimedNote(0, 1, 60, 200)

    note = TimedNote(1.0, 0.5)
    assert note.end == pytest.approx(1.5)


def test_is_on_beat_and_off_beats() -> None:
    assert is_on_beat(2.0)
    assert not is_on_beat(2.5)

    notes = [TimedNote(0, 1), TimedNote(1.5, 0.5)]
    assert off_beat_onsets(notes) == (1.5,)


def test_syncopation_detection() -> None:
    straight = [TimedNote(0, 1), TimedNote(1, 1), TimedNote(2, 1), TimedNote(3, 1)]
    assert not is_syncopated(straight)

    offbeat_short = [TimedNote(0, 1), TimedNote(1.5, 0.5)]
    assert not is_syncopated(offbeat_short)

    syncopated = [TimedNote(0, 0.5), TimedNote(0.5, 1.0), TimedNote(1.5, 1.0)]
    assert is_syncopated(syncopated)


def test_swing() -> None:
    notes = [TimedNote(0, 0.5), TimedNote(0.5, 0.5), TimedNote(1, 0.5)]
    swung = swing(notes)
    assert swung[0].start == pytest.approx(0.0)
    assert swung[1].start == pytest.approx(2.0 / 3.0)
    assert swung[2].start == pytest.approx(1.0)

    with pytest.raises(ValueError):
        swing(notes, ratio=1.5)


def test_aggregate_measures() -> None:
    notes = [TimedNote(i, 1) for i in range(8)]
    signature = TimeSignature(4, 4)
    assert total_beats(notes) == pytest.approx(8.0)
    assert bars_used(notes, signature) == pytest.approx(2.0)
    assert density(notes, signature) == pytest.approx(4.0)
    assert density([], signature) == 0.0


def test_beat_grid() -> None:
    grid = beat_grid(TimeSignature(4, 4), bars=2, grid=QUARTER)
    assert len(grid) == 8
    assert grid[0] == 0.0
    assert grid[-1] == pytest.approx(7.0)

    with pytest.raises(ValueError):
        beat_grid(TimeSignature(4, 4), bars=0)
    with pytest.raises(ValueError):
        TimeSignature(4, 4).subdivisions(0)
