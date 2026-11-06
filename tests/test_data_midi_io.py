"""MIDI okuma/yazma/analiz testleri."""

from __future__ import annotations

from pathlib import Path

import mido
import pytest
from ophelia.data.midi_io import (
    DRUM_CHANNEL,
    MidiError,
    NoteEvent,
    analyze,
    iter_notes,
    load_midi,
    max_polyphony,
    write_midi,
)
from tests.conftest import make_notes


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    notes = make_notes((60, 62, 64), step=480, duration=240)
    path = tmp_path / "a.mid"
    write_midi(notes, path)

    assert path.is_file()
    reloaded = iter_notes(load_midi(path))
    assert [(n.start, n.duration, n.pitch) for n in reloaded] == [
        (0, 240, 60),
        (480, 240, 62),
        (960, 240, 64),
    ]


def test_write_is_deterministic(tmp_path: Path) -> None:
    notes = make_notes((72, 71, 69, 67))
    first = tmp_path / "a.mid"
    second = tmp_path / "b.mid"
    write_midi(notes, first)
    write_midi(notes, second)
    assert first.read_bytes() == second.read_bytes()


def test_analyze_basic_metrics(midi_file) -> None:
    path = midi_file("x.mid", make_notes((60, 64, 67, 72), step=480, duration=480))
    stats = analyze(path)

    assert stats.n_notes == 4
    assert stats.n_tracks == 2  # meta + tek kanal
    assert stats.pitch_min == 60
    assert stats.pitch_max == 72
    assert stats.unique_pitches == 4
    assert stats.max_polyphony == 1
    assert stats.ticks_per_beat == 480
    assert stats.time_signatures == ("4/4",)
    assert not stats.has_percussion
    # 4 vuruş @100 BPM = 2.4 sn
    assert stats.duration_seconds == pytest.approx(2.4, abs=0.05)


def test_analysis_detects_polyphony(tmp_path: Path) -> None:
    chord = [
        NoteEvent(0, 480, pitch, 90, 0, 0, 0) for pitch in (60, 64, 67, 71)
    ]
    path = tmp_path / "chord.mid"
    write_midi(chord, path)
    stats = analyze(path)
    assert stats.max_polyphony == 4


def test_analysis_detects_percussion(tmp_path: Path) -> None:
    notes = [
        NoteEvent(0, 480, 36, 100, DRUM_CHANNEL, 0, 0),
        NoteEvent(480, 480, 38, 100, DRUM_CHANNEL, 0, 0),
    ]
    path = tmp_path / "drums.mid"
    write_midi(notes, path)
    stats = analyze(path)
    assert stats.has_percussion
    assert DRUM_CHANNEL in stats.channels


def test_missing_file_raises() -> None:
    with pytest.raises(MidiError):
        load_midi("yok/boyle/dosya.mid")


def test_corrupt_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bozuk.mid"
    bad.write_bytes(b"bu bir midi dosyasi degil")
    with pytest.raises(MidiError):
        load_midi(bad)


def test_max_polyphony_helper() -> None:
    notes = [
        NoteEvent(0, 100, 60, 90, 0, 0, 0),
        NoteEvent(50, 100, 62, 90, 0, 0, 0),
        NoteEvent(200, 100, 64, 90, 0, 0, 0),
    ]
    assert max_polyphony(notes) == 2


def test_unclosed_note_is_closed_at_track_end(tmp_path: Path) -> None:
    path = tmp_path / "acik.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.Message("note_on", note=60, velocity=90, time=0))
    track.append(mido.Message("note_on", note=62, velocity=90, time=480))
    track.append(mido.MetaMessage("end_of_track", time=480))
    midi.tracks.append(track)
    midi.save(str(path))

    notes = iter_notes(load_midi(path))
    assert len(notes) == 2
    assert all(note.duration > 0 for note in notes)


def test_program_change_is_captured(tmp_path: Path) -> None:
    path = tmp_path / "prog.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.Message("program_change", program=42, time=0))
    track.append(mido.Message("note_on", note=60, velocity=90, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    midi.tracks.append(track)
    midi.save(str(path))

    notes = iter_notes(load_midi(path))
    assert notes[0].program == 42
