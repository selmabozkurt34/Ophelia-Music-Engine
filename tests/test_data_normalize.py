"""Normalizasyon testleri: PPQ, ızgara, perküsyon, kırpma, determinizm."""

from __future__ import annotations

from pathlib import Path

from ophelia.data.midi_io import DRUM_CHANNEL, NoteEvent, analyze, iter_notes, load_midi, write_midi
from ophelia.data.normalize import NormalizeConfig, normalize_file, normalize_notes


def test_notes_rescaled_to_target_ppq() -> None:
    notes = [NoteEvent(0, 96, 60, 90, 0, 0, 0), NoteEvent(96, 96, 62, 90, 0, 0, 0)]
    cfg = NormalizeConfig(ticks_per_beat=480, grid_division=0)
    result = normalize_notes(notes, source_ppq=96, config=cfg)

    assert [n.start for n in result] == [0, 480]
    assert [n.duration for n in result] == [480, 480]


def test_quantize_to_grid() -> None:
    notes = [NoteEvent(50, 130, 60, 90, 0, 0, 0)]
    cfg = NormalizeConfig(ticks_per_beat=480, grid_division=4)  # 120 tick ızgara
    result = normalize_notes(notes, source_ppq=480, config=cfg)

    assert result[0].start == 0
    assert result[0].duration == 120


def test_percussion_dropped() -> None:
    notes = [
        NoteEvent(0, 480, 60, 90, 0, 0, 0),
        NoteEvent(0, 480, 36, 100, DRUM_CHANNEL, 0, 0),
    ]
    result = normalize_notes(notes, 480, NormalizeConfig())
    assert len(result) == 1
    assert result[0].channel == 0


def test_percussion_kept_when_configured() -> None:
    notes = [NoteEvent(0, 480, 36, 100, DRUM_CHANNEL, 0, 0)]
    cfg = NormalizeConfig(drop_percussion=False, merge_channels=False)
    result = normalize_notes(notes, 480, cfg)
    assert len(result) == 1
    assert result[0].channel == DRUM_CHANNEL


def test_pitch_clamped_to_range() -> None:
    notes = [NoteEvent(0, 480, 110, 90, 0, 0, 0), NoteEvent(0, 480, 5, 90, 0, 0, 0)]
    cfg = NormalizeConfig(clamp_range=(21, 108))
    result = normalize_notes(notes, 480, cfg)
    assert {n.pitch for n in result} == {21, 108}


def test_channels_merged_to_zero() -> None:
    notes = [
        NoteEvent(0, 480, 60, 90, 3, 0, 0),
        NoteEvent(0, 480, 64, 90, 5, 0, 0),
    ]
    result = normalize_notes(notes, 480, NormalizeConfig(merge_channels=True))
    assert {n.channel for n in result} == {0}


def test_sub_grid_note_snaps_to_one_grid_unit() -> None:
    """Izgara altı nota 1 tick'lik hayalet değil, bir ızgara birimi olmalı."""
    notes = [NoteEvent(0, 10, 60, 90, 0, 0, 0)]
    cfg = NormalizeConfig(ticks_per_beat=480, grid_division=4)  # 120 tick
    result = normalize_notes(notes, source_ppq=480, config=cfg)

    assert result[0].duration == 120
    assert result[0].duration % cfg.grid_ticks == 0
    assert result[0].start % cfg.grid_ticks == 0


def test_all_durations_land_on_grid() -> None:
    notes = [
        NoteEvent(17, 10, 60, 90, 0, 0, 0),
        NoteEvent(53, 233, 62, 90, 0, 0, 0),
        NoteEvent(101, 7, 64, 90, 0, 0, 0),
    ]
    cfg = NormalizeConfig(ticks_per_beat=480, grid_division=4)
    result = normalize_notes(notes, source_ppq=480, config=cfg)

    assert all(note.duration % cfg.grid_ticks == 0 for note in result)
    assert all(note.start % cfg.grid_ticks == 0 for note in result)


def test_notes_sorted_and_positive_duration() -> None:
    notes = [
        NoteEvent(960, 480, 67, 90, 0, 0, 0),
        NoteEvent(0, 10, 60, 90, 0, 0, 0),
    ]
    result = normalize_notes(notes, 480, NormalizeConfig(grid_division=4))
    assert [n.start for n in result] == sorted(n.start for n in result)
    assert all(n.duration > 0 for n in result)


def test_invalid_ppq_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        normalize_notes([], 0, NormalizeConfig())


def test_normalize_file_writes_canonical_file(tmp_path: Path) -> None:
    source = tmp_path / "src.mid"
    target = tmp_path / "dst.mid"
    notes = [NoteEvent(0, 96, 60, 90, 0, 0, 0), NoteEvent(96, 96, 64, 90, 0, 0, 0)]
    write_midi(notes, source, ticks_per_beat=96, tempo_bpm=90)

    result = normalize_file(source, target, NormalizeConfig())

    assert target.is_file()
    assert result.src_ticks_per_beat == 96
    assert result.n_notes_in == 2
    assert result.n_notes_out == 2
    assert result.dropped_notes == 0

    stats = analyze(target)
    assert stats.ticks_per_beat == 480
    assert stats.time_signatures == ("4/4",)


def test_normalization_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "src.mid"
    first = tmp_path / "a.mid"
    second = tmp_path / "b.mid"
    write_midi([NoteEvent(17, 233, 61, 88, 0, 0, 0)], source, ticks_per_beat=96)

    cfg = NormalizeConfig()
    normalize_file(source, first, cfg)
    normalize_file(source, second, cfg)
    assert first.read_bytes() == second.read_bytes()


def test_normalized_file_reloads_same_notes(tmp_path: Path) -> None:
    source = tmp_path / "src.mid"
    target = tmp_path / "dst.mid"
    write_midi(
        [NoteEvent(0, 480, 60, 90, 0, 0, 0), NoteEvent(480, 480, 62, 70, 0, 0, 0)],
        source,
    )
    normalize_file(source, target)
    notes = iter_notes(load_midi(target))
    assert [(n.start, n.pitch, n.velocity) for n in notes] == [
        (0, 60, 90),
        (480, 62, 70),
    ]
