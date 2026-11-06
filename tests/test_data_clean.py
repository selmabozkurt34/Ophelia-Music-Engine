"""Temizleme filtreleri testleri."""

from __future__ import annotations

from pathlib import Path

import pytest
from ophelia.data.clean import (
    REASON_TEXT,
    CleanConfig,
    check_stats,
    clean_corpus,
    clean_file,
)
from ophelia.data.midi_io import MidiStats, NoteEvent, write_midi
from tests.conftest import make_notes


def _stats(**overrides) -> MidiStats:
    base = {
        "ticks_per_beat": 480,
        "n_tracks": 2,
        "n_notes": 100,
        "duration_seconds": 30.0,
        "duration_beats": 50.0,
        "tempos": (600_000,),
        "time_signatures": ("4/4",),
        "programs": (0,),
        "channels": (0,),
        "pitch_min": 48,
        "pitch_max": 72,
        "unique_pitches": 12,
        "max_polyphony": 4,
        "has_percussion": False,
    }
    base.update(overrides)
    return MidiStats(**base)


def test_good_file_passes() -> None:
    assert check_stats(_stats(), CleanConfig()) == ()


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"n_notes": 3}, "too_few_notes"),
        ({"duration_seconds": 1.0}, "too_short"),
        ({"duration_seconds": 5000.0}, "too_long"),
        ({"max_polyphony": 30}, "too_polyphonic"),
        ({"unique_pitches": 2}, "too_few_pitches"),
        ({"ticks_per_beat": 48}, "low_resolution"),
        ({"n_tracks": 99}, "too_many_tracks"),
        ({"has_percussion": True}, "percussion"),
        ({"tempos": (100_000,)}, "tempo_out_of_range"),  # 600 BPM
    ],
)
def test_single_failure_reasons(override: dict, reason: str) -> None:
    assert reason in check_stats(_stats(**override), CleanConfig())


def test_empty_file_short_circuits() -> None:
    assert check_stats(_stats(n_notes=0), CleanConfig()) == ("empty",)


def test_multiple_reasons_accumulate() -> None:
    reasons = check_stats(_stats(n_notes=1, duration_seconds=0.5), CleanConfig())
    assert set(reasons) == {"too_few_notes", "too_short"}


def test_percussion_can_be_allowed() -> None:
    cfg = CleanConfig(allow_percussion=True)
    assert "percussion" not in check_stats(_stats(has_percussion=True), cfg)


def test_reason_text_covers_all_codes() -> None:
    for code in ("empty", "unreadable", "percussion", "too_polyphonic"):
        assert code in REASON_TEXT


def test_clean_file_accepts_valid(tmp_path: Path) -> None:
    notes = make_notes(tuple(range(60, 76)), step=480, duration=240)
    path = tmp_path / "iyi.mid"
    write_midi(notes, path)

    cfg = CleanConfig(min_notes=8, min_duration_s=0.1)
    result = clean_file(path, cfg)
    assert result.ok
    assert result.reasons == ()
    assert result.stats is not None


def test_clean_file_rejects_too_short(tmp_path: Path) -> None:
    path = tmp_path / "kisa.mid"
    write_midi(make_notes((60, 62)), path, )
    result = clean_file(path, CleanConfig(min_duration_s=10.0, min_notes=2))
    assert not result.ok
    assert "too_short" in result.reasons


def test_clean_file_handles_unreadable(tmp_path: Path) -> None:
    bad = tmp_path / "bozuk.mid"
    bad.write_bytes(b"xx")
    result = clean_file(bad)
    assert not result.ok
    assert result.reasons == ("unreadable",)


def test_clean_corpus_summary(tmp_path: Path) -> None:
    good = tmp_path / "a.mid"
    write_midi(make_notes(tuple(range(60, 80))), good)
    bad = tmp_path / "b.mid"
    write_midi(make_notes((60,)), bad)

    cfg = CleanConfig(min_notes=5, min_duration_s=0.1)
    summary = clean_corpus([good, bad], cfg)

    assert len(summary.accepted) == 1
    assert len(summary.rejected) == 1
    assert summary.reason_counts["too_few_notes"] == 1
    assert summary.total == 2
    assert summary.acceptance_rate == pytest.approx(0.5)


def test_empty_corpus_summary() -> None:
    summary = clean_corpus([])
    assert summary.total == 0
    assert summary.acceptance_rate == 0.0


def test_rejected_result_reason_text(tmp_path: Path) -> None:
    result = clean_corpus([], CleanConfig())
    assert result.reason_counts == {}

    bad = tmp_path / "c.mid"
    write_midi([NoteEvent(0, 480, 60, 90, 0, 0, 0)], bad)
    single = clean_file(bad, CleanConfig(min_notes=10))
    assert "nota sayısı yetersiz" in single.reason_text
