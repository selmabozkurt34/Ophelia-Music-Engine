"""Kalite raporu testleri."""

from __future__ import annotations

from pathlib import Path

from ophelia.data.midi_io import NoteEvent, write_midi
from ophelia.data.quality import DatasetReport, build_report, format_report


def _write_c_major_scale(path: Path, octave_base: int = 60) -> Path:
    scale = (0, 2, 4, 5, 7, 9, 11)
    notes = [
        NoteEvent(index * 480, 480, octave_base + step, 90, 0, 0, 0)
        for index, step in enumerate(scale)
    ]
    write_midi(notes, path)
    return path


def test_report_counts_and_totals(tmp_path: Path) -> None:
    first = _write_c_major_scale(tmp_path / "a.mid", 60)
    second = _write_c_major_scale(tmp_path / "b.mid", 72)

    report = build_report([first, second])

    assert report.n_files == 2
    assert report.n_failed == 0
    assert report.total_notes == 14
    assert report.mean_notes == 7.0
    assert report.total_seconds > 0
    assert report.mean_polyphony == 1.0
    assert report.mean_duration > 0


def test_report_tracks_time_signature_and_pitch(tmp_path: Path) -> None:
    path = _write_c_major_scale(tmp_path / "a.mid")
    report = build_report([path])

    assert report.time_signatures["4/4"] == 1
    assert report.pitch_histogram[60] == 1
    assert report.pitch_class_histogram[0] == 1  # C
    assert report.pitch_histogram.most_common(1)[0][0] >= 60


def test_report_estimates_key(tmp_path: Path) -> None:
    path = _write_c_major_scale(tmp_path / "a.mid")
    report = build_report([path])

    assert report.key_distribution, "tonalite tahmini üretilmeli"
    assert sum(report.key_distribution.values()) == 1


def test_report_handles_unreadable_files(tmp_path: Path) -> None:
    good = _write_c_major_scale(tmp_path / "a.mid")
    bad = tmp_path / "bozuk.mid"
    bad.write_bytes(b"midi degil")

    report = build_report([good, bad])

    assert report.n_files == 1
    assert report.n_failed == 1
    assert "bozuk.mid" in report.failed_files


def test_report_source_counting(tmp_path: Path) -> None:
    path = _write_c_major_scale(tmp_path / "a.mid")
    report = build_report([path], sources={"a.mid": "music21-bach-chorales"})
    assert report.sources["music21-bach-chorales"] == 1


def test_report_to_dict_is_json_safe(tmp_path: Path) -> None:
    import json

    path = _write_c_major_scale(tmp_path / "a.mid")
    payload = build_report([path]).to_dict()

    assert payload["n_files"] == 1
    assert isinstance(payload["top_keys"], list)
    json.dumps(payload)  # istisna atmamalı


def test_format_report_lines(tmp_path: Path) -> None:
    path = _write_c_major_scale(tmp_path / "a.mid")
    lines = format_report(build_report([path]))

    assert any("Dosya sayısı" in line for line in lines)
    assert any("Toplam nota" in line for line in lines)
    assert any("Tonalite" in line for line in lines)


def test_empty_report_defaults() -> None:
    report = DatasetReport()
    assert report.total_hours == 0.0
    assert report.mean_duration == 0.0
    assert report.mean_notes == 0.0
    assert report.notes_per_second == 0.0
    assert report.top_keys() == []


def test_empty_input_report() -> None:
    report = build_report([])
    assert report.n_files == 0
    assert format_report(report)
