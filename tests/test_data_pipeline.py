"""Boru hattı uçtan uca testleri (sentetik veri, ağ yok)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ophelia.data.clean import CleanConfig
from ophelia.data.manifest import STAGE_PROCESSED, Manifest
from ophelia.data.midi_io import NoteEvent, write_midi
from ophelia.data.pipeline import PipelineConfig, build_dataset, find_midi_files
from ophelia.data.split import SplitRatios


def _write_piece(path: Path, base_pitch: int, n_notes: int = 40) -> Path:
    """Kabul eşiklerini geçecek uzunlukta basit bir melodi yazar."""
    notes = [
        NoteEvent(index * 120, 120, base_pitch + (index % 12), 90, 0, 0, 0)
        for index in range(n_notes)
    ]
    write_midi(notes, path)
    return path


@pytest.fixture
def pipeline_dirs(tmp_path: Path) -> dict[str, Path]:
    dirs = {
        "raw_dir": tmp_path / "raw",
        "processed_dir": tmp_path / "processed",
        "splits_dir": tmp_path / "splits",
        "manifests_dir": tmp_path / "manifests",
    }
    for key in ("raw_dir", "processed_dir", "splits_dir", "manifests_dir"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture
def populated_raw(pipeline_dirs: dict[str, Path]) -> dict[str, Path]:
    source_dir = pipeline_dirs["raw_dir"] / "music21-bach-chorales"
    source_dir.mkdir(parents=True, exist_ok=True)
    for index in range(12):
        # Her dosya farklı içerik: taban perde ve nota sayısı birlikte değişir
        _write_piece(source_dir / f"eser_{index:03d}.mid", 55 + index, n_notes=40 + index)
    return pipeline_dirs


def test_find_midi_files(populated_raw: dict[str, Path]) -> None:
    files = find_midi_files(populated_raw["raw_dir"])
    assert len(files) == 12
    assert all(path.suffix == ".mid" for path in files)


def test_find_midi_files_missing_dir(tmp_path: Path) -> None:
    assert find_midi_files(tmp_path / "yok") == []


def test_pipeline_end_to_end(populated_raw: dict[str, Path]) -> None:
    config = PipelineConfig(
        source_slug="music21-bach-chorales",
        clean=CleanConfig(min_notes=10, min_duration_s=0.1),
        ratios=SplitRatios(0.75, 0.15, 0.1),
        seed=42,
        **populated_raw,
    )
    result = build_dataset(config)

    assert result.written == 12
    assert result.rejected == 0
    assert result.duplicates == 0
    assert sum(result.splits.values()) == 12
    assert result.report is not None
    assert result.report.n_files == 12

    processed = list((populated_raw["processed_dir"] / "music21-bach-chorales").glob("*.mid"))
    assert len(processed) == 12

    splits_payload = json.loads((populated_raw["splits_dir"] / "splits.json").read_text("utf-8"))
    assert splits_payload["total"] == 12

    manifest = Manifest.load(populated_raw["manifests_dir"])
    assert "music21-bach-chorales" in manifest.sources
    assert len(manifest.files_of("music21-bach-chorales", STAGE_PROCESSED)) == 12


def test_pipeline_rejects_noncommercial_source(populated_raw: dict[str, Path]) -> None:
    config = PipelineConfig(source_slug="maestro", **populated_raw)
    with pytest.raises(PermissionError):
        build_dataset(config)


def test_pipeline_rejects_unknown_source(populated_raw: dict[str, Path]) -> None:
    config = PipelineConfig(source_slug="olmayan", **populated_raw)
    with pytest.raises(ValueError):
        build_dataset(config)


def test_pipeline_warns_on_empty_raw(pipeline_dirs: dict[str, Path]) -> None:
    config = PipelineConfig(source_slug="music21-bach-chorales", **pipeline_dirs)
    result = build_dataset(config)
    assert result.written == 0
    assert result.warnings


def test_pipeline_detects_duplicates(pipeline_dirs: dict[str, Path]) -> None:
    source_dir = pipeline_dirs["raw_dir"] / "music21-bach-chorales"
    source_dir.mkdir(parents=True, exist_ok=True)
    first = _write_piece(source_dir / "a.mid", 60)
    (source_dir / "a_kopya.mid").write_bytes(first.read_bytes())

    config = PipelineConfig(
        source_slug="music21-bach-chorales",
        clean=CleanConfig(min_notes=5, min_duration_s=0.1),
        **pipeline_dirs,
    )
    result = build_dataset(config)

    assert result.duplicates == 1
    assert result.written == 1


def test_pipeline_rejects_low_quality(pipeline_dirs: dict[str, Path]) -> None:
    source_dir = pipeline_dirs["raw_dir"] / "music21-bach-chorales"
    source_dir.mkdir(parents=True, exist_ok=True)
    write_midi([NoteEvent(0, 120, 60, 90, 0, 0, 0)], source_dir / "kisa.mid")

    config = PipelineConfig(
        source_slug="music21-bach-chorales",
        clean=CleanConfig(min_notes=32),
        **pipeline_dirs,
    )
    result = build_dataset(config)

    assert result.written == 0
    assert result.rejected == 1
    assert "too_few_notes" in result.reason_counts


def test_pipeline_limit(populated_raw: dict[str, Path]) -> None:
    config = PipelineConfig(
        source_slug="music21-bach-chorales",
        clean=CleanConfig(min_notes=10, min_duration_s=0.1),
        limit=5,
        **populated_raw,
    )
    result = build_dataset(config)
    assert result.written == 5


def test_manifest_verifies_after_build(populated_raw: dict[str, Path]) -> None:
    config = PipelineConfig(
        source_slug="music21-bach-chorales",
        clean=CleanConfig(min_notes=10, min_duration_s=0.1),
        **populated_raw,
    )
    build_dataset(config)

    manifest = Manifest.load(populated_raw["manifests_dir"])
    data_root = populated_raw["processed_dir"].parent
    result = manifest.verify(data_root)
    assert result["missing"] == []
    assert result["corrupted"] == []
