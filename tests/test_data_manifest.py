"""Manifest (lisans/köken kaydı) testleri."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ophelia.data.manifest import (
    ADDED,
    DUPLICATE,
    STAGE_PROCESSED,
    STAGE_RAW,
    UNCHANGED,
    UPDATED,
    FileRecord,
    Manifest,
    SourceRecord,
    sha256_file,
)


def _source(slug: str = "pdmx", license_id: str = "CC0-1.0") -> SourceRecord:
    return SourceRecord(
        slug=slug,
        name="Test kaynağı",
        license_id=license_id,
        url="https://example.org",
        attribution="Test",
        retrieved_at="2026-01-01T00:00:00Z",
    )


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    payload = b"ophelia"
    path = tmp_path / "x.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_add_source_is_idempotent() -> None:
    manifest = Manifest()
    manifest.add_source(_source())
    manifest.add_source(_source())
    assert len(manifest.sources) == 1


def test_conflicting_license_raises() -> None:
    manifest = Manifest()
    manifest.add_source(_source(license_id="CC0-1.0"))
    with pytest.raises(ValueError, match="farklı lisans"):
        manifest.add_source(_source(license_id="CC-BY-4.0"))


def test_add_file_requires_source() -> None:
    manifest = Manifest()
    with pytest.raises(KeyError):
        manifest.add_file(FileRecord("a.mid", "b" * 64, 1, "yok", "CC0-1.0"))


def test_duplicate_content_is_skipped(tmp_path: Path) -> None:
    manifest = Manifest()
    manifest.add_source(_source())
    record = FileRecord("a.mid", "b" * 64, 10, "pdmx", "CC0-1.0", STAGE_RAW)

    assert manifest.add_file(record) == ADDED
    duplicate = FileRecord("a2.mid", "b" * 64, 10, "pdmx", "CC0-1.0", STAGE_RAW)
    assert manifest.add_file(duplicate) == DUPLICATE
    assert manifest.has_hash("b" * 64)
    assert not manifest.has_hash("c" * 64)


def test_same_path_new_content_is_updated() -> None:
    manifest = Manifest()
    manifest.add_source(_source())
    manifest.add_file(FileRecord("a.mid", "1" * 64, 10, "pdmx", "CC0-1.0", STAGE_RAW))

    status = manifest.add_file(FileRecord("a.mid", "2" * 64, 11, "pdmx", "CC0-1.0", STAGE_RAW))
    assert status == UPDATED
    assert len(manifest.files) == 1
    assert manifest.files[0].sha256 == "2" * 64


def test_same_path_same_content_is_unchanged() -> None:
    manifest = Manifest()
    manifest.add_source(_source())
    record = FileRecord("a.mid", "3" * 64, 10, "pdmx", "CC0-1.0", STAGE_RAW)
    manifest.add_file(record)
    assert manifest.add_file(record) == UNCHANGED
    assert len(manifest.files) == 1


def test_find_by_path() -> None:
    manifest = Manifest()
    manifest.add_source(_source())
    manifest.add_file(FileRecord("a.mid", "4" * 64, 10, "pdmx", "CC0-1.0", STAGE_RAW))
    assert manifest.find_by_path("a.mid", STAGE_RAW) is not None
    assert manifest.find_by_path("a.mid", STAGE_PROCESSED) is None
    assert manifest.find_by_path("yok.mid") is None


def test_same_hash_different_stage_allowed() -> None:
    manifest = Manifest()
    manifest.add_source(_source())
    manifest.add_file(FileRecord("raw/a.mid", "d" * 64, 1, "pdmx", "CC0-1.0", STAGE_RAW))
    assert (
        manifest.add_file(
            FileRecord("processed/a.mid", "d" * 64, 1, "pdmx", "CC0-1.0", STAGE_PROCESSED)
        )
        == ADDED
    )


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    manifest = Manifest(tmp_path / "manifests")
    manifest.add_source(_source())
    manifest.add_file(
        FileRecord("raw/a.mid", "e" * 64, 42, "pdmx", "CC0-1.0", STAGE_RAW, "2026-01-01T00:00:00Z")
    )
    manifest.save()

    sources_payload = json.loads((tmp_path / "manifests" / "sources.json").read_text("utf-8"))
    assert sources_payload["sources"][0]["slug"] == "pdmx"

    reloaded = Manifest.load(tmp_path / "manifests")
    assert reloaded.sources["pdmx"].license_id == "CC0-1.0"
    assert len(reloaded.files) == 1
    assert reloaded.files[0].size_bytes == 42
    assert reloaded.license_of("pdmx") == "CC0-1.0"


def test_load_missing_manifest_is_empty(tmp_path: Path) -> None:
    manifest = Manifest.load(tmp_path / "yok")
    assert manifest.sources == {}
    assert manifest.files == []


def test_files_of_filters_by_source_and_stage() -> None:
    manifest = Manifest()
    manifest.add_source(_source())
    manifest.add_file(FileRecord("a", "1" * 64, 1, "pdmx", "CC0-1.0", STAGE_RAW))
    manifest.add_file(FileRecord("b", "2" * 64, 1, "pdmx", "CC0-1.0", STAGE_PROCESSED))

    assert len(manifest.files_of("pdmx")) == 2
    assert len(manifest.files_of("pdmx", STAGE_RAW)) == 1


def test_verify_detects_missing_and_corrupted(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    (data_root / "raw").mkdir(parents=True)
    payload = b"icerik"
    file_path = data_root / "raw" / "a.mid"
    file_path.write_bytes(payload)

    manifest = Manifest(tmp_path / "manifests")
    manifest.add_source(_source())
    manifest.add_file(
        FileRecord("raw/a.mid", sha256_file(file_path), len(payload), "pdmx", "CC0-1.0")
    )
    manifest.add_file(FileRecord("raw/yok.mid", "f" * 64, 1, "pdmx", "CC0-1.0"))

    result = manifest.verify(data_root)
    assert result["missing"] == ["raw/yok.mid"]
    assert result["corrupted"] == []

    file_path.write_bytes(b"degisti")
    result = manifest.verify(data_root)
    assert result["corrupted"] == ["raw/a.mid"]


def test_license_of_unknown_raises() -> None:
    with pytest.raises(KeyError):
        Manifest().license_of("pdmx")
