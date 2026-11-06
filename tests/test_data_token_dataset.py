"""Token veri seti testleri: diziler, ofsetler, kaydet/yükle, doğrulama."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ophelia.data.midi_io import NoteEvent, write_midi
from ophelia.data.token_dataset import (
    RoundtripReport,
    SplitArrays,
    TokenDataset,
    build_token_dataset,
    collect_split_paths,
    verify_roundtrip,
)
from ophelia.data.tokens import MidiTokenizer


def _write_piece(path: Path, base: int = 60, count: int = 8) -> Path:
    notes = [
        NoteEvent(index * 120, 120, base + (index % 5), 90, 0, 0, 0)
        for index in range(count)
    ]
    write_midi(notes, path)
    return path


@pytest.fixture
def pieces(tmp_path: Path) -> dict[str, list[Path]]:
    splits: dict[str, list[Path]] = {}
    for split, count in (("train", 3), ("val", 2), ("test", 1)):
        directory = tmp_path / "processed" / split
        directory.mkdir(parents=True)
        splits[split] = [
            _write_piece(directory / f"{split}{index}.mid", 60 + index)
            for index in range(count)
        ]
    return splits


def test_split_arrays_validation() -> None:
    tokens = np.asarray([1, 2, 3], dtype=np.int32)
    offsets = np.asarray([0, 3], dtype=np.int64)

    arrays = SplitArrays(tokens, offsets, ("a",))
    assert len(arrays) == 1
    assert arrays.total_tokens == 3
    assert list(arrays.sequence(0)) == [1, 2, 3]

    with pytest.raises(ValueError):
        SplitArrays(np.asarray([1], dtype=np.int64), offsets, ("a",))
    with pytest.raises(ValueError):
        SplitArrays(tokens, np.asarray([0], dtype=np.int64), ("a",))


def test_split_arrays_indexing() -> None:
    arrays = SplitArrays(
        np.asarray([1, 2, 3, 4], dtype=np.int32),
        np.asarray([0, 2, 4], dtype=np.int64),
        ("a", "b"),
    )
    assert list(arrays.sequence(1)) == [3, 4]
    assert list(arrays.sequence_of("b")) == [3, 4]
    with pytest.raises(IndexError):
        arrays.sequence(2)


def test_build_dataset_shapes(pieces: dict[str, list[Path]]) -> None:
    tokenizer = MidiTokenizer()
    dataset = build_token_dataset(pieces, tokenizer)

    assert dataset.counts() == {"train": 3, "val": 2, "test": 1}
    assert dataset.total_files == 6
    assert dataset.vocab_size == tokenizer.vocabulary.size
    assert dataset.total_tokens > 0

    train = dataset.splits["train"]
    assert train.tokens.dtype == np.int32
    assert train.offsets.dtype == np.int64
    assert train.offsets[0] == 0
    assert int(train.offsets[-1]) == train.total_tokens
    assert train.files[0].startswith("train")


def test_build_dataset_limit(pieces: dict[str, list[Path]]) -> None:
    dataset = build_token_dataset(pieces, MidiTokenizer(), limit_per_split=1)
    assert dataset.counts() == {"train": 1, "val": 1, "test": 1}


def test_sequences_start_and_end_with_bos_eos(pieces: dict[str, list[Path]]) -> None:
    tokenizer = MidiTokenizer()
    dataset = build_token_dataset(pieces, tokenizer)
    vocab = tokenizer.vocabulary

    train = dataset.splits["train"]
    first = vocab.to_tokens([int(value) for value in train.sequence(0)])
    assert first[0] == "<bos>"
    assert first[-1] == "<eos>"


def test_save_and_load_roundtrip(pieces: dict[str, list[Path]], tmp_path: Path) -> None:
    tokenizer = MidiTokenizer()
    dataset = build_token_dataset(pieces, tokenizer)
    target = dataset.save(tmp_path / "tokens", tokenizer)

    assert (target / "vocab.json").is_file()
    assert (target / "meta.json").is_file()
    assert (target / "train_tokens.npy").is_file()

    meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
    assert meta["counts"] == {"train": 3, "val": 2, "test": 1}
    assert meta["vocab_size"] == tokenizer.vocabulary.size
    assert meta["grid_ticks"] == tokenizer.config.grid_ticks
    assert len(meta["files"]["train"]) == 3

    loaded = TokenDataset.load(target, tokenizer)
    assert loaded.counts() == dataset.counts()
    assert loaded.splits["train"].files == dataset.splits["train"].files
    assert np.array_equal(
        loaded.splits["train"].tokens, dataset.splits["train"].tokens
    )


def test_load_missing_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TokenDataset.load(tmp_path / "yok", MidiTokenizer())


def test_verify_roundtrip_success(pieces: dict[str, list[Path]]) -> None:
    all_paths = [path for value in pieces.values() for path in value]
    report = verify_roundtrip(all_paths, MidiTokenizer())

    assert report.checked == 6
    assert report.matched == 6
    assert report.ok
    assert report.rate == 1.0


def test_verify_roundtrip_reports_failures(tmp_path: Path) -> None:
    bad = tmp_path / "bozuk.mid"
    bad.write_bytes(b"midi degil")

    report = verify_roundtrip([bad], MidiTokenizer())
    assert report.checked == 1
    assert report.matched == 0
    assert not report.ok
    assert "bozuk.mid" in report.failures[0][0]


def test_roundtrip_report_defaults() -> None:
    report = RoundtripReport()
    assert report.ok  # hiç kontrol yoksa hata da yok
    assert report.rate == 0.0


def test_collect_split_paths(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    (processed / "src").mkdir(parents=True)
    for name in ("a", "b", "c"):
        _write_piece(processed / "src" / f"{name}.mid")

    splits_file = tmp_path / "splits.json"
    splits_file.write_text(
        json.dumps(
            {
                "counts": {"train": 1, "val": 1, "test": 1},
                "total": 3,
                "splits": {"train": ["a"], "val": ["b"], "test": ["c"]},
            }
        ),
        encoding="utf-8",
    )

    result = collect_split_paths(splits_file, processed)
    assert [path.name for path in result["train"]] == ["a.mid"]
    assert [path.name for path in result["test"]] == ["c.mid"]


def test_collect_split_paths_skips_missing_entries(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_piece(processed / "a.mid")

    splits_file = tmp_path / "splits.json"
    splits_file.write_text(
        json.dumps({"splits": {"train": ["a", "yok"], "val": [], "test": []}}),
        encoding="utf-8",
    )

    result = collect_split_paths(splits_file, processed)
    assert len(result["train"]) == 1
    assert result["val"] == []
