"""``ophelia model ...`` komutları ve checkpoint yükleme testleri."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from ophelia.cli.main import app
from ophelia.data.tokens import MidiTokenizer
from ophelia.model.checkpoint import load_tokenizer, save_checkpoint
from ophelia.model.config import ModelConfig
from ophelia.model.transformer import OpheliaGPT
from typer.testing import CliRunner

runner = CliRunner()


def _checkpoint_dir(tmp_path: Path, *, write_vocab: bool = True) -> Path:
    """Küçük bir checkpoint dizini kurar (sözlük + veri künyesi)."""
    tokenizer = MidiTokenizer()
    config = ModelConfig(
        vocab_size=tokenizer.vocabulary.size,
        context=32,
        d_model=16,
        n_layer=1,
        n_head=2,
        dropout=0.0,
    )
    torch.manual_seed(0)
    model = OpheliaGPT(config)
    target = tmp_path / "ophelia-test"
    save_checkpoint(target / "best.pt", model, config, step=7, best_val_loss=1.5)

    if write_vocab:
        tokenizer.vocabulary.save(target / "vocab.json")
        (target / "dataset_meta.json").write_text(
            json.dumps({"grid_ticks": 120, "max_time_units": 1024, "velocity_bins": None}),
            encoding="utf-8",
        )
    return target / "best.pt"


def test_model_info_reports_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint_dir(tmp_path)
    result = runner.invoke(app, ["model", "info", "--model", str(checkpoint)])

    assert result.exit_code == 0, result.output
    assert "Adım       : 7" in result.output
    assert "Val kaybı  : 1.5" in result.output


def test_model_info_without_checkpoint(tmp_path: Path) -> None:
    result = runner.invoke(app, ["model", "info", "--model", str(tmp_path / "yok.pt")])
    assert result.exit_code == 1
    assert "Checkpoint bulunamadı" in result.output


def test_model_generate_requires_vocab(tmp_path: Path) -> None:
    checkpoint = _checkpoint_dir(tmp_path, write_vocab=False)
    result = runner.invoke(app, ["model", "generate", "--model", str(checkpoint)])

    assert result.exit_code == 1
    assert "sözlük bulunamadı" in result.output


def test_model_generate_rejects_vocab_mismatch(tmp_path: Path) -> None:
    checkpoint = _checkpoint_dir(tmp_path)
    # Farklı yapılandırmayla üretilmiş sözlük: sessizce yanlış MIDI yazmamalı.
    smaller = MidiTokenizer()
    smaller.config = type(smaller.config)(velocity_bins=16)
    smaller.vocabulary = type(smaller.vocabulary)(
        tuple(token for token in smaller.vocabulary.tokens if not token.startswith("VELOCITY_"))
        + tuple(f"VELOCITY_{value}" for value in range(16))
    )
    smaller.vocabulary.save(checkpoint.parent / "vocab.json")

    result = runner.invoke(app, ["model", "generate", "--model", str(checkpoint)])
    assert result.exit_code == 1
    assert "uyuşmuyor" in result.output


def test_load_tokenizer_reads_settings(tmp_path: Path) -> None:
    checkpoint = _checkpoint_dir(tmp_path)
    (checkpoint.parent / "dataset_meta.json").write_text(
        json.dumps({"grid_ticks": 240, "max_duration_units": 64, "velocity_bins": 32}),
        encoding="utf-8",
    )

    tokenizer = load_tokenizer(checkpoint)
    assert tokenizer.config.grid_ticks == 240
    assert tokenizer.config.max_duration_units == 64
    assert tokenizer.config.velocity_bins == 32
    assert tokenizer.vocabulary.size > 0


def test_load_tokenizer_without_vocab(tmp_path: Path) -> None:
    checkpoint = _checkpoint_dir(tmp_path, write_vocab=False)
    with pytest.raises(FileNotFoundError):
        load_tokenizer(checkpoint)
