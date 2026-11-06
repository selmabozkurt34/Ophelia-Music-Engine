"""Eğitim döngüsü testleri (küçük sentetik veri, saniyeler sürer)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ophelia.data.token_dataset import SplitArrays
from ophelia.model.config import ModelConfig, TrainConfig
from ophelia.model.dataset import TokenWindows
from ophelia.model.train import evaluate, lr_at, train
from ophelia.model.transformer import OpheliaGPT


def _arrays(sequences: list[list[int]]) -> SplitArrays:
    tokens: list[int] = []
    offsets = [0]
    for sequence in sequences:
        tokens.extend(sequence)
        offsets.append(len(tokens))
    return SplitArrays(
        tokens=np.asarray(tokens, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        files=tuple(f"f{index}" for index in range(len(sequences))),
    )


def _tiny_config() -> ModelConfig:
    return ModelConfig(vocab_size=20, context=16, d_model=16, n_layer=1, n_head=2, dropout=0.0)


def _tiny_train_config(**overrides) -> TrainConfig:
    base = {
        "batch_size": 2,
        "grad_accum": 1,
        "max_steps": 3,
        "warmup_steps": 1,
        "eval_interval": 2,
        "eval_batches": 2,
        "log_every": 1,
        "checkpoint_every": 2,
        "threads": 1,
    }
    base.update(overrides)
    return TrainConfig(**base)


def test_lr_warmup_and_decay() -> None:
    config = TrainConfig(lr=1e-3, min_lr=1e-5, warmup_steps=10, max_steps=100)

    assert lr_at(0, config) < lr_at(5, config) < lr_at(9, config)
    assert lr_at(9, config) == pytest.approx(1e-3)
    assert lr_at(50, config) < lr_at(10, config)
    assert lr_at(100, config) == pytest.approx(1e-5, rel=0.05)
    assert lr_at(200, config) >= config.min_lr * 0.99


def test_train_writes_checkpoint_and_log(tmp_path: Path) -> None:
    train_arrays = _arrays([[index % 20 for index in range(60)] for _ in range(3)])
    val_arrays = _arrays([[index % 20 for index in range(40)]])

    model_config = _tiny_config()
    model = OpheliaGPT(model_config)
    train_config = _tiny_train_config()

    result = train(
        model,
        TokenWindows(train_arrays, model_config.context, pad_id=0),
        TokenWindows(val_arrays, model_config.context, pad_id=0),
        model_config,
        train_config,
        pad_id=0,
        out_dir=tmp_path / "run",
    )

    assert result.steps == 3
    assert result.final_train_loss is not None
    assert torch.isfinite(torch.tensor(result.final_train_loss))
    assert result.best_val_loss is not None
    assert result.best_checkpoint is not None
    assert Path(result.best_checkpoint).is_file()

    log = json.loads((tmp_path / "run" / "train_log.json").read_text(encoding="utf-8"))
    assert log["steps"] == 3
    assert log["model_config"]["vocab_size"] == 20
    assert log["history"]

    meta = json.loads((tmp_path / "run" / "best.json").read_text(encoding="utf-8"))
    assert meta["step"] == 3


def test_train_reduces_loss_on_repetitive_data(tmp_path: Path) -> None:
    pattern = [1, 2, 3, 4, 5, 6, 7, 8] * 20
    arrays = _arrays([pattern, pattern])

    model_config = ModelConfig(
        vocab_size=12, context=16, d_model=32, n_layer=2, n_head=2, dropout=0.0
    )
    model = OpheliaGPT(model_config)
    config = _tiny_train_config(batch_size=4, max_steps=25, lr=1e-2, eval_interval=10)

    result = train(
        model,
        TokenWindows(arrays, 16, pad_id=0),
        TokenWindows(arrays, 16, pad_id=0),
        model_config,
        config,
        pad_id=0,
        out_dir=tmp_path / "fit",
    )

    assert result.final_train_loss is not None
    assert result.final_train_loss < 6.0


def test_evaluate_returns_finite_loss() -> None:
    arrays = _arrays([[index % 20 for index in range(40)]])
    dataset = TokenWindows(arrays, 16, pad_id=0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    model = OpheliaGPT(_tiny_config())
    loss = evaluate(model, loader, max_batches=2, pad_id=0)
    assert np.isfinite(loss)


def test_train_config_validation() -> None:
    with pytest.raises(ValueError):
        TrainConfig(batch_size=0)
    with pytest.raises(ValueError):
        TrainConfig(lr=0)
    assert TrainConfig(batch_size=4, grad_accum=3).effective_batch == 12
