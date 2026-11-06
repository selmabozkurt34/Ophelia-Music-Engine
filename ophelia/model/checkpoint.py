"""Model kaydı ve geri yükleme.

Checkpoint tek bir ``.pt`` dosyasıdır: model ağırlıkları + mimari + eğitim
özeti. Sözlük ayrı ``vocab.json`` olarak yazılır (üretimde gerekli).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ophelia.data.tokens import MidiTokenizer, TokenizerConfig, Vocabulary
from ophelia.model.config import ModelConfig, TrainConfig
from ophelia.model.transformer import OpheliaGPT


@dataclass
class CheckpointInfo:
    """Checkpoint meta bilgisi."""

    path: Path
    model_config: ModelConfig
    train_config: TrainConfig | None = None
    step: int = 0
    best_val_loss: float | None = None
    extra: dict[str, object] = field(default_factory=dict)


def save_checkpoint(
    path: str | Path,
    model: OpheliaGPT,
    model_config: ModelConfig,
    *,
    train_config: TrainConfig | None = None,
    step: int = 0,
    best_val_loss: float | None = None,
    extra: dict[str, object] | None = None,
) -> Path:
    """Modeli diske yazar; dizin yoksa oluşturur."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "model_config": model_config.to_dict(),
        "state_dict": model.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
    }
    if train_config is not None:
        payload["train_config"] = train_config.to_dict()
    if extra:
        payload["extra"] = extra

    torch.save(payload, target)

    meta = {
        "model_config": model_config.to_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "params": model.num_params,
    }
    target.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def load_tokenizer(checkpoint_path: str | Path) -> MidiTokenizer:
    """Checkpoint dizininden sözlüğü ve tokenizer ayarlarını yükler.

    Üretim, modelin eğitildiği sözlükle **birebir aynı** tokenizer'ı kullanmak
    zorundadır; ayarlar (ızgara, sınırlar, velocity binleri) ``dataset_meta.json``
    içinden okunur. Eksik sözlük sessizce varsayılana düşmez, hata verir.
    """
    directory = Path(checkpoint_path).parent
    vocab_file = directory / "vocab.json"
    if not vocab_file.is_file():
        raise FileNotFoundError(f"sözlük bulunamadı: {vocab_file}")

    meta_file = directory / "dataset_meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
    tokenizer = MidiTokenizer(
        TokenizerConfig(
            grid_ticks=int(meta.get("grid_ticks", 120)),
            max_time_units=int(meta.get("max_time_units", 1024)),
            max_duration_units=int(meta.get("max_duration_units", 1024)),
            velocity_bins=meta.get("velocity_bins"),
        )
    )
    tokenizer.vocabulary = Vocabulary.load(vocab_file)
    return tokenizer


def load_checkpoint(path: str | Path) -> tuple[OpheliaGPT, CheckpointInfo]:
    """Checkpoint'i okur ve modeli kurar."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint bulunamadı: {source}")

    payload = torch.load(source, map_location="cpu", weights_only=False)
    model_config = ModelConfig.from_dict(payload["model_config"])
    model = OpheliaGPT(model_config)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    train_payload = payload.get("train_config")
    info = CheckpointInfo(
        path=source,
        model_config=model_config,
        train_config=TrainConfig.from_dict(train_payload) if train_payload else None,
        step=int(payload.get("step", 0)),
        best_val_loss=payload.get("best_val_loss"),
        extra=dict(payload.get("extra") or {}),
    )
    return model, info
