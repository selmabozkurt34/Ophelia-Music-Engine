"""5. aşama — Beste modeli.

Decoder-only transformer, pencere veri yükleyici, eğitim döngüsü ve örnekleme.
"""

from ophelia.model.checkpoint import (
    CheckpointInfo,
    load_checkpoint,
    load_tokenizer,
    save_checkpoint,
)
from ophelia.model.config import ModelConfig, SampleConfig, TrainConfig
from ophelia.model.dataset import TokenWindows, WindowSpec
from ophelia.model.generate import (
    GenerationError,
    GenerationResult,
    generate_piece,
    generate_tokens,
    limit_polyphony,
    tokens_to_notes,
)
from ophelia.model.train import TrainResult, evaluate, lr_at, train
from ophelia.model.transformer import OpheliaGPT

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "SampleConfig",
    "OpheliaGPT",
    "TokenWindows",
    "WindowSpec",
    "train",
    "evaluate",
    "lr_at",
    "TrainResult",
    "save_checkpoint",
    "load_checkpoint",
    "load_tokenizer",
    "CheckpointInfo",
    "generate_tokens",
    "generate_piece",
    "tokens_to_notes",
    "limit_polyphony",
    "GenerationResult",
    "GenerationError",
]
