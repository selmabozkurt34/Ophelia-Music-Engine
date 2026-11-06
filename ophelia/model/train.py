"""Eğitim döngüsü: AdamW + warmup/cosine, gradyan biriktirme, checkpoint.

CPU'da uzun iş: her ``checkpoint_every`` adımda disk durumu güncellenir, böylece
eğitim kesilse bile elde olan model kaybolmaz.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ophelia.core.logging import get_logger
from ophelia.model.checkpoint import save_checkpoint
from ophelia.model.config import ModelConfig, TrainConfig
from ophelia.model.dataset import TokenWindows
from ophelia.model.transformer import OpheliaGPT

log = get_logger(__name__)


@dataclass
class TrainResult:
    """Eğitim özeti."""

    steps: int = 0
    best_val_loss: float | None = None
    final_train_loss: float | None = None
    history: list[dict[str, float | int | None]] = field(default_factory=list)
    best_checkpoint: Path | None = None
    seconds: float = 0.0

    @property
    def seconds_per_step(self) -> float:
        return self.seconds / self.steps if self.steps else 0.0


def lr_at(step: int, config: TrainConfig) -> float:
    """Doğrusal warmup + kosinüs azalma."""
    warmup = max(1, config.warmup_steps)
    if step < warmup:
        return config.lr * (step + 1) / warmup

    span = max(1, config.max_steps - warmup)
    progress = min(1.0, (step - warmup) / span)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + (config.lr - config.min_lr) * cosine


def _infinite(loader: DataLoader):
    """DataLoader'ı sonsuz akışa çevirir (epoch sınırı yok)."""
    while True:
        yield from loader


@torch.no_grad()
def evaluate(
    model: OpheliaGPT,
    loader: DataLoader,
    *,
    max_batches: int,
    pad_id: int,
) -> float:
    """Doğrulama kaybı (bit başına değil, ortalama cross-entropy)."""
    model.eval()
    total = 0.0
    count = 0

    for index, (inputs, targets) in enumerate(loader):
        if index >= max_batches:
            break
        _, loss = model(inputs, targets, pad_id=pad_id)
        if loss is None:  # pragma: no cover - targets her zaman var
            continue
        total += float(loss.item())
        count += 1

    model.train()
    return total / count if count else float("nan")


def train(
    model: OpheliaGPT,
    train_windows: TokenWindows,
    val_windows: TokenWindows,
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    pad_id: int,
    out_dir: str | Path,
    on_step: object | None = None,
) -> TrainResult:
    """Modeli eğitir; en iyi checkpoint'i ``out_dir/best.pt`` olarak yazar."""
    torch.manual_seed(train_config.seed)
    torch.set_num_threads(train_config.threads)

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(
        train_windows,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(train_config.seed),
    )
    val_loader = torch.utils.data.DataLoader(
        val_windows,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        betas=(0.9, 0.95),
        weight_decay=train_config.weight_decay,
    )

    model.train()
    stream = _infinite(train_loader)
    result = TrainResult()
    started = time.perf_counter()
    running = 0.0
    running_count = 0

    for step in range(train_config.max_steps):
        lr = lr_at(step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for _ in range(train_config.grad_accum):
            inputs, targets = next(stream)
            _, loss = model(inputs, targets, pad_id=pad_id)
            if loss is None:  # pragma: no cover
                continue
            (loss / train_config.grad_accum).backward()
            step_loss += float(loss.item()) / train_config.grad_accum

        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

        result.steps = step + 1
        result.final_train_loss = step_loss
        running += step_loss
        running_count += 1

        if (step + 1) % train_config.log_every == 0:
            average = running / max(1, running_count)
            running, running_count = 0.0, 0
            elapsed = time.perf_counter() - started
            log.info(
                "adım %d/%d • kayıp %.4f • lr %.2e • %.1f sn",
                step + 1,
                train_config.max_steps,
                average,
                lr,
                elapsed,
            )
            result.history.append(
                {"step": step + 1, "train_loss": round(average, 5), "lr": round(lr, 8)}
            )
            if callable(on_step):
                on_step(step + 1, average)

        if (step + 1) % train_config.eval_interval == 0 or step + 1 == train_config.max_steps:
            val_loss = evaluate(
                model,
                val_loader,
                max_batches=train_config.eval_batches,
                pad_id=pad_id,
            )
            log.info("  doğrulama kaybı: %.4f", val_loss)
            result.history.append(
                {"step": step + 1, "val_loss": round(val_loss, 5), "lr": round(lr, 8)}
            )

            if result.best_val_loss is None or val_loss < result.best_val_loss:
                result.best_val_loss = val_loss
                result.best_checkpoint = save_checkpoint(
                    target_dir / "best.pt",
                    model,
                    model_config,
                    train_config=train_config,
                    step=step + 1,
                    best_val_loss=val_loss,
                    extra={"vocab": "vocab.json"},
                )

        if (step + 1) % train_config.checkpoint_every == 0:
            save_checkpoint(
                target_dir / "last.pt",
                model,
                model_config,
                train_config=train_config,
                step=step + 1,
                best_val_loss=result.best_val_loss,
            )

    result.seconds = time.perf_counter() - started

    if result.best_checkpoint is None:
        result.best_checkpoint = save_checkpoint(
            target_dir / "best.pt",
            model,
            model_config,
            train_config=train_config,
            step=result.steps,
            best_val_loss=result.final_train_loss,
        )

    (target_dir / "train_log.json").write_text(
        json.dumps(
            {
                "steps": result.steps,
                "seconds": round(result.seconds, 1),
                "seconds_per_step": round(result.seconds_per_step, 3),
                "best_val_loss": result.best_val_loss,
                "final_train_loss": result.final_train_loss,
                "model_config": model_config.to_dict(),
                "train_config": train_config.to_dict(),
                "history": result.history,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
