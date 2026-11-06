"""Proje genelinde tek biçimli günlükleme."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

from ophelia.core.paths import LOGS_DIR

_CONFIGURED = False


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Kök günlükleyiciyi yapılandırır (birden fazla çağrıda güvenli)."""
    global _CONFIGURED

    handlers: list[logging.Handler] = [
        RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    ]

    if log_file is not None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_file), encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=_CONFIGURED,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Modül günlükleyicisi döndürür."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
