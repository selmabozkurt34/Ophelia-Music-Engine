"""YAML tabanlı yapılandırma yükleyici.

Tüm alt sistemler (veri, model, ses) parametrelerini buradan okur; böylece
deney ayarları kod içine gömülmez.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ophelia.core.paths import DEFAULT_CONFIG_FILE


@dataclass
class ProjectConfig:
    name: str = "Ophelia Music Engine"
    version: str = "0.1.0"
    seed: int = 42


@dataclass
class MidiConfig:
    ticks_per_beat: int = 480
    tempo_bpm: float = 100.0
    time_signature: tuple[int, int] = (4, 4)
    key: str = "C major"
    velocity_min: int = 40
    velocity_max: int = 110


@dataclass
class AudioConfig:
    sample_rate: int = 44100
    bit_depth: int = 16
    channels: int = 2


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    midi: MidiConfig = field(default_factory=MidiConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Yapılandırmayı diskten okur; dosya yoksa varsayılanları döndürür."""
        config_path = Path(path) if path is not None else DEFAULT_CONFIG_FILE
        if not config_path.is_file():
            return cls()

        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        midi_raw = dict(data.get("midi", {}))
        if "time_signature" in midi_raw:
            midi_raw["time_signature"] = tuple(midi_raw["time_signature"])

        return cls(
            project=ProjectConfig(**data.get("project", {})),
            midi=MidiConfig(**midi_raw),
            audio=AudioConfig(**data.get("audio", {})),
            raw=data,
        )


def load_config(path: str | Path | None = None) -> Config:
    """Kısayol: :meth:`Config.load`."""
    return Config.load(path)
