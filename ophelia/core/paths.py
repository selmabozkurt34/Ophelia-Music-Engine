"""Proje içindeki standart dizinlerin tek kaynağı.

Kod hiçbir yerde göreli yol hesaplamaz; her modül bu sabitleri kullanır.
"""

from __future__ import annotations

from pathlib import Path

# ophelia/core/paths.py -> ophelia/ -> proje kökü
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# Veri hattı aşamaları (3. aşama)
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SPLITS_DIR: Path = DATA_DIR / "splits"
MANIFESTS_DIR: Path = DATA_DIR / "manifests"
TOKENS_DIR: Path = DATA_DIR / "tokens"

# Çıktı türleri (7-9. aşamalar)
MIDI_OUT_DIR: Path = OUTPUTS_DIR / "midi"
WAV_OUT_DIR: Path = OUTPUTS_DIR / "wav"
# Notasyon çıktısı: MusicXML dosyaları (düzenlenebilir nota).
SCORE_OUT_DIR: Path = OUTPUTS_DIR / "scores"

DEFAULT_CONFIG_FILE: Path = CONFIG_DIR / "default.yaml"


def ensure_dirs() -> None:
    """Çalışma zamanında gerekli tüm dizinleri oluşturur."""
    for directory in (
        CONFIG_DIR,
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        SPLITS_DIR,
        MANIFESTS_DIR,
        TOKENS_DIR,
        MODELS_DIR,
        MIDI_OUT_DIR,
        WAV_OUT_DIR,
        SCORE_OUT_DIR,
        LOGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
