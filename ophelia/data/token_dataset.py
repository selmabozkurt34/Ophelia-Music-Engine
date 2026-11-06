"""Tokenize edilmiş veri seti: bölüm başına bitişik ``int32`` dizi + ofsetler.

Eğitim (5. aşama) dosya dosya okumak yerine tek bir düz diziden dilim alır;
7,6 GB RAM'li bir makinede bu, veri yükleyicisini basit ve hızlı tutar.

Dizin yapısı (``data/tokens/``)::

    vocab.json           token sözlüğü (sıralı)
    meta.json            yapılandırma, sayılar, dosya listesi
    train_tokens.npy     int32, bitişik
    train_offsets.npy    int64, uzunluk = dosya sayısı + 1
    val_* , test_*
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ophelia.core.paths import TOKENS_DIR
from ophelia.data.manifest import utc_now
from ophelia.data.midi_io import NoteEvent, iter_notes, load_midi
from ophelia.data.split import SPLITS
from ophelia.data.tokens import MidiTokenizer, TokenizerConfig, tokens_equal


@dataclass
class RoundtripReport:
    """Gidiş-dönüş doğrulamasının sonucu."""

    checked: int = 0
    matched: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures and self.checked == self.matched

    @property
    def rate(self) -> float:
        return self.matched / self.checked if self.checked else 0.0


@dataclass
class SplitArrays:
    """Bir bölümün bitişik token dizisi ve ofsetleri."""

    tokens: np.ndarray
    offsets: np.ndarray
    files: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.tokens.dtype != np.int32:
            raise ValueError("token dizisi int32 olmalı")
        if len(self.offsets) != len(self.files) + 1:
            raise ValueError("ofset uzunluğu dosya sayısı + 1 olmalı")

    def __len__(self) -> int:
        return len(self.files)

    def sequence(self, index: int) -> np.ndarray:
        """``index``'inci dosyanın token dizisi."""
        if not 0 <= index < len(self.files):
            raise IndexError(f"dosya indeksi aralık dışında: {index}")
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return self.tokens[start:end]

    def sequence_of(self, name: str) -> np.ndarray:
        return self.sequence(self.files.index(name))

    @property
    def total_tokens(self) -> int:
        return int(len(self.tokens))


@dataclass
class TokenDataset:
    """Bölüm adı -> diziler."""

    splits: dict[str, SplitArrays] = field(default_factory=dict)
    config: TokenizerConfig = field(default_factory=TokenizerConfig)
    vocab_size: int = 0
    created_at: str = field(default_factory=utc_now)

    def counts(self) -> dict[str, int]:
        return {name: len(arrays) for name, arrays in self.splits.items()}

    def token_counts(self) -> dict[str, int]:
        return {name: arrays.total_tokens for name, arrays in self.splits.items()}

    @property
    def total_tokens(self) -> int:
        return sum(arrays.total_tokens for arrays in self.splits.values())

    @property
    def total_files(self) -> int:
        return sum(len(arrays) for arrays in self.splits.values())

    def meta(self) -> dict[str, object]:
        return {
            "created_at": self.created_at,
            "vocab_size": self.vocab_size,
            "grid_ticks": self.config.grid_ticks,
            "max_time_units": self.config.max_time_units,
            "max_duration_units": self.config.max_duration_units,
            "velocity_bins": self.config.velocity_bins,
            "counts": self.counts(),
            "token_counts": self.token_counts(),
            "files": {name: list(arrays.files) for name, arrays in self.splits.items()},
            "total_files": self.total_files,
            "total_tokens": self.total_tokens,
        }

    def save(self, directory: str | Path, tokenizer: MidiTokenizer) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)

        for name, arrays in self.splits.items():
            np.save(target / f"{name}_tokens.npy", arrays.tokens)
            np.save(target / f"{name}_offsets.npy", arrays.offsets)

        tokenizer.vocabulary.save(target / "vocab.json")
        (target / "meta.json").write_text(
            json.dumps(self.meta(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, directory: str | Path, tokenizer: MidiTokenizer) -> TokenDataset:
        root = Path(directory)
        meta_path = root / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"token veri seti bulunamadı: {meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        config = TokenizerConfig(
            grid_ticks=int(meta["grid_ticks"]),
            max_time_units=int(meta["max_time_units"]),
            max_duration_units=int(meta["max_duration_units"]),
            velocity_bins=meta.get("velocity_bins"),
        )

        splits: dict[str, SplitArrays] = {}
        files_by_split = meta.get("files", {})
        for name in meta.get("counts", {}):
            tokens = np.load(root / f"{name}_tokens.npy")
            offsets = np.load(root / f"{name}_offsets.npy")
            files = tuple(files_by_split.get(name, []))
            if len(files) != len(offsets) - 1:
                raise ValueError(f"{name}: dosya listesi ofsetlerle uyuşmuyor")
            splits[name] = SplitArrays(tokens, offsets, files)

        return cls(
            splits=splits,
            config=config,
            vocab_size=int(meta.get("vocab_size", 0)),
            created_at=str(meta.get("created_at", utc_now())),
        )


def collect_split_paths(
    splits_file: str | Path,
    processed_dir: str | Path,
) -> dict[str, list[Path]]:
    """``splits.json`` içindeki dosya adlarını gerçek MIDI yollarına çevirir."""
    payload = json.loads(Path(splits_file).read_text(encoding="utf-8"))
    by_stem: dict[str, Path] = {
        path.stem: path for path in Path(processed_dir).rglob("*.mid")
    }

    result: dict[str, list[Path]] = {}
    for split in SPLITS:
        names = payload.get("splits", {}).get(split, [])
        result[split] = [by_stem[name] for name in names if name in by_stem]
    return result


def build_token_dataset(
    paths_by_split: dict[str, list[Path]],
    tokenizer: MidiTokenizer,
    *,
    limit_per_split: int | None = None,
) -> TokenDataset:
    """Her bölümü token'lara çevirip bitişik dizilere paketler."""
    splits: dict[str, SplitArrays] = {}

    for name in SPLITS:
        paths = paths_by_split.get(name, [])
        if limit_per_split is not None:
            paths = paths[:limit_per_split]

        chunks: list[np.ndarray] = []
        offsets: list[int] = [0]
        files: list[str] = []
        cursor = 0

        for path in paths:
            ids = tokenizer.encode_ids(tokenizer.encode_file(path))
            array = np.asarray(ids, dtype=np.int32)
            chunks.append(array)
            cursor += int(array.size)
            offsets.append(cursor)
            files.append(Path(path).stem)

        tokens = (
            np.concatenate(chunks) if chunks else np.asarray([], dtype=np.int32)
        ).astype(np.int32, copy=False)
        splits[name] = SplitArrays(
            tokens=tokens,
            offsets=np.asarray(offsets, dtype=np.int64),
            files=tuple(files),
        )

    return TokenDataset(
        splits=splits,
        config=tokenizer.config,
        vocab_size=tokenizer.vocabulary.size,
    )


def verify_roundtrip(
    paths: list[Path] | tuple[Path, ...],
    tokenizer: MidiTokenizer,
) -> RoundtripReport:
    """Her dosyayı encode → decode edip notaların birebir aynı olduğunu doğrular."""
    report = RoundtripReport()

    for path in paths:
        report.checked += 1
        try:
            original = iter_notes(load_midi(path))
            rebuilt = tokenizer.decode_notes(tokenizer.encode_notes(original))
        except Exception as exc:  # noqa: BLE001 - raporlamak için geniş yakalama
            report.failures.append((Path(path).name, f"{type(exc).__name__}: {exc}"))
            continue

        if _same_notes(original, rebuilt):
            report.matched += 1
        else:
            report.failures.append((Path(path).name, "notalar farklı"))

    return report


def _same_notes(left: list[NoteEvent], right: list[NoteEvent]) -> bool:
    """Karşılaştırma: kanal/program farkı olmadan içerik eşitliği.

    Token temsili kanal/program bilgisini taşır; ancak normalizasyon sonrası
    tüm notalar tek kanaldadır. Yine de tam karşılaştırma yapılır.
    """
    return tokens_equal(left, right)


def summarize_paths(paths_by_split: dict[str, list[Path]]) -> str:
    """Kısa özet satırı (log/CLI)."""
    parts = [f"{name}={len(paths)}" for name, paths in paths_by_split.items()]
    return " • ".join(parts)


DEFAULT_OUTPUT_DIR = TOKENS_DIR
