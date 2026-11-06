"""Veri seti boru hattı: tara → temizle → normalize et → böl → raporla.

Boru hattı ``data/raw`` altındaki dosyaları işler, ``data/processed`` altına
kanonik kopyalarını yazar ve manifest ile bölünmeleri günceller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ophelia.core.console import make_console
from ophelia.core.logging import get_logger
from ophelia.core.paths import (
    MANIFESTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SPLITS_DIR,
)
from ophelia.data.clean import CleanConfig, clean_file
from ophelia.data.manifest import (
    DUPLICATE,
    STAGE_PROCESSED,
    FileRecord,
    Manifest,
    SourceRecord,
    sha256_file,
    utc_now,
)
from ophelia.data.normalize import NormalizeConfig, normalize_file
from ophelia.data.quality import DatasetReport, build_report
from ophelia.data.sources import assert_usable, get_source
from ophelia.data.split import SplitRatios, assign_splits

MIDI_SUFFIXES = (".mid", ".midi", ".MID")

log = get_logger(__name__)
console = make_console()


@dataclass
class PipelineConfig:
    """Boru hattı parametreleri.

    Not: bölütleme dosya kökü üzerinden yapılır. Aynı eserin farklı
    aranjmanları farklı dosyalar olarak girerse bölümler arasında sızıntı
    olabilir; bu durumda bölütleme açık bir grup eşlemesiyle yapılmalıdır.
    """

    source_slug: str
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    splits_dir: Path = SPLITS_DIR
    manifests_dir: Path = MANIFESTS_DIR
    clean: CleanConfig = field(default_factory=CleanConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    ratios: SplitRatios = field(default_factory=SplitRatios)
    seed: int = 42
    limit: int | None = None


@dataclass
class PipelineResult:
    """Boru hattı çıktısı."""

    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    written: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    rejected_files: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)
    report: DatasetReport | None = None
    processed_files: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def find_midi_files(directory: str | Path) -> list[Path]:
    """Dizin altındaki MIDI dosyalarını (alt dizinler dahil) sıralı bulur."""
    root = Path(directory)
    if not root.is_dir():
        return []
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in MIDI_SUFFIXES
    ]
    return sorted(files)


def _source_record(slug: str) -> SourceRecord:
    source = get_source(slug)
    return SourceRecord(
        slug=source.slug,
        name=source.name,
        license_id=source.license_id,
        url=source.url,
        attribution=source.license.attribution_text(source.name),
        notes="; ".join(source.caveats) if source.caveats else "",
    )


def build_dataset(config: PipelineConfig) -> PipelineResult:
    """Tüm veri hattını çalıştırır."""
    # 1) Lisans kapısı: onaysız/kapalı kaynakla çalışmayı reddet
    source = assert_usable(config.source_slug)

    raw_files = find_midi_files(config.raw_dir)
    if config.limit is not None:
        raw_files = raw_files[: config.limit]

    result = PipelineResult()
    if not raw_files:
        result.warnings.append(f"{config.raw_dir} altında MIDI dosyası bulunamadı")
        return result

    log.info("%s: %d dosya işlenecek", source.name, len(raw_files))

    manifest = Manifest.load(config.manifests_dir)
    manifest.add_source(_source_record(source.slug))

    # 2) Temizleme + normalizasyon + manifest
    raw_hashes: dict[str, str] = {}
    for raw_path in raw_files:
        raw_hash = sha256_file(raw_path)
        if raw_hash in raw_hashes:
            result.duplicates += 1
            continue
        raw_hashes[raw_hash] = raw_path.name

        result.accepted += 1
        clean_result = clean_file(raw_path, config.clean)
        if not clean_result.ok:
            result.accepted -= 1
            result.rejected += 1
            result.rejected_files.append((raw_path.name, clean_result.reasons))
            for reason in clean_result.reasons:
                result.reason_counts[reason] = result.reason_counts.get(reason, 0) + 1
            continue

        target = config.processed_dir / source.slug / f"{raw_path.stem}.mid"
        normalize_file(raw_path, target, config.normalize)

        status = manifest.add_file(
            FileRecord(
                relpath=str(target.relative_to(config.processed_dir.parent)),
                sha256=sha256_file(target),
                size_bytes=target.stat().st_size,
                source=source.slug,
                license_id=source.license_id,
                stage=STAGE_PROCESSED,
                added_at=utc_now(),
            )
        )

        if status == DUPLICATE:
            # Normalizasyon sonrası başka bir dosyayla birebir aynı: tekrar say.
            target.unlink(missing_ok=True)
            result.duplicates += 1
            continue

        result.written += 1
        result.processed_files.append(target)

    # 3) Bölütleme (dosya adı köküne göre; aynı içerik zaten tekilleştirildi)
    keys = [path.stem for path in result.processed_files]
    split_result = assign_splits(keys, ratios=config.ratios, seed=config.seed)
    split_result.save(config.splits_dir / "splits.json")
    result.splits = split_result.counts

    # 4) Kalite raporu (bölünmüş dosyalar üzerinde)
    result.report = build_report(tuple(result.processed_files))
    manifest.set_file_count(source.slug, result.written)

    manifest.save()
    return result
