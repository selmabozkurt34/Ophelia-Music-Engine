"""Veri kaynağı ve dosya kayıt sistemi.

Her indirilen/türetilen MIDI dosyası, kaynağı ve lisansıyla birlikte kaydedilir.
Kayıtlar ``data/manifests/`` altında iki dosyada tutulur:

* ``sources.json`` — kaynak başına lisans, atıf ve indirme zamanı
* ``files.jsonl`` — dosya başına sha256, boyut, kaynak ve aşama (raw/processed)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from ophelia.core.paths import MANIFESTS_DIR

SOURCES_FILE = "sources.json"
FILES_FILE = "files.jsonl"

STAGE_RAW = "raw"
STAGE_PROCESSED = "processed"

# add_file sonuçları
ADDED = "added"
UPDATED = "updated"
UNCHANGED = "unchanged"
DUPLICATE = "duplicate"


def utc_now() -> str:
    """ISO-8601 UTC zaman damgası."""
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Dosyanın sha256 özeti (parça parça okur)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceRecord:
    """Bir kaynağın lisans ve köken kaydı."""

    slug: str
    name: str
    license_id: str
    url: str
    attribution: str = ""
    retrieved_at: str = field(default_factory=utc_now)
    file_count: int = 0
    notes: str = ""


@dataclass(frozen=True)
class FileRecord:
    """Tek bir dosyanın kaydı."""

    relpath: str
    sha256: str
    size_bytes: int
    source: str
    license_id: str
    stage: str = STAGE_RAW
    added_at: str = field(default_factory=utc_now)

    @property
    def key(self) -> str:
        """Tekilleştirme anahtarı: içerik özeti."""
        return self.sha256


class Manifest:
    """Kaynak + dosya kayıtlarının bellekteki temsili."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else MANIFESTS_DIR
        self.sources: dict[str, SourceRecord] = {}
        self.files: list[FileRecord] = []

    # --- yollar -------------------------------------------------------------
    @property
    def sources_path(self) -> Path:
        return self.root / SOURCES_FILE

    @property
    def files_path(self) -> Path:
        return self.root / FILES_FILE

    # --- kaynak -------------------------------------------------------------
    def add_source(self, record: SourceRecord) -> SourceRecord:
        """Kaynağı ekler; aynı slug farklı lisansla gelirse hata verir."""
        existing = self.sources.get(record.slug)
        if existing is not None:
            if existing.license_id != record.license_id:
                raise ValueError(
                    f"'{record.slug}' kaynağı farklı lisansla kayıtlı: "
                    f"{existing.license_id} != {record.license_id}"
                )
            return existing
        self.sources[record.slug] = record
        return record

    def set_file_count(self, slug: str, count: int) -> None:
        """Kaynağın veri setine giren dosya sayısını günceller."""
        record = self.sources[slug]
        if record.file_count != count:
            self.sources[slug] = replace(record, file_count=count)

    def license_of(self, slug: str) -> str:
        try:
            return self.sources[slug].license_id
        except KeyError as exc:
            raise KeyError(f"kaynak kayıtlı değil: {slug}") from exc

    # --- dosya --------------------------------------------------------------
    def has_hash(self, sha256: str, stage: str | None = None) -> bool:
        return any(
            record.sha256 == sha256 and (stage is None or record.stage == stage)
            for record in self.files
        )

    def add_file(self, record: FileRecord) -> str:
        """Dosyayı kaydeder ve ne olduğunu söyler.

        Anahtar ``(relpath, stage)``'dir: aynı yol yeniden üretildiğinde eski
        kayıt **değiştirilir** (yinelenen geçmiş kaydı birikmez). Farklı
        yollardaki aynı içerik ise ``DUPLICATE`` olarak bildirilir.

        Dönen değer: :data:`ADDED`, :data:`UPDATED`, :data:`UNCHANGED` ya da
        :data:`DUPLICATE`.
        """
        if record.source not in self.sources:
            raise KeyError(f"önce kaynağı ekleyin: {record.source}")

        for index, existing in enumerate(self.files):
            if existing.relpath == record.relpath and existing.stage == record.stage:
                if existing.sha256 == record.sha256 and existing.size_bytes == record.size_bytes:
                    return UNCHANGED
                self.files[index] = record
                return UPDATED

        if self.has_hash(record.sha256, stage=record.stage):
            return DUPLICATE
        self.files.append(record)
        return ADDED

    def find_by_path(self, relpath: str, stage: str | None = None) -> FileRecord | None:
        """Yola göre kayıt arar."""
        for record in self.files:
            if record.relpath == relpath and (stage is None or record.stage == stage):
                return record
        return None

    def files_of(self, slug: str, stage: str | None = None) -> tuple[FileRecord, ...]:
        return tuple(
            record
            for record in self.files
            if record.source == slug and (stage is None or record.stage == stage)
        )

    # --- kalıcılık ----------------------------------------------------------
    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

        payload = {
            "generated_at": utc_now(),
            "sources": [asdict(record) for record in sorted(self.sources.values(), key=lambda r: r.slug)],
        }
        self.sources_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        lines = [json.dumps(asdict(record), ensure_ascii=False) for record in self.files]
        self.files_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    @classmethod
    def load(cls, root: Path | None = None) -> Manifest:
        manifest = cls(root)
        if manifest.sources_path.is_file():
            payload = json.loads(manifest.sources_path.read_text(encoding="utf-8"))
            for entry in payload.get("sources", []):
                record = SourceRecord(**entry)
                manifest.sources[record.slug] = record

        if manifest.files_path.is_file():
            for line in manifest.files_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    manifest.files.append(FileRecord(**json.loads(line)))
        return manifest

    # --- doğrulama ----------------------------------------------------------
    def verify(self, data_root: Path) -> dict[str, list[str]]:
        """Kayıtlı dosyaların varlığını ve özetlerini denetler."""
        missing: list[str] = []
        corrupted: list[str] = []

        for record in self.files:
            path = data_root / record.relpath
            if not path.is_file():
                missing.append(record.relpath)
                continue
            if sha256_file(path) != record.sha256:
                corrupted.append(record.relpath)

        return {"missing": missing, "corrupted": corrupted}
