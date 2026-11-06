"""Tohum veri setini üretir: music21 korpusundaki Bach eserleri -> MIDI.

Neden bu kaynak?
    * Eserler J.S. Bach'a ait (ö. 1750) ve dünya çapında kamu malı.
    * Kodlamalar BSD-3 lisanslı music21 paketiyle dağıtılıyor.
    * Ticari kullanıma kapalı hiçbir koşul yok (bkz. ophelia/data/sources.py).

Kullanım:
    ./.venv/Scripts/python.exe scripts/fetch_corpus.py --limit 10        # deneme
    ./.venv/Scripts/python.exe scripts/fetch_corpus.py                   # tamamı
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ophelia.core.console import make_console  # noqa: E402
from ophelia.core.paths import MANIFESTS_DIR, RAW_DIR  # noqa: E402
from ophelia.data.manifest import (  # noqa: E402
    STAGE_RAW,
    FileRecord,
    Manifest,
    SourceRecord,
    sha256_file,
    utc_now,
)
from ophelia.data.sources import assert_usable  # noqa: E402

SOURCE_SLUG = "music21-bach-chorales"
console = make_console()


def fetch(limit: int | None, out_dir: Path) -> tuple[int, int]:
    """Korpusu MIDI'ye çevirir; (yazılan, atlanan) döner."""
    from music21 import corpus

    source = assert_usable(SOURCE_SLUG)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = list(corpus.getComposer("bach"))
    if limit is not None:
        paths = paths[:limit]

    manifest = Manifest.load(MANIFESTS_DIR)
    manifest.add_source(
        SourceRecord(
            slug=source.slug,
            name=source.name,
            license_id=source.license_id,
            url=source.url,
            attribution=source.license.attribution_text(source.name),
            retrieved_at=utc_now(),
            notes="; ".join(source.caveats),
        )
    )

    written = 0
    skipped = 0
    for index, corpus_path in enumerate(paths, start=1):
        target = out_dir / f"{Path(corpus_path.name).stem}.mid"
        if not target.is_file():
            try:
                score = corpus.parse(corpus_path)
                score.write("midi", fp=str(target))
                written += 1
            except Exception as exc:  # noqa: BLE001 - tek dosya hatası akışı durdurmasın
                console.print(f"[yellow]atlandı[/yellow] {corpus_path.name}: {exc}")
                skipped += 1
                continue

        # Dosya diskte varsa da kaydı garanti altına al (manifest yeniden kurulabilir).
        manifest.add_file(
            FileRecord(
                relpath=str(target.relative_to(RAW_DIR.parent)),
                sha256=sha256_file(target),
                size_bytes=target.stat().st_size,
                source=source.slug,
                license_id=source.license_id,
                stage=STAGE_RAW,
                added_at=utc_now(),
            )
        )

        if index % 25 == 0 or index == len(paths):
            console.print(f"  {index}/{len(paths)} işlendi ({written} yazıldı)")

    manifest.set_file_count(source.slug, len(manifest.files_of(source.slug, STAGE_RAW)))
    manifest.save()
    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="ilk N dosya")
    parser.add_argument(
        "--out",
        type=Path,
        default=RAW_DIR / SOURCE_SLUG,
        help="çıktı dizini (varsayılan: data/raw/music21-bach-chorales)",
    )
    args = parser.parse_args()

    console.rule("[bold]Tohum veri seti: Bach koralleri[/bold]")
    written, skipped = fetch(args.limit, args.out)
    existing = len(list(args.out.glob("*.mid")))
    console.print(
        f"[green]{written} yeni MIDI yazıldı[/green], {skipped} atlandı, "
        f"diskte {existing} dosya → {args.out}"
    )
    return 0 if existing else 1


if __name__ == "__main__":
    raise SystemExit(main())
