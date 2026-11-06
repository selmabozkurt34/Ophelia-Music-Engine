"""Veri setini üretir: temizleme + normalizasyon + bölütleme + rapor.

Kullanım:
    ./.venv/Scripts/python.exe scripts/build_dataset.py --source music21-bach-chorales
    ./.venv/Scripts/python.exe scripts/build_dataset.py --source music21-bach-chorales --limit 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ophelia.core.console import make_console  # noqa: E402
from ophelia.data.clean import REASON_TEXT, CleanConfig  # noqa: E402
from ophelia.data.pipeline import PipelineConfig, build_dataset  # noqa: E402
from ophelia.data.quality import format_report  # noqa: E402
from ophelia.data.split import SplitRatios  # noqa: E402
from rich.table import Table  # noqa: E402

console = make_console()


def _print_result(result) -> None:  # noqa: ANN001 - PipelineResult
    table = Table(title="Boru hattı", title_justify="left")
    table.add_column("Adım", style="cyan")
    table.add_column("Sonuç", justify="right")
    table.add_row("Kabul edilen", str(result.accepted))
    table.add_row("Elenen", str(result.rejected))
    table.add_row("Tekrar (yinelenen)", str(result.duplicates))
    table.add_row("Yazılan", str(result.written))
    console.print(table)

    if result.reason_counts:
        reasons = Table(title="Eleme sebepleri", title_justify="left")
        reasons.add_column("Sebep", style="cyan")
        reasons.add_column("Dosya", justify="right")
        for code, count in sorted(result.reason_counts.items(), key=lambda kv: -kv[1]):
            reasons.add_row(REASON_TEXT.get(code, code), str(count))
        console.print(reasons)

    if result.splits:
        console.print(
            "Bölünme: "
            + " • ".join(f"{name}={count}" for name, count in result.splits.items())
        )

    if result.report is not None:
        console.rule("[bold]Veri seti kalitesi[/bold]")
        for line in format_report(result.report):
            console.print(f"  {line}")

    for warning in result.warnings:
        console.print(f"[yellow]uyarı:[/yellow] {warning}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="music21-bach-chorales", help="kaynak slug")
    parser.add_argument("--limit", type=int, default=None, help="ilk N dosya")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--min-notes", type=int, default=32)
    parser.add_argument("--max-polyphony", type=int, default=12)
    args = parser.parse_args()

    config = PipelineConfig(
        source_slug=args.source,
        clean=CleanConfig(min_notes=args.min_notes, max_polyphony=args.max_polyphony),
        ratios=SplitRatios(args.train, args.val, args.test),
        seed=args.seed,
        limit=args.limit,
    )

    try:
        result = build_dataset(config)
    except (PermissionError, ValueError) as exc:
        console.print(f"[red]Hata:[/red] {exc}")
        return 1

    _print_result(result)
    return 0 if result.written else 1


if __name__ == "__main__":
    raise SystemExit(main())
