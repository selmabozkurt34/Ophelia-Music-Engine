"""Veri setini token'lara çevirir ve kayıpsızlığı doğrular.

Kullanım:
    ./.venv/Scripts/python.exe scripts/tokenize_dataset.py
    ./.venv/Scripts/python.exe scripts/tokenize_dataset.py --limit 20 --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ophelia.core.console import make_console  # noqa: E402
from ophelia.core.paths import PROCESSED_DIR, SPLITS_DIR, TOKENS_DIR  # noqa: E402
from ophelia.data.token_dataset import (  # noqa: E402
    build_token_dataset,
    collect_split_paths,
    verify_roundtrip,
)
from ophelia.data.tokens import MidiTokenizer, TokenizerConfig  # noqa: E402
from rich.table import Table  # noqa: E402

console = make_console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=TOKENS_DIR)
    parser.add_argument("--grid", type=int, default=120, help="ızgara (tick)")
    parser.add_argument("--limit", type=int, default=None, help="bölüm başına ilk N dosya")
    parser.add_argument("--verify", action="store_true", help="gidiş-dönüşü doğrula")
    args = parser.parse_args()

    splits_file = SPLITS_DIR / "splits.json"
    if not splits_file.is_file():
        console.print("[red]splits.json yok.[/red] Önce: ophelia data build")
        return 1

    tokenizer = MidiTokenizer(TokenizerConfig(grid_ticks=args.grid))
    paths = collect_split_paths(splits_file, PROCESSED_DIR)
    total_files = sum(len(value) for value in paths.values())
    if not total_files:
        console.print("[red]İşlenmiş MIDI bulunamadı.[/red]")
        return 1

    console.rule("[bold]Tokenizasyon[/bold]")
    console.print(
        f"  {total_files} dosya • ızgara {args.grid} tick • sözlük {tokenizer.vocabulary.size} token"
    )

    dataset = build_token_dataset(paths, tokenizer, limit_per_split=args.limit)
    target = dataset.save(args.out, tokenizer)

    table = Table(title="Token veri seti", title_justify="left")
    table.add_column("Bölüm", style="cyan")
    table.add_column("Dosya", justify="right")
    table.add_column("Token", justify="right")
    table.add_column("Ortalama", justify="right")
    for name, arrays in dataset.splits.items():
        average = arrays.total_tokens / len(arrays) if len(arrays) else 0
        table.add_row(name, str(len(arrays)), f"{arrays.total_tokens:,}", f"{average:,.0f}")
    console.print(table)
    console.print(f"  toplam {dataset.total_tokens:,} token → {target}")

    if args.verify:
        console.rule("[bold]Gidiş-dönüş doğrulaması[/bold]")
        all_paths = [path for paths_list in paths.values() for path in paths_list]
        if args.limit is not None:
            all_paths = [path for value in paths.values() for path in value[: args.limit]]
        report = verify_roundtrip(all_paths, tokenizer)
        console.print(
            f"  {report.matched}/{report.checked} dosya kayıpsız "
            f"(%{report.rate * 100:.1f})"
        )
        for name, detail in report.failures[:5]:
            console.print(f"  [red]{name}[/red]: {detail}")
        if not report.ok:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
