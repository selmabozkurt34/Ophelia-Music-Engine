"""Geliştirme ortamını denetler ve özet rapor basar.

Kullanım:
    ./.venv/Scripts/python.exe scripts/check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Proje kökünü sys.path'e ekle (paket kurulmadan da çalışabilsin)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ophelia import __project__, __version__  # noqa: E402
from ophelia.core.console import make_console  # noqa: E402
from ophelia.core.envcheck import hardware_report, run_all_checks, summarize  # noqa: E402
from rich.table import Table  # noqa: E402

PASS_THRESHOLD_NOTE = "7. aşamaya kadar gerekli değil"


def main() -> int:
    console = make_console()
    console.rule(f"[bold]{__project__} v{__version__} — Ortam Denetimi")

    hardware = hardware_report()
    console.print(
        f"[cyan]OS[/cyan]      : {hardware['os']}\n"
        f"[cyan]CPU[/cyan]     : {hardware['cpu']} ({hardware['cpu_count']} iş parçacığı)\n"
        f"[cyan]RAM[/cyan]     : {hardware['ram_gb']} GB\n"
        f"[cyan]GPU[/cyan]     : {hardware['gpu']}\n"
        f"[cyan]PyTorch[/cyan] : {hardware['torch']}\n"
        f"[cyan]Python[/cyan]  : {hardware['python']} ({hardware['python_executable']})\n"
    )

    checks = run_all_checks()
    table = Table(title="Kontroller", title_justify="left")
    table.add_column("Durum", justify="center", width=7)
    table.add_column("Bileşen", style="cyan", no_wrap=True)
    table.add_column("Ayrıntı", overflow="fold")

    for check in checks:
        style = "green" if check.ok else ("yellow" if check.name.startswith("Ses fontu") else "red")
        table.add_row(f"[{style}]{check.symbol}[/{style}]", check.name, check.detail)

    console.print(table)

    ok_count, total = summarize(checks)
    # Ses fontu 7. aşamada gerekli olduğundan başarısızlığı bloklamaz.
    blocking_failures = [
        c for c in checks if not c.ok and not c.name.startswith("Ses fontu")
    ]

    if not blocking_failures:
        console.print(
            f"[green]Ortam hazır — {ok_count}/{total} kontrol geçti.[/green]"
            f"\n[dim](Ses fontu eksikse bu normaldir: {PASS_THRESHOLD_NOTE}.)[/dim]"
        )
        return 0

    console.print(f"[red]{len(blocking_failures)} kontrol başarısız.[/red]")
    for failure in blocking_failures:
        console.print(f"  • {failure.name}: {failure.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
