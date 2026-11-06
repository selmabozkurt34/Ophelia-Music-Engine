"""Eğitilmiş modelle eser üretir ve MIDI olarak yazar.

Kullanım:
    ./.venv/Scripts/python.exe scripts/generate_music.py --model models/ophelia-v1/best.pt
    ./.venv/Scripts/python.exe scripts/generate_music.py --tokens 300 --temperature 0.8 --seed 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ophelia.core.console import make_console  # noqa: E402
from ophelia.core.paths import MIDI_OUT_DIR, MODELS_DIR  # noqa: E402
from ophelia.data.midi_io import analyze  # noqa: E402
from ophelia.model import (  # noqa: E402
    SampleConfig,
    generate_piece,
    load_checkpoint,
    load_tokenizer,  # noqa: E402
)
from ophelia.theory.scales import estimate_key  # noqa: E402

console = make_console()


def _latest_checkpoint() -> Path | None:
    candidates = sorted(MODELS_DIR.glob("*/best.pt"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None, help="best.pt yolu")
    parser.add_argument("--out", type=Path, default=None, help="çıktı .mid yolu")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--polyphony", type=int, default=6)
    parser.add_argument(
        "--min-notes",
        type=int,
        default=120,
        help="EOS'tan önce gereken en az nota sayısı (parça uzunluğu alt sınırı)",
    )
    parser.add_argument("--tempo", type=float, default=100.0)
    args = parser.parse_args()

    model_path = args.model or _latest_checkpoint()
    if model_path is None or not Path(model_path).is_file():
        console.print("[red]Checkpoint bulunamadı.[/red] Önce: ophelia model train")
        return 1

    model, info = load_checkpoint(model_path)
    try:
        tokenizer = load_tokenizer(model_path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    if tokenizer.vocabulary.size != info.model_config.vocab_size:
        console.print(
            f"[red]Sözlük boyutu uyuşmuyor:[/red] {tokenizer.vocabulary.size} ≠ "
            f"{info.model_config.vocab_size}"
        )
        return 1

    sample_config = SampleConfig(
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        max_polyphony=args.polyphony,
        min_notes=args.min_notes,
    )

    out_path = args.out or (MIDI_OUT_DIR / f"ophelia_{info.step:04d}_{args.tokens}.mid")
    result = generate_piece(
        model, tokenizer, sample_config, midi_path=out_path, tempo_bpm=args.tempo
    )

    stats = analyze(out_path)
    histogram: dict[int, int] = {}
    for note in result.notes:
        histogram[note.pitch % 12] = histogram.get(note.pitch % 12, 0) + 1
    key = estimate_key(histogram) if histogram else None

    console.rule("[bold]Üretim[/bold]")
    console.print(f"  model       : {model_path}")
    console.print(f"  adım        : {info.step} • parametre {info.model_config.parameter_estimate:,}")
    console.print(f"  token       : {result.n_tokens} (EOS: {result.stopped_at_eos})")
    console.print(f"  nota        : {result.n_notes}")
    console.print(f"  süre        : {stats.duration_seconds:.1f} sn")
    console.print(f"  tonalite    : {key.name if key else '-'}")
    console.print(f"  çıktı       : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
