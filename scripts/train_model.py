"""Beste modelini eğitir.

Kullanım:
    ./.venv/Scripts/python.exe scripts/train_model.py
    ./.venv/Scripts/python.exe scripts/train_model.py --steps 50 --context 64   # hızlı deneme
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ophelia.core.console import make_console  # noqa: E402
from ophelia.core.paths import MODELS_DIR, TOKENS_DIR  # noqa: E402
from ophelia.data.manifest import utc_now  # noqa: E402
from ophelia.data.token_dataset import TokenDataset  # noqa: E402
from ophelia.data.tokens import MidiTokenizer  # noqa: E402
from ophelia.model import ModelConfig, OpheliaGPT, TrainConfig, train  # noqa: E402
from ophelia.model.dataset import TokenWindows, pad_token_id  # noqa: E402

console = make_console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=TOKENS_DIR)
    parser.add_argument("--out", type=Path, default=None, help="checkpoint dizini")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=150)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (args.data / "meta.json").is_file():
        console.print("[red]Token veri seti yok.[/red] Önce: ophelia data tokenize")
        return 1

    dataset = TokenDataset.load(args.data, MidiTokenizer())
    tokenizer = MidiTokenizer(dataset.config)
    if tokenizer.vocabulary.size != dataset.vocab_size:
        console.print("[red]Sözlük boyutu veri setiyle uyuşmuyor.[/red]")
        return 1

    pad_id = pad_token_id(tokenizer.vocabulary.tokens)
    model_config = ModelConfig(
        vocab_size=dataset.vocab_size,
        context=args.context,
        d_model=args.d_model,
        n_layer=args.layers,
        n_head=args.heads,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        batch_size=args.batch,
        max_steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_interval=args.eval_interval,
        checkpoint_every=args.eval_interval,
        threads=args.threads,
        seed=args.seed,
    )

    train_windows = TokenWindows(dataset.splits["train"], args.context, pad_id=pad_id)
    val_windows = TokenWindows(dataset.splits["val"], args.context, pad_id=pad_id)

    model = OpheliaGPT(model_config)
    out_dir = args.out or (MODELS_DIR / f"ophelia-{utc_now().replace(':', '')}")

    console.rule("[bold]Eğitim[/bold]")
    console.print(f"  parametre      : {model.num_params:,}")
    console.print(f"  bağlam         : {args.context} token")
    console.print(f"  pencere        : train {len(train_windows)} • val {len(val_windows)}")
    console.print(f"  adım           : {args.steps} × batch {args.batch}")
    console.print(f"  checkpoint     : {out_dir}")

    # Sözlük ve veri künyesi eğitimden **önce** yazılır: eğitim yarıda kesse de
    # model diriltilebilir kalır (checkpoint tek başına sözlüksüz işe yaramaz).
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.vocabulary.save(out_dir / "vocab.json")
    (out_dir / "dataset_meta.json").write_text(
        json.dumps(dataset.meta(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    result = train(
        model,
        train_windows,
        val_windows,
        model_config,
        train_config,
        pad_id=pad_id,
        out_dir=out_dir,
    )

    console.rule("[bold]Sonuç[/bold]")
    console.print(f"  son eğitim kaybı : {result.final_train_loss:.4f}")
    console.print(f"  en iyi doğrulama : {result.best_val_loss}")
    console.print(f"  süre             : {result.seconds / 60:.1f} dk")
    console.print(f"  adım başına      : {result.seconds_per_step:.2f} sn")
    console.print(f"  checkpoint       : {result.best_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
