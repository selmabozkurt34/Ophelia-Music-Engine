"""Ophelia komut satırı arayüzü.

Şu an yalnızca 1. aşamanın komutları vardır; sonraki aşamalar buraya
alt komut olarak eklenir.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from ophelia import __project__, __version__
from ophelia.core.config import Config
from ophelia.core.console import make_console
from ophelia.core.envcheck import hardware_report, run_all_checks, summarize
from ophelia.core.paths import DEFAULT_CONFIG_FILE, PROJECT_ROOT
from ophelia.theory.harmony import (
    PROGRESSIONS,
    describe_progression,
    detect_cadence,
    function_of_degree,
    progression,
    resolve_progression_name,
    voice_lead,
)
from ophelia.theory.pitch import midi_to_note
from ophelia.theory.scales import Key

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=f"{__project__} — yapay zekâ destekli beste motoru.",
)
data_app = typer.Typer(no_args_is_help=True, help="Veri seti araçları (3-4. aşama).")
app.add_typer(data_app, name="data")
model_app = typer.Typer(no_args_is_help=True, help="Beste modeli (5. aşama).")
app.add_typer(model_app, name="model")

console = make_console()


@app.command("env")
def env_report() -> None:
    """Donanımı ve geliştirme ortamını denetler."""
    console.rule(f"[bold]{__project__} v{__version__} — Ortam Raporu")

    hardware = hardware_report()
    hw_table = Table(title="Donanım", show_header=False, title_justify="left")
    hw_table.add_column("Özellik", style="cyan", no_wrap=True)
    hw_table.add_column("Değer")
    hw_table.add_row("İşletim sistemi", str(hardware["os"]))
    hw_table.add_row("CPU", str(hardware["cpu"]))
    hw_table.add_row("Çekirdek / iş parçacığı", str(hardware["cpu_count"]))
    hw_table.add_row(
        "RAM", f"{hardware['ram_gb']} GB" if hardware["ram_gb"] else "bilinmiyor"
    )
    hw_table.add_row("GPU", str(hardware["gpu"]))
    hw_table.add_row("PyTorch", str(hardware["torch"] or "kurulu değil"))
    hw_table.add_row("CUDA", "etkin" if hardware["cuda"] else "yok (CPU modu)")
    console.print(hw_table)

    checks = run_all_checks()
    ok_count, total = summarize(checks)
    check_table = Table(title="Ortam Kontrolleri", title_justify="left")
    check_table.add_column("Durum", justify="center", width=7)
    check_table.add_column("Bileşen", style="cyan", no_wrap=True)
    check_table.add_column("Ayrıntı", overflow="fold")
    for check in checks:
        style = "green" if check.ok else "red"
        check_table.add_row(f"[{style}]{check.symbol}[/{style}]", check.name, check.detail)
    console.print(check_table)

    if ok_count == total:
        console.print(f"[green]Ortam hazır: {ok_count}/{total} kontrol geçti.[/green]")
    else:
        console.print(f"[yellow]{ok_count}/{total} kontrol geçti.[/yellow]")


@app.command("info")
def info() -> None:
    """Proje ve yapılandırma özetini gösterir."""
    config = Config.load()
    console.print(f"[bold]{config.project.name}[/bold] v{__version__}")
    console.print(f"Proje kökü : {PROJECT_ROOT}")
    console.print(f"Yapılandırma: {DEFAULT_CONFIG_FILE}")
    console.print(
        f"Tempo {config.midi.tempo_bpm} BPM • "
        f"{config.midi.time_signature[0]}/{config.midi.time_signature[1]} • "
        f"{config.midi.key} • {config.audio.sample_rate} Hz"
    )


@app.command("paths")
def paths() -> None:
    """Proje dizinlerini oluşturur ve listeler."""
    from ophelia.core.paths import ensure_dirs

    ensure_dirs()
    for directory in sorted(PROJECT_ROOT.iterdir()):
        if directory.name.startswith("."):
            continue
        marker = "/" if directory.is_dir() else ""
        console.print(f"  {directory.name}{marker}")


@app.command("theory")
def theory(
    key: Annotated[
        str,
        typer.Option("--key", "-k", help="Tonalite, örn. 'C major', 'A minor', 'F# dorian'"),
    ] = "C major",
    chord_progression: Annotated[
        str,
        typer.Option("--progression", "-p", help="Progresyon adı (pop, blues...) veya dereceler (1-6-4-5)"),
    ] = "I-V-vi-IV",
    seventh: Annotated[bool, typer.Option("--seventh", help="Yedili akorlar kullan")] = False,
) -> None:
    """Bir tonalitenin gamını, diyatonik akorlarını ve progresyonunu gösterir."""
    try:
        tonic = Key.parse(key)
    except ValueError as exc:
        console.print(f"[red]Geçersiz tonalite:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.rule(f"[bold]{tonic.name}[/bold]")

    scale_notes = " ".join(
        midi_to_note(note, flats=tonic.flats) for note in tonic.scale.midi_notes(4)
    )
    console.print(f"Gam          : {scale_notes}")

    table = Table(title="Diyatonik akorlar", title_justify="left")
    table.add_column("Derece", style="cyan", width=7)
    table.add_column("Üçlü", width=8)
    table.add_column("Yedili", width=9)
    table.add_column("Fonksiyon")
    for degree in range(1, 8):
        triad = tonic.chord(degree)
        seventh_chord = tonic.chord(degree, seventh=True)
        function = function_of_degree(degree)
        table.add_row(
            tonic.roman(degree),
            triad.to_symbol(flats=tonic.flats),
            seventh_chord.to_symbol(flats=tonic.flats),
            function,
        )
    console.print(table)

    try:
        degrees = _resolve_degrees(chord_progression)
        chords = progression(tonic, degrees, seventh=seventh)
    except ValueError as exc:
        console.print(f"[red]Geçersiz progresyon:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("\n[bold]Progresyon[/bold]")
    for line in describe_progression(tonic, degrees):
        console.print(f"  {line}")

    voiced = voice_lead(chords)
    voicing_table = Table(title="Ses yürütme (MIDI)", title_justify="left")
    voicing_table.add_column("Akor", style="cyan", width=8)
    voicing_table.add_column("Seslendirme")
    for chord, notes in zip(chords, voiced, strict=True):
        names = " ".join(midi_to_note(note, flats=tonic.flats) for note in notes)
        voicing_table.add_row(chord.to_symbol(flats=tonic.flats), names)
    console.print(voicing_table)

    cadence = detect_cadence(degrees)
    console.print(f"Kadans: {cadence or 'yok'}")


_ROMAN_INDEX = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7}


def _parse_degrees(text: str) -> list[int]:
    """``"1-6-4-5"`` veya ``"I-V-vi-IV"`` metnini derece listesine çevirir."""
    tokens = text.replace("-", " ").replace(",", " ").split()
    if not tokens:
        raise ValueError("boş progresyon")

    degrees: list[int] = []
    for token in tokens:
        if token.isdigit():
            degrees.append(int(token))
            continue
        degree = _ROMAN_INDEX.get(token.upper())
        if degree is None:
            raise ValueError(f"derece çözümlenemedi: {token!r}")
        degrees.append(degree)
    return degrees


def _resolve_degrees(text: str) -> list[int]:
    """Önce hazır progresyon adı olarak dener, olmazsa dereceleri ayrıştırır."""
    try:
        return list(PROGRESSIONS[resolve_progression_name(text)])
    except ValueError:
        return _parse_degrees(text)


@app.command("config")
def show_config(
    file: Annotated[
        Path | None, typer.Option("--file", "-f", help="Okunacak YAML dosyası")
    ] = None,
) -> None:
    """Yapılandırma dosyasını ham hâliyle yazdırır."""
    import yaml

    config = Config.load(file)
    console.print(yaml.safe_dump(config.raw, allow_unicode=True, sort_keys=False))


@data_app.command("sources")
def data_sources(
    status: Annotated[
        str,
        typer.Option("--status", "-s", help="all | approved | rejected | review"),
    ] = "all",
) -> None:
    """Değerlendirilmiş MIDI kaynaklarını ve lisans kararlarını listeler."""
    from ophelia.data.sources import SOURCES, approved_sources, rejected_sources, review_sources

    filters = {
        "all": lambda: tuple(SOURCES.values()),
        "approved": approved_sources,
        "rejected": rejected_sources,
        "review": review_sources,
    }
    if status not in filters:
        console.print(f"[red]Geçersiz durum:[/red] {status} (all|approved|rejected|review)")
        raise typer.Exit(code=1)

    selected = filters[status]()
    table = Table(title="Veri kaynakları", title_justify="left")
    table.add_column("Kaynak", style="cyan", no_wrap=True)
    table.add_column("Lisans", width=16)
    table.add_column("Ticari", width=7)
    table.add_column("Durum", width=10)
    table.add_column("Dosya", justify="right", width=9)

    for source in selected:
        colour = {"approved": "green", "rejected": "red", "review": "yellow"}[source.verdict]
        table.add_row(
            source.slug,
            source.license_id,
            "evet" if source.license.commercial_use else "hayır",
            f"[{colour}]{source.verdict}[/{colour}]",
            f"{source.file_count:,}" if source.file_count else "?",
        )
    console.print(table)

    console.print("\n[bold]Gerekçeler[/bold]")
    for source in selected:
        console.print(f"  [cyan]{source.slug}[/cyan]: {source.rationale}")
        for caveat in source.caveats:
            console.print(f"    - {caveat}")


@data_app.command("build")
def data_build(
    source: Annotated[str, typer.Option("--source", "-s")] = "music21-bach-chorales",
    limit: Annotated[int | None, typer.Option("--limit", help="İlk N dosya")] = None,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Veri hattını çalıştırır: temizle → normalize → böl → raporla."""
    from ophelia.data.pipeline import PipelineConfig, build_dataset
    from ophelia.data.quality import format_report

    try:
        result = build_dataset(PipelineConfig(source_slug=source, seed=seed, limit=limit))
    except (PermissionError, ValueError) as exc:
        console.print(f"[red]Hata:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"Kabul {result.accepted} • elenen {result.rejected} • "
        f"yazılan {result.written} • tekrar {result.duplicates}"
    )
    if result.splits:
        console.print(
            "Bölünme: " + " • ".join(f"{k}={v}" for k, v in result.splits.items())
        )
    if result.report is not None:
        for line in format_report(result.report):
            console.print(f"  {line}")
    for warning in result.warnings:
        console.print(f"[yellow]uyarı:[/yellow] {warning}")

    if not result.written:
        raise typer.Exit(code=1)


@data_app.command("report")
def data_report() -> None:
    """İşlenmiş veri setinin kalite raporunu basar."""
    from ophelia.core.paths import PROCESSED_DIR
    from ophelia.data.pipeline import find_midi_files
    from ophelia.data.quality import build_report, format_report

    files = find_midi_files(PROCESSED_DIR)
    if not files:
        console.print("[yellow]İşlenmiş dosya yok.[/yellow] Önce: ophelia data build")
        raise typer.Exit(code=1)

    report = build_report(files)
    console.rule("[bold]Veri seti kalitesi[/bold]")
    for line in format_report(report):
        console.print(f"  {line}")


@data_app.command("tokenize")
def data_tokenize(
    grid: Annotated[int, typer.Option("--grid", help="Izgara (tick), 480 PPQ'da 120 = 16'lık")] = 120,
    limit: Annotated[int | None, typer.Option("--limit", help="Bölüm başına ilk N dosya")] = None,
    verify: Annotated[bool, typer.Option("--verify", help="Gidiş-dönüşü doğrula")] = False,
) -> None:
    """Veri setini token dizilerine çevirir (data/tokens)."""
    from ophelia.core.paths import PROCESSED_DIR, SPLITS_DIR, TOKENS_DIR
    from ophelia.data.token_dataset import (
        build_token_dataset,
        collect_split_paths,
        verify_roundtrip,
    )
    from ophelia.data.tokens import MidiTokenizer, TokenizerConfig

    splits_file = SPLITS_DIR / "splits.json"
    if not splits_file.is_file():
        console.print("[red]splits.json yok.[/red] Önce: ophelia data build")
        raise typer.Exit(code=1)

    tokenizer = MidiTokenizer(TokenizerConfig(grid_ticks=grid))
    paths = collect_split_paths(splits_file, PROCESSED_DIR)
    dataset = build_token_dataset(paths, tokenizer, limit_per_split=limit)
    dataset.save(TOKENS_DIR, tokenizer)

    console.print(
        f"Sözlük {tokenizer.vocabulary.size} token • ızgara {grid} tick • "
        f"toplam {dataset.total_tokens:,} token"
    )
    for name, arrays in dataset.splits.items():
        console.print(f"  {name}: {len(arrays)} dosya, {arrays.total_tokens:,} token")

    if verify:
        all_paths = [path for value in paths.values() for path in value]
        if limit is not None:
            all_paths = [path for value in paths.values() for path in value[:limit]]
        report = verify_roundtrip(all_paths, tokenizer)
        console.print(
            f"Gidiş-dönüş: {report.matched}/{report.checked} kayıpsız "
            f"(%{report.rate * 100:.1f})"
        )
        if not report.ok:
            for name, detail in report.failures[:5]:
                console.print(f"  [red]{name}[/red]: {detail}")
            raise typer.Exit(code=1)


@data_app.command("roundtrip")
def data_roundtrip(
    limit: Annotated[int | None, typer.Option("--limit", help="İlk N dosya")] = None,
    grid: Annotated[int, typer.Option("--grid")] = 120,
) -> None:
    """İşlenmiş dosyaların token gidiş-dönüşünü doğrular."""
    from ophelia.core.paths import PROCESSED_DIR
    from ophelia.data.pipeline import find_midi_files
    from ophelia.data.token_dataset import verify_roundtrip
    from ophelia.data.tokens import MidiTokenizer, TokenizerConfig

    files = find_midi_files(PROCESSED_DIR)
    if limit is not None:
        files = files[:limit]
    if not files:
        console.print("[yellow]İşlenmiş dosya yok.[/yellow]")
        raise typer.Exit(code=1)

    report = verify_roundtrip(files, MidiTokenizer(TokenizerConfig(grid_ticks=grid)))
    console.print(
        f"{report.matched}/{report.checked} dosya kayıpsız (%{report.rate * 100:.1f})"
    )
    if not report.ok:
        for name, detail in report.failures[:5]:
            console.print(f"  [red]{name}[/red]: {detail}")
        raise typer.Exit(code=1)


@data_app.command("verify")
def data_verify() -> None:
    """Manifest kayıtlarını diskteki dosyalarla karşılaştırır."""
    from ophelia.core.paths import DATA_DIR, MANIFESTS_DIR
    from ophelia.data.manifest import Manifest

    manifest = Manifest.load(MANIFESTS_DIR)
    result = manifest.verify(DATA_DIR)
    missing, corrupted = result["missing"], result["corrupted"]

    console.print(
        f"Kaynak: {len(manifest.sources)} • dosya kaydı: {len(manifest.files)}"
    )
    if missing or corrupted:
        console.print(f"[red]eksik: {len(missing)} • bozuk: {len(corrupted)}[/red]")
        raise typer.Exit(code=1)
    console.print("[green]Tüm kayıtlar doğrulandı.[/green]")


@model_app.command("train")
def model_train(
    steps: Annotated[int, typer.Option("--steps")] = 1200,
    context: Annotated[int, typer.Option("--context")] = 256,
    d_model: Annotated[int, typer.Option("--d-model")] = 128,
    layers: Annotated[int, typer.Option("--layers")] = 3,
    heads: Annotated[int, typer.Option("--heads")] = 4,
    dropout: Annotated[float, typer.Option("--dropout")] = 0.25,
    batch: Annotated[int, typer.Option("--batch")] = 16,
    lr: Annotated[float, typer.Option("--lr")] = 3e-4,
    weight_decay: Annotated[float, typer.Option("--weight-decay")] = 0.05,
    eval_interval: Annotated[int, typer.Option("--eval-interval")] = 150,
    out: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    """Token veri setiyle beste modelini eğitir."""
    from ophelia.core.paths import MODELS_DIR, TOKENS_DIR
    from ophelia.data.manifest import utc_now
    from ophelia.data.token_dataset import TokenDataset
    from ophelia.data.tokens import MidiTokenizer
    from ophelia.model import ModelConfig, OpheliaGPT, TrainConfig, train
    from ophelia.model.dataset import TokenWindows, pad_token_id

    if not (TOKENS_DIR / "meta.json").is_file():
        console.print("[red]Token veri seti yok.[/red] Önce: ophelia data tokenize")
        raise typer.Exit(code=1)

    dataset = TokenDataset.load(TOKENS_DIR, MidiTokenizer())
    tokenizer = MidiTokenizer(dataset.config)
    pad_id = pad_token_id(tokenizer.vocabulary.tokens)

    model_config = ModelConfig(
        vocab_size=dataset.vocab_size,
        context=context,
        d_model=d_model,
        n_layer=layers,
        n_head=heads,
        dropout=dropout,
    )
    train_config = TrainConfig(
        batch_size=batch,
        max_steps=steps,
        lr=lr,
        weight_decay=weight_decay,
        eval_interval=eval_interval,
        checkpoint_every=eval_interval,
    )
    windows = TokenWindows(dataset.splits["train"], context, pad_id=pad_id)
    validation = TokenWindows(dataset.splits["val"], context, pad_id=pad_id)

    target = out or (MODELS_DIR / f"ophelia-{utc_now().replace(':', '')}")
    # Sözlük ve tokenizer künyesi eğitimden **önce** yazılır: eğitim yarıda kesse de
    # checkpoint sözlüksüz kalmaz.
    target.mkdir(parents=True, exist_ok=True)
    tokenizer.vocabulary.save(target / "vocab.json")
    (target / "dataset_meta.json").write_text(
        json.dumps(dataset.meta(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    model = OpheliaGPT(model_config)
    result = train(
        model, windows, validation, model_config, train_config, pad_id=pad_id, out_dir=target
    )

    console.print(
        f"Kayıp {result.final_train_loss:.4f} • val {result.best_val_loss} • "
        f"{result.seconds / 60:.1f} dk → {result.best_checkpoint}"
    )


@model_app.command("generate")
def model_generate(
    checkpoint: Annotated[Path | None, typer.Option("--model", "-m")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    tokens: Annotated[int, typer.Option("--tokens")] = 1200,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.95,
    top_k: Annotated[int, typer.Option("--top-k")] = 40,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    polyphony: Annotated[int, typer.Option("--polyphony")] = 6,
    min_notes: Annotated[int, typer.Option("--min-notes")] = 120,
    constrained: Annotated[bool, typer.Option("--constrained/--free")] = True,
) -> None:
    """Eğitilmiş modelle yeni bir eser üretir (MIDI)."""
    from ophelia.core.paths import MIDI_OUT_DIR, MODELS_DIR
    from ophelia.data.midi_io import analyze
    from ophelia.model import SampleConfig, generate_piece, load_checkpoint, load_tokenizer

    path = checkpoint
    if path is None:
        candidates = sorted(MODELS_DIR.glob("*/best.pt"), key=lambda item: item.stat().st_mtime)
        path = candidates[-1] if candidates else None
    if path is None or not Path(path).is_file():
        console.print("[red]Checkpoint bulunamadı.[/red] Önce: ophelia model train")
        raise typer.Exit(code=1)

    model, info = load_checkpoint(path)
    try:
        tokenizer = load_tokenizer(path)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if tokenizer.vocabulary.size != info.model_config.vocab_size:
        console.print(
            f"[red]Sözlük boyutu uyuşmuyor:[/red] {tokenizer.vocabulary.size} ≠ "
            f"{info.model_config.vocab_size}"
        )
        raise typer.Exit(code=1)

    target = out or (MIDI_OUT_DIR / f"ophelia_{info.step:04d}_{tokens}.mid")
    result = generate_piece(
        model,
        tokenizer,
        SampleConfig(
            max_new_tokens=tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            max_polyphony=polyphony,
            min_notes=min_notes,
            constrained=constrained,
        ),
        midi_path=target,
    )
    stats = analyze(target)

    console.print(
        f"{result.n_notes} nota • {stats.duration_seconds:.1f} sn • "
        f"token {result.n_tokens} (EOS: {result.stopped_at_eos}) → {target}"
    )


@model_app.command("info")
def model_info(
    checkpoint: Annotated[Path | None, typer.Option("--model", "-m")] = None,
) -> None:
    """Checkpoint'in özetini gösterir."""
    from ophelia.core.paths import MODELS_DIR
    from ophelia.model import load_checkpoint

    path = checkpoint
    if path is None:
        candidates = sorted(MODELS_DIR.glob("*/best.pt"), key=lambda item: item.stat().st_mtime)
        path = candidates[-1] if candidates else None
    if path is None or not Path(path).is_file():
        console.print("[yellow]Checkpoint bulunamadı.[/yellow]")
        raise typer.Exit(code=1)

    _, info = load_checkpoint(path)
    config = info.model_config
    console.print(f"Model      : {path}")
    console.print(f"Adım       : {info.step}")
    console.print(f"Parametre  : {config.parameter_estimate:,}")
    console.print(
        f"Mimari     : {config.n_layer} katman · d_model {config.d_model} · "
        f"{config.n_head} başlık · bağlam {config.context}"
    )
    console.print(f"Val kaybı  : {info.best_val_loss}")


audio_app = typer.Typer(no_args_is_help=True, help="Enstrüman ve ses üretimi (7. aşama).")
app.add_typer(audio_app, name="audio")


@audio_app.command("instruments")
def audio_instruments(
    role: Annotated[str | None, typer.Option("--role", "-r")] = None,
) -> None:
    """Enstrüman kataloğunu listeler (kimlik, rol, GM program, aralık)."""
    from ophelia.theory.instruments import all_instruments, by_role

    try:
        instruments = by_role(role) if role else all_instruments()
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table(title="Enstrümanlar", title_justify="left")
    table.add_column("Kimlik", style="cyan", no_wrap=True)
    table.add_column("Ad")
    table.add_column("Rol")
    table.add_column("GM", justify="right")
    table.add_column("Aralık", justify="right")
    for instrument in instruments:
        table.add_row(
            instrument.slug,
            instrument.name,
            instrument.role,
            str(instrument.gm_program),
            f"{instrument.low}–{instrument.high}",
        )
    console.print(table)


@audio_app.command("render")
def audio_render(
    midi: Annotated[Path, typer.Argument(help="Tek kanallı MIDI dosyası")],
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    ensemble: Annotated[str | None, typer.Option("--ensemble", "-e")] = None,
    roles: Annotated[str | None, typer.Option("--roles", "-r")] = None,
    multitrack: Annotated[bool, typer.Option("--multitrack/--no-multitrack")] = True,
    reverb: Annotated[
        float, typer.Option("--reverb", min=0.0, max=1.0, help="Salon oranı (0 = kuru miks)")
    ] = 0.24,
    mastering: Annotated[
        bool,
        typer.Option("--master/--no-master", help="−14 LUFS hedefi + true peak tavanı"),
    ] = True,
) -> None:
    """MIDI'yi enstrümanlara dağıtıp stereo WAV'a basar."""
    import mido

    from ophelia.audio import (
        MixSettings,
        arrange,
        arrange_by_role,
        chorale_ensemble,
        peak,
        render_with_stats,
        rms,
        stereo_width,
        write_multitrack_midi,
        write_wav,
    )
    from ophelia.core.paths import WAV_OUT_DIR
    from ophelia.data.midi_io import iter_notes, load_midi
    from ophelia.theory.instruments import get

    if not midi.is_file():
        console.print(f"[red]MIDI yok:[/red] {midi}")
        raise typer.Exit(code=1)

    slugs = tuple(part.strip() for part in ensemble.split(",")) if ensemble else chorale_ensemble()
    role_list = tuple(role.strip() for role in roles.split(",") if role.strip()) if roles else ()
    try:
        for slug in slugs:
            get(slug)
        notes = list(iter_notes(load_midi(midi)))
        parts = arrange_by_role(notes, role_list, slugs) if role_list else arrange(notes, slugs)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    tempo = 100.0
    for track in mido.MidiFile(str(midi)).tracks:
        for message in track:
            if message.type == "set_tempo":
                tempo = 60_000_000 / message.tempo
                break

    target = out or (WAV_OUT_DIR / f"{midi.stem}.wav")
    settings = MixSettings(reverb=reverb, target_lufs=-14.0 if mastering else None)
    samples, stats = render_with_stats(parts, tempo_bpm=tempo, settings=settings)
    write_wav(samples, str(target))
    if multitrack:
        write_multitrack_midi(parts, str(target.with_suffix(".mid")), tempo_bpm=tempo)

    console.rule("[bold]Ses[/bold]")
    for part in parts:
        console.print(
            f"  • {part.instrument.name:<18} {part.part_role:<8} kanal {part.channel} · "
            f"GM {part.instrument.gm_program} · {len(part.notes)} nota · pan {part.pan:+.2f}"
        )
    console.print(f"  tempo            : {tempo:.0f} BPM")
    console.print(
        f"  tepe · rms · genişlik : {peak(samples):.2f} · {rms(samples):.3f} · "
        f"{stereo_width(samples):.3f}"
    )
    if stats is not None:
        console.print(
            f"  yayın            : {stats.lufs_after:.1f} LUFS · true peak "
            f"{stats.true_peak_db:.1f} dBTP · salon {reverb:.2f}"
        )
    else:
        console.print("  yayın            : mastering kapalı (tepe normalizasyonu)")
    console.print(f"  çıktı            : {target}")


notation_app = typer.Typer(
    no_args_is_help=True, help="Notasyon: MusicXML dışa aktarımı (düzenlenebilir nota)."
)
app.add_typer(notation_app, name="notation")


@notation_app.command("export")
def notation_export(
    midi: Annotated[Path, typer.Argument(help="MIDI dosyası (tek ya da çok kanallı)")],
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    title: Annotated[str | None, typer.Option("--title", "-t")] = None,
) -> None:
    """MIDI'yi MusicXML'e çevirir; MuseScore/Dorico/Sibelius'ta açılabilir."""
    import mido

    from ophelia.core.paths import SCORE_OUT_DIR
    from ophelia.data.midi_io import MidiError
    from ophelia.notation import detect_fifths, parts_from_midi, write_musicxml

    if not midi.is_file():
        console.print(f"[red]MIDI yok:[/red] {midi}")
        raise typer.Exit(code=1)

    tempo = 100.0
    for track in mido.MidiFile(str(midi)).tracks:
        for message in track:
            if message.type == "set_tempo":
                tempo = 60_000_000 / message.tempo
                break

    target = out or (SCORE_OUT_DIR / f"{midi.stem}.musicxml")
    try:
        parts = parts_from_midi(midi)
        path = write_musicxml(
            parts,
            target,
            tempo_bpm=tempo,
            **(({"title": title}) if title else {}),
        )
    except (MidiError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    fifths = detect_fifths(parts)
    console.rule("[bold]Nota[/bold]")
    for part in parts:
        console.print(
            f"  • {part.name:<18} kanal {part.channel} · GM {part.program} · "
            f"{len(part.notes)} nota"
        )
    console.print(f"  tempo            : {tempo:.0f} BPM")
    console.print(f"  anahtar imi     : fifths {fifths:+d}")
    console.print(f"  çıktı            : {path}")


abtest_app = typer.Typer(
    no_args_is_help=True,
    help="Kör A/B dinleme testi: gerçek koral ↔ model üretimi (tercih + güven aralığı).",
)
app.add_typer(abtest_app, name="abtest")


def _latest_session(directory: Path | None) -> Path:
    """Verilen klasörü ya da en yeni test oturumunu döndürür."""
    from ophelia.core.paths import ABTEST_OUT_DIR

    if directory is not None:
        return directory
    candidates = sorted(ABTEST_OUT_DIR.glob("*/key.json"))
    if not candidates:
        console.print("[red]Oturum yok.[/red] Önce: ophelia abtest build")
        raise typer.Exit(code=1)
    return candidates[-1].parent


def _pool(path: Path | None, fallback: object) -> list[Path]:
    """Verilen klasör/dosyayı havuza çevirir; verilmezse varsayılan havuz."""
    if path is None:
        return list(fallback)  # type: ignore[call-overload]
    if path.is_dir():
        return sorted(path.glob("*.mid"))
    return [path]


@abtest_app.command("build")
def abtest_build(
    trials: Annotated[int, typer.Option("--trials", "-n", help="Deneme (çift) sayısı")] = 10,
    seconds: Annotated[float, typer.Option("--seconds", "-s", help="Her kaydın süresi")] = 20.0,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    real: Annotated[
        Path | None, typer.Option("--real", help="Gerçek MIDI klasörü (varsayılan: test bölütü)")
    ] = None,
    generated: Annotated[
        Path | None, typer.Option("--generated", "-g", help="Üretilmiş MIDI klasörü")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    title: Annotated[str, typer.Option("--title")] = "Ophelia kör A/B dinleme testi",
) -> None:
    """Kör test oturumu kurar: sesleri üretir, etiket anahtarını diske yazar."""
    from datetime import datetime

    from ophelia.core.paths import ABTEST_OUT_DIR, PROJECT_ROOT
    from ophelia.eval import (
        AbTestError,
        build_session,
        default_generated_pool,
        default_real_pool,
        required_votes,
    )

    real_pool = _pool(real, default_real_pool(PROJECT_ROOT))
    generated_pool = _pool(generated, default_generated_pool(PROJECT_ROOT))
    target = out or ABTEST_OUT_DIR / datetime.now().strftime("oturum-%Y%m%d-%H%M%S")

    try:
        session = build_session(
            real_pool,
            generated_pool,
            target,
            trials=trials,
            seconds=seconds,
            seed=seed,
            title=title,
        )
    except (AbTestError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.rule("[bold]Kör test[/bold]")
    console.print(
        f"  deneme        : {len(session.trials)} × 2 kayıt · {session.seconds:.0f} sn · "
        f"{session.audio_dir}"
    )
    console.print(f"  gerçek havuz  : {len(real_pool)} dosya (test bölütü — eğitimde görülmedi)")
    console.print(f"  üretim havuzu : {len(generated_pool)} dosya")
    console.print(f"  etiket anahtarı: {session.directory / 'key.json'} [yellow](dinleyiciye gitmez)[/yellow]")
    console.print(f"  sıradaki      : ophelia abtest serve --dir {session.directory}")
    console.print(
        f"  örneklem      : ±10 puan kesinlik için ~{required_votes(0.1)} kararlı oy gerekir"
    )


@abtest_app.command("serve")
def abtest_serve(
    directory: Annotated[Path | None, typer.Option("--dir", "-d", help="Oturum klasörü")] = None,
    port: Annotated[int, typer.Option("--port", "-p")] = 8800,
    browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
    lan: Annotated[
        bool,
        typer.Option(
            "--lan/--local",
            help="Yerel ağa aç: aynı ağdaki başka bilgisayarlardan dinlenebilsin",
        ),
    ] = False,
    host: Annotated[
        str,
        typer.Option("--host", help="Bağlanılacak adres (varsayılan 127.0.0.1)"),
    ] = "127.0.0.1",
) -> None:
    """Dinleyici sayfasını açar ve oyları kaydeder.

    Varsayılan olarak yalnızca bu bilgisayardan erişilir. Dört-beş kişi kendi
    bilgisayarından dinleyecekse `--lan` kullanın; adres ekrana yazılır.
    """
    from ophelia.eval import AbTestError, load_session, serve

    bind = "0.0.0.0" if lan else host

    target = _latest_session(directory)
    try:
        session = load_session(target)
    except AbTestError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    try:
        serve(session.directory, host=bind, port=port, open_browser=browser)
    except OSError as exc:
        console.print(f"[red]Sunucu açılamadı:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


@abtest_app.command("rapor")
def abtest_rapor(
    directory: Annotated[Path | None, typer.Option("--dir", "-d", help="Oturum klasörü")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Yazılacak HTML dosyası")] = None,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
) -> None:
    """Sonuçları tarayıcıda okunacak tek dosyalık HTML rapor olarak yazar.

    Test sürerken canlı görünüm için sunucunun `/rapor` adresini kullanın;
    bu komut ise paylaşılabilir/arşivlenebilir dosyayı üretir.
    """
    import webbrowser

    from ophelia.eval import REPORT_FILE, AbTestError, write_report

    target = _latest_session(directory)
    try:
        path = write_report(target, out or target / REPORT_FILE)
    except AbTestError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Rapor yazıldı:[/green] {path}")
    console.print(
        "  canlı sürüm: ophelia abtest serve … → "
        "sunucu bilgisayarında http://127.0.0.1:<port>/rapor"
    )
    if open_browser:
        webbrowser.open(path.resolve().as_uri())


def _ci(interval: tuple[float, float] | None) -> str:
    if interval is None:
        return "—"
    low, high = interval
    return f"%95 GA {low * 100:.1f}–{high * 100:.1f}%"


@abtest_app.command("score")
def abtest_score(
    sources: Annotated[
        list[Path] | None,
        typer.Argument(help="Oy dosyaları ya da klasörleri (varsayılan: oturumun oyları)"),
    ] = None,
    directory: Annotated[Path | None, typer.Option("--dir", "-d", help="Oturum klasörü")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="Raporu JSON olarak yaz")] = None,
) -> None:
    """Oyları etiket anahtarıyla birleştirip tercih oranını ve güven aralığını verir."""
    from ophelia.eval import AbTestError, load_session, load_votes, summarize

    target = _latest_session(directory)
    try:
        session = load_session(target)
    except AbTestError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    paths = list(sources) if sources else [session.votes_path]
    try:
        votes, broken = load_votes(paths)
    except AbTestError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if not votes:
        console.print(
            "[yellow]Hiç oy yok.[/yellow] Önce dinleyicilerin teste girmesi gerek: "
            f"ophelia abtest serve --dir {session.directory}"
        )
        raise typer.Exit(code=1)

    report = summarize(session, votes, broken_lines=broken)
    console.rule("[bold]Sonuç[/bold]")
    console.print(f"  oturum         : {session.session_id} · {len(session.trials)} deneme")
    console.print(f"  oy · dinleyici : {report.votes} · {len(report.listeners)}")
    console.print(
        f"  tercih         : gerçek {report.real_choice} · üretim {report.generated_choice} · "
        f"eşit {report.ties}"
    )
    console.print(f"  gerçek oranı   : {_pct(report.rate_real)} · {_ci(report.ci)}")
    console.print(
        "  p-değeri       : " + ("—" if report.p_value is None else f"{report.p_value:.4f}")
    )
    console.print(
        f"  tanıma         : {report.identification_correct}/{report.identification_n} = "
        f"{_pct(report.identification_rate)} · {_ci(report.identification_ci)}"
    )
    if report.mean_confidence is not None:
        console.print(f"  güven (1-5)    : {report.mean_confidence:.2f}")
    if report.mean_seconds is not None:
        console.print(f"  karar süresi   : {report.mean_seconds:.1f} sn")
    if report.duplicate_votes:
        console.print(
            f"  [yellow]yinelenen oy : {report.duplicate_votes} (aynı dinleyici, aynı deneme —"
            " ilki sayıldı)[/yellow]"
        )
    if broken:
        console.print(f"  [yellow]bozuk satır : {broken}[/yellow]")
    console.print(f"  [bold]yorum[/bold]          : {report.verdict}")
    console.print(f"  tanıma yorumu  : {report.identification_verdict()}")

    trials = Table(title="Deneme bazında", title_justify="left")
    trials.add_column("Deneme", style="cyan", no_wrap=True)
    trials.add_column("Oy", justify="right")
    trials.add_column("Gerçek", justify="right")
    trials.add_column("Üretim", justify="right")
    trials.add_column("Eşit", justify="right")
    trials.add_column("Doğru tanıma", justify="right")
    for item in report.per_trial:
        trials.add_row(
            item.trial_id,
            str(item.votes),
            str(item.real_choice),
            str(item.generated_choice),
            str(item.ties),
            f"{item.correct_guesses}/{item.guesses}",
        )
    console.print(trials)

    listeners = Table(title="Dinleyici bazında", title_justify="left")
    listeners.add_column("Dinleyici", style="cyan", no_wrap=True)
    listeners.add_column("Oy", justify="right")
    listeners.add_column("Gerçek tercihi", justify="right")
    listeners.add_column("Doğru tanıma", justify="right")
    for item in report.listeners:
        listeners.add_row(
            item.listener,
            str(item.votes),
            _pct(item.rate),
            str(item.correct_guesses),
        )
    console.print(listeners)

    if json_out is not None:
        payload = {
            "session": session.session_id,
            "trials": len(session.trials),
            "votes": report.votes,
            "listeners": len(report.listeners),
            "real_choice": report.real_choice,
            "generated_choice": report.generated_choice,
            "ties": report.ties,
            "rate_real": report.rate_real,
            "ci95": report.ci,
            "p_value": report.p_value,
            "identification": {
                "n": report.identification_n,
                "correct": report.identification_correct,
                "rate": report.identification_rate,
                "ci95": report.identification_ci,
                "p_value": report.identification_p,
            },
            "mean_confidence": report.mean_confidence,
            "mean_seconds": report.mean_seconds,
            "verdict": report.verdict,
            "identification_verdict": report.identification_verdict(),
            "per_trial": [vars(item) for item in report.per_trial],
            "per_listener": [vars(item) for item in report.listeners],
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        console.print(f"  JSON           : {json_out}")


@app.command("ui")
def ui_command(
    port: Annotated[int, typer.Option("--port", "-p")] = 8810,
    browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
    lan: Annotated[
        bool,
        typer.Option("--lan/--local", help="Yerel ağa aç: başka bilgisayarlardan da kullanılsın"),
    ] = False,
    host: Annotated[str, typer.Option("--host", help="Bağlanılacak adres")] = "127.0.0.1",
) -> None:
    """Yerel web arayüzü: komut → beste → dinle → WAV/MIDI/MusicXML indir."""
    from ophelia.ui import serve

    try:
        serve(host="0.0.0.0" if lan else host, port=port, open_browser=browser)
    except OSError as exc:
        console.print(f"[red]Arayüz açılamadı:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command("compose")
def compose_command(
    command: Annotated[str, typer.Argument(help="Doğal dil isteği, örn. 'hüzünlü la minör 30 sn'")],
    model_path: Annotated[Path | None, typer.Option("--model", "-m")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    temperature: Annotated[float, typer.Option("--temperature")] = 0.95,
    explain: Annotated[bool, typer.Option("--explain", "-x")] = False,
    ensemble: Annotated[str | None, typer.Option("--ensemble", "-e")] = None,
    roles: Annotated[str | None, typer.Option("--roles", "-r", help="Rol tabanlı düzenleme: melodi,bas")] = None,
    wav: Annotated[bool, typer.Option("--wav/--no-wav")] = False,
    musicxml: Annotated[
        bool, typer.Option("--musicxml/--no-musicxml", help="Yanına .musicxml (nota) yaz")
    ] = False,
) -> None:
    """Doğal dil komutundan eser üretir: komut → parametre → model → MIDI/WAV."""
    from ophelia.core.paths import MIDI_OUT_DIR, MODELS_DIR
    from ophelia.nlp import compose, load_models, parse_command

    spec = parse_command(command, seed=seed)
    if explain:
        console.rule("[bold]Çözümleme[/bold]")
        for note in spec.notes:
            console.print(f"  • {note}")
        console.print(
            f"  → tonalite {spec.key.name if spec.key else '—'} · "
            f"{spec.tempo_bpm:.0f} BPM · {spec.duration_seconds:.0f} sn · "
            f"en fazla {spec.max_polyphony} ses"
        )

    path = model_path
    if path is None:
        candidates = sorted(MODELS_DIR.glob("*/best.pt"), key=lambda item: item.stat().st_mtime)
        path = candidates[-1] if candidates else None
    if path is None or not Path(path).is_file():
        console.print("[red]Checkpoint bulunamadı.[/red] Önce: ophelia model train")
        raise typer.Exit(code=1)

    try:
        model, tokenizer = load_models(path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    # Dosya adı komutu da taşır: aynı seed'le iki farklı istek birbirini ezmesin.
    digest = hashlib.sha1(command.strip().lower().encode("utf-8")).hexdigest()[:8]
    target = out or (MIDI_OUT_DIR / f"ophelia_{seed if seed is not None else 'auto'}_{digest}.mid")
    from dataclasses import replace as dataclass_replace

    overrides: dict[str, object] = {}
    if ensemble:
        overrides["ensemble"] = tuple(part.strip() for part in ensemble.split(",") if part.strip())
    if roles:
        overrides["roles"] = tuple(role.strip() for role in roles.split(",") if role.strip())
    request = dataclass_replace(spec, **overrides) if overrides else spec

    wav_path = target.with_suffix(".wav") if wav else None
    result = compose(
        command,
        model,
        tokenizer,
        out_path=target,
        wav_path=wav_path,
        xml_path=target.with_suffix(".musicxml") if musicxml else None,
        seed=seed,
        temperature=temperature,
        spec=request,
    )

    console.rule("[bold]Eser[/bold]")
    console.print(f"  tonalite istenen : {result.spec.key.name if result.spec.key else '—'}")
    console.print(f"  modelin bulduğu : {result.adjust.detected.name if result.adjust.detected else '—'}")
    if result.adjust.transposed:
        console.print(f"  aktarım          : {result.adjust.semitones:+d} yarım ton")
    if result.adjust.mode_mismatch:
        console.print("  [yellow]mod uyuşmazlığı[/yellow]: istenen mod ile üretilen mod farklı")
    console.print(f"  tempo            : {result.spec.tempo_bpm:.0f} BPM")
    console.print(f"  nota · süre      : {result.n_notes} · {result.duration_seconds:.1f} sn")
    fit = result.adjust.scale_fit(result.notes, result.spec.key)
    console.print(
        f"  gam uyumu        : {fit * 100:.0f}% "
        f"({result.spec.key.name if result.spec.key else 'serbest'})"
    )
    console.print(f"  tahmin           : {result.final_key.name if result.final_key else '—'}")
    if result.parts:
        console.print(
            "  enstrümanlar     : "
            + ", ".join(
                f"{part.instrument.name} = {part.part_role} ({len(part.notes)})"
                for part in result.parts
            )
        )
    console.print(f"  çıktı            : {target}")
    if result.wav_path:
        console.print(f"  ses              : {result.wav_path}")
    if result.xml_path:
        console.print(f"  nota (MusicXML)  : {result.xml_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
