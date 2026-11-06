"""Üretilen MIDI'leri dinlenebilir bir önizlemeye çevirir (WAV + HTML).

Her parça için basit eklemeli sentezle WAV yazar ve tek bir sayfada oynatıcı +
piyano rulosu + ölçütler olarak listeler. Amaç profesyonel ses üretimi değil
(o 7-8. aşama): eserin kulağa nasıl geldiğini **hemen** dinleyebilmek.

Kullanım:
    ./.venv/Scripts/python.exe scripts/render_preview.py \\
        --midi outputs/midi/eser1.mid outputs/midi/eser2.mid --out-dir outputs/preview
"""

from __future__ import annotations

import argparse
import html
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from ophelia.core.console import make_console  # noqa: E402
from ophelia.core.paths import OUTPUTS_DIR  # noqa: E402
from ophelia.data.midi_io import NoteEvent, analyze, iter_notes, load_midi  # noqa: E402
from ophelia.theory.scales import estimate_key  # noqa: E402

console = make_console()

SAMPLE_RATE = 22050
HARMONICS = ((1, 1.0), (2, 0.32), (3, 0.14), (4, 0.06))


def _midi_to_freq(pitch: int) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def synthesize(
    notes: list[NoteEvent], *, tempo_bpm: float = 100.0, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Notaları basit eklemeli sentezle tek kanallı dalga formuna çevirir."""
    if not notes:
        return np.zeros(sample_rate // 10, dtype=np.float32)

    ticks_per_second = 480 * tempo_bpm / 60.0
    end_tick = max(note.end for note in notes)
    total = int(end_tick / ticks_per_second * sample_rate) + sample_rate // 2
    buffer = np.zeros(total, dtype=np.float32)

    for note in notes:
        start = int(note.start / ticks_per_second * sample_rate)
        length = max(1, int(note.duration / ticks_per_second * sample_rate))
        stop = min(total, start + length)
        if stop <= start:
            continue

        time = np.arange(stop - start, dtype=np.float32) / sample_rate
        frequency = _midi_to_freq(note.pitch)
        wave = np.zeros_like(time)
        for harmonic, weight in HARMONICS:
            if frequency * harmonic < sample_rate / 2:
                wave += weight * np.sin(2 * math.pi * frequency * harmonic * time)

        # Zarf: hızlı atak + üstel sönüm + kısa bırakma (org/klavye hissi)
        attack = max(1, int(0.006 * sample_rate))
        release = max(1, int(0.05 * sample_rate))
        envelope = np.ones_like(time)
        envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
        envelope[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
        envelope *= np.exp(-time * 1.1)

        buffer[start:stop] += wave * envelope * (note.velocity / 127.0) * 0.25

    peak = float(np.max(np.abs(buffer))) or 1.0
    return (buffer / peak * 0.85).astype(np.float32)


def write_wav(samples: np.ndarray, path: Path, *, sample_rate: int = SAMPLE_RATE) -> Path:
    """16-bit PCM WAV yazar."""
    from scipy.io import wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, (samples * 32767).astype(np.int16))
    return path


def tempo_of(path: Path, default: float = 100.0) -> float:
    """MIDI'deki ilk tempo işaretini BPM'e çevirir."""
    import mido

    for track in mido.MidiFile(str(path)).tracks:
        for message in track:
            if message.type == "set_tempo":
                return 60_000_000 / message.tempo
    return default


def piano_roll_svg(notes: list[NoteEvent], *, width: int = 1080, height: int = 220) -> str:
    """Notaları zaman-perde düzleminde SVG olarak çizer."""
    if not notes:
        return "<p>Nota yok.</p>"

    end = max(note.end for note in notes)
    low = min(note.pitch for note in notes)
    high = max(note.pitch for note in notes)
    span = max(1, high - low + 1)
    colors = ("#d9b46a", "#8fb98c", "#8aa9c9", "#c98a8a")

    rows = []
    for note in notes:
        x = note.start / end * width
        w = max(2.0, note.duration / end * width)
        y = height - (note.pitch - low + 1) / span * height
        h = max(3.0, height / span)
        color = colors[note.pitch % len(colors)]
        rows.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="1.5" fill="{color}" opacity="{0.55 + 0.45 * note.velocity / 127:.2f}"/>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'preserveAspectRatio="none" role="img" aria-label="piyano rulosu">'
        f'<rect width="100%" height="100%" fill="#0b0e14"/>' + "".join(rows) + "</svg>"
    )


def average_polyphony(notes: list[NoteEvent]) -> float:
    """Nota süresi boyunca ortalama eşzamanlı nota sayısı (olay taraması)."""
    events: list[tuple[int, int]] = []
    for note in notes:
        events.append((note.start, 1))
        events.append((note.end, -1))
    if not events:
        return 0.0

    events.sort()
    active = 0
    weighted = 0.0
    previous = events[0][0]
    for tick, delta in events:
        weighted += active * (tick - previous)
        previous = tick
        active += delta
    return weighted / max(1, events[-1][0])


def _card(index: int, title: str, subtitle: str, midi: Path, wav: Path) -> str:
    notes = list(iter_notes(load_midi(midi)))
    stats = analyze(midi)
    histogram: dict[int, int] = {}
    for note in notes:
        histogram[note.pitch % 12] = histogram.get(note.pitch % 12, 0) + 1
    key = estimate_key(histogram) if histogram else None

    return f"""
<section class="card">
  <h2>{html.escape(title)}</h2>
  <p class="cmd">{html.escape(subtitle)}</p>
  <audio controls preload="metadata" src="{wav.name}"></audio>
  {piano_roll_svg(notes)}
  <div class="grid">
    <div><div class="k">Süre</div><div class="v">{stats.duration_seconds:.1f} sn</div></div>
    <div><div class="k">Nota</div><div class="v">{len(notes)}</div></div>
    <div><div class="k">Polifoni</div><div class="v">{average_polyphony(notes):.2f}</div></div>
    <div><div class="k">Tempo</div><div class="v">{tempo_of(midi):.0f} BPM</div></div>
    <div><div class="k">Tonalite (tahmin)</div><div class="v">{key.name if key else "—"}</div></div>
    <div><div class="k">MIDI</div><div class="v">
      <a download="{midi.name}" href="{midi.name}">indir</a></div></div>
  </div>
</section>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi", type=Path, nargs="+", required=True)
    parser.add_argument("--title", default="Ophelia — üretilen eserler")
    parser.add_argument(
        "--command",
        action="append",
        default=None,
        help="Her parça için gösterilecek komut (sırayla; verilmezse dosya adı)",
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR / "preview")
    args = parser.parse_args()

    missing = [str(path) for path in args.midi if not path.is_file()]
    if missing:
        console.print(f"[red]MIDI yok:[/red] {', '.join(missing)}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    wav_sources: list[Path] = []
    for index, midi in enumerate(args.midi):
        tempo = tempo_of(midi)
        notes = list(iter_notes(load_midi(midi)))
        if not notes:
            console.print(f"[yellow]{midi.name} nota içermiyor, atlandı.[/yellow]")
            continue

        # Yanında hazır bir miks varsa (``ophelia audio render``/``--wav``) onu
        # kullan: enstrüman sesi, önizleme sentezinden daha gerçekçi.
        mixed = midi.with_suffix(".wav")
        target_wav = args.out_dir / f"parca_{index + 1}.wav"
        if mixed.is_file():
            shutil.copyfile(mixed, target_wav)
            source = mixed
            tag = "miks"
        else:
            write_wav(synthesize(notes, tempo_bpm=tempo), target_wav)
            source = target_wav
            tag = "önizleme"
        wav = source
        wav_sources.append(wav)
        local_midi = args.out_dir / f"parca_{index + 1}.mid"
        shutil.copyfile(midi, local_midi)

        commands = args.command or []
        subtitle = commands[index] if index < len(commands) else midi.name
        cards.append(
            _card(index + 1, f"Parça {index + 1} · {tag}", subtitle, local_midi, target_wav)
        )
        console.print(
            f"  • {midi.name} → {target_wav.name} ({tag}, "
            f"{target_wav.stat().st_size / 1024:.0f} KB)"
        )

    page = f"""<!doctype html>
<html lang="tr">
<meta charset="utf-8">
<title>{html.escape(args.title)}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; padding: 32px; background: #0b0e14; color: #e6e9ef;
         font: 15px/1.55 "Segoe UI", system-ui, sans-serif; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; font-weight: 600; }}
  h2 {{ font-size: 16px; margin: 0 0 2px; font-weight: 600; }}
  .sub {{ color: #8b95a7; margin-bottom: 24px; font-size: 13px; }}
  .card {{ background: #12161d; border: 1px solid #1e2530; border-radius: 14px;
           padding: 18px; margin-bottom: 18px; }}
  .cmd {{ color: #8b95a7; font-size: 13px; margin: 0 0 12px; font-family: ui-monospace, monospace; }}
  audio {{ width: 100%; margin-bottom: 12px; }}
  svg {{ border-radius: 8px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
           gap: 14px; margin-top: 12px; }}
  .k {{ color: #8b95a7; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }}
  .v {{ font-size: 17px; font-variant-numeric: tabular-nums; }}
  a {{ color: #8aa9c9; }}
</style>
<h1>{html.escape(args.title)}</h1>
<div class="sub">Ophelia Music Engine · 5. aşama modeli (940.160 parametre) + 6. aşama komut katmanı</div>
{"".join(cards)}
</html>
"""

    index_path = args.out_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")
    console.print(f"[green]{len(cards)} parça[/green] → {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
