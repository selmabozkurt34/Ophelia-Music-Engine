"""Kanonik biçime dönüştürme.

Her MIDI dosyası aynı biçime getirilir:

* sabit PPQ (varsayılan 480)
* tek tempo ve tek ölçü
* notalar ızgaraya oturtulur (varsayılan 16'lık = PPQ/4)
* perküsyon kanalı atılır, perde aralığı kırpılır
* çıktı deterministiktir (aynı girdi → aynı baytlar)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ophelia.data.midi_io import (
    DEFAULT_TEMPO_BPM,
    DEFAULT_TIME_SIGNATURE,
    DRUM_CHANNEL,
    TICKS_PER_BEAT,
    NoteEvent,
    iter_notes,
    load_midi,
    write_midi,
)

PIANO_RANGE = (21, 108)


@dataclass(frozen=True)
class NormalizeConfig:
    """Normalizasyon parametreleri."""

    ticks_per_beat: int = TICKS_PER_BEAT
    tempo_bpm: float = DEFAULT_TEMPO_BPM
    time_signature: tuple[int, int] = DEFAULT_TIME_SIGNATURE
    grid_division: int = 4  # PPQ / 4 = 16'lık nota ızgarası
    clamp_range: tuple[int, int] = PIANO_RANGE
    drop_percussion: bool = True
    merge_channels: bool = True

    @property
    def grid_ticks(self) -> int:
        if self.grid_division <= 0:
            return 1
        return max(1, self.ticks_per_beat // self.grid_division)


@dataclass(frozen=True)
class NormalizeResult:
    """Dönüştürme sonucu."""

    source: Path
    target: Path
    n_notes_in: int
    n_notes_out: int
    dropped_notes: int
    src_ticks_per_beat: int


def _quantize(value: int, grid: int) -> int:
    return int(round(value / grid)) * grid


def normalize_notes(
    notes: list[NoteEvent],
    source_ppq: int,
    config: NormalizeConfig,
) -> list[NoteEvent]:
    """Notaları hedef PPQ'ya ölçekler, ızgaraya oturtur ve temizler."""
    if source_ppq <= 0:
        raise ValueError(f"geçersiz PPQ: {source_ppq}")

    scale = config.ticks_per_beat / source_ppq
    grid = config.grid_ticks
    low, high = config.clamp_range
    normalized: list[NoteEvent] = []

    for note in notes:
        if config.drop_percussion and note.channel == DRUM_CHANNEL:
            continue

        pitch = max(low, min(high, note.pitch))
        # Başlangıç ve süre ayrı ayrı oturtulur: kısa notalar yanlışlıkla
        # uzamasın (bitiş noktasını oturtmak süreyi şişiriyordu).
        start = _quantize(round(note.start * scale), grid)
        # Izgara altındaki notalar **bir ızgara birimine** yuvarlanır; aksi hâlde
        # 1 tick'lik hayalet notalar oluşur ve token temsili ızgaraya oturmaz.
        duration = max(grid, _quantize(round(note.duration * scale), grid))

        normalized.append(
            NoteEvent(
                start=start,
                duration=duration,
                pitch=pitch,
                velocity=max(1, min(127, note.velocity)),
                channel=0 if config.merge_channels else note.channel,
                program=note.program,
                track=0,
            )
        )

    normalized.sort(key=lambda note: (note.start, note.pitch, note.duration))
    return normalized


def normalize_file(
    source: str | Path,
    target: str | Path,
    config: NormalizeConfig | None = None,
) -> NormalizeResult:
    """Dosyayı okur, normalize eder ve hedefe yazar."""
    cfg = config or NormalizeConfig()
    source_path = Path(source)
    target_path = Path(target)

    midi = load_midi(source_path)
    notes = iter_notes(midi)
    normalized = normalize_notes(notes, midi.ticks_per_beat, cfg)

    write_midi(
        normalized,
        target_path,
        ticks_per_beat=cfg.ticks_per_beat,
        tempo_bpm=cfg.tempo_bpm,
        time_signature=cfg.time_signature,
    )

    return NormalizeResult(
        source=source_path,
        target=target_path,
        n_notes_in=len(notes),
        n_notes_out=len(normalized),
        dropped_notes=len(notes) - len(normalized),
        src_ticks_per_beat=midi.ticks_per_beat,
    )
