"""MIDI okuma, analiz ve kanonik yazma.

Tüm analiz fonksiyonları saf Python + ``mido`` ile çalışır; numpy/music21
gerektirmez, böylece binlerce dosya üzerinde hızlı kalır.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

TICKS_PER_BEAT = 480
DRUM_CHANNEL = 9

DEFAULT_TEMPO_BPM = 100.0
DEFAULT_TIME_SIGNATURE = (4, 4)


class MidiError(Exception):
    """MIDI dosyası okunamadı/geçersiz."""


@dataclass(frozen=True)
class NoteEvent:
    """Ayrıştırılmış tek nota (mutlak tick cinsinden)."""

    start: int
    duration: int
    pitch: int
    velocity: int
    channel: int
    program: int
    track: int

    @property
    def end(self) -> int:
        return self.start + self.duration


@dataclass(frozen=True)
class MidiStats:
    """Bir MIDI dosyasının kalite analizi için gereken özet."""

    ticks_per_beat: int
    n_tracks: int
    n_notes: int
    duration_seconds: float
    duration_beats: float
    tempos: tuple[int, ...]
    time_signatures: tuple[str, ...]
    programs: tuple[int, ...]
    channels: tuple[int, ...]
    pitch_min: int
    pitch_max: int
    unique_pitches: int
    max_polyphony: int
    has_percussion: bool

    @property
    def is_empty(self) -> bool:
        return self.n_notes == 0

    @property
    def pitch_range(self) -> int:
        return 0 if self.is_empty else self.pitch_max - self.pitch_min


def load_midi(path: str | Path) -> mido.MidiFile:
    """MIDI dosyasını okur; bozuk dosyalarda :class:`MidiError` fırlatır."""
    file_path = Path(path)
    if not file_path.is_file():
        raise MidiError(f"dosya bulunamadı: {file_path}")
    try:
        return mido.MidiFile(str(file_path))
    except Exception as exc:  # noqa: BLE001 - mido çok çeşitli hatalar atıyor
        raise MidiError(f"MIDI okunamadı ({file_path.name}): {exc}") from exc


def iter_notes(midi: mido.MidiFile) -> list[NoteEvent]:
    """Dosyadaki tüm notaları mutlak tick ile düzleştirir."""
    notes: list[NoteEvent] = []
    default_program = 0

    for track_index, track in enumerate(midi.tracks):
        absolute = 0
        program = default_program
        # Aynı perde aynı kanalda üst üste binebilir (gerçek dosyalarda görülüyor);
        # tek bir slot tutmak ilk notayı sessizce yutar. Bu yüzden kuyruk.
        open_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}

        for message in track:
            absolute += message.time

            if message.type == "program_change":
                program = message.program
                continue

            if message.type == "note_on" and message.velocity > 0:
                key = (message.channel, message.note)
                open_notes.setdefault(key, []).append((absolute, message.velocity))
                continue

            if message.type in ("note_off", "note_on"):
                key = (message.channel, message.note)
                pending = open_notes.get(key)
                if not pending:
                    continue
                start, velocity = pending.pop(0)
                if not pending:
                    del open_notes[key]
                notes.append(
                    NoteEvent(
                        start=start,
                        duration=max(1, absolute - start),
                        pitch=message.note,
                        velocity=velocity,
                        channel=message.channel,
                        program=program,
                        track=track_index,
                    )
                )

        # Kapanmayan notaları parça sonunda kapat
        for (channel, pitch), pending in open_notes.items():
            for start, velocity in pending:
                notes.append(
                    NoteEvent(
                        start=start,
                        duration=max(1, absolute - start),
                        pitch=pitch,
                        velocity=velocity,
                        channel=channel,
                        program=program,
                        track=track_index,
                    )
                )

    notes.sort(key=lambda note: (note.start, note.pitch))
    return notes


def max_polyphony(notes: list[NoteEvent]) -> int:
    """Aynı anda çalan en fazla nota sayısı."""
    events: list[tuple[int, int]] = []
    for note in notes:
        events.append((note.start, 1))
        events.append((note.end, -1))
    events.sort(key=lambda item: (item[0], item[1]))

    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def analyze(path: str | Path) -> MidiStats:
    """Dosyayı açar ve kalite metriklerini döndürür."""
    midi = load_midi(path)
    notes = iter_notes(midi)

    tempos: list[int] = []
    signatures: list[str] = []
    for track in midi.tracks:
        for message in track:
            if message.type == "set_tempo":
                tempo = int(message.tempo)
                if tempo not in tempos:
                    tempos.append(tempo)
            elif message.type == "time_signature":
                signature = f"{message.numerator}/{message.denominator}"
                if signature not in signatures:
                    signatures.append(signature)

    pitches = [note.pitch for note in notes]
    channels = sorted({note.channel for note in notes})
    programs = sorted({note.program for note in notes})
    ticks = midi.ticks_per_beat or TICKS_PER_BEAT
    total_ticks = max((note.end for note in notes), default=0)

    return MidiStats(
        ticks_per_beat=ticks,
        n_tracks=len(midi.tracks),
        n_notes=len(notes),
        duration_seconds=float(midi.length),
        duration_beats=total_ticks / ticks,
        tempos=tuple(tempos),
        time_signatures=tuple(signatures),
        programs=tuple(programs),
        channels=tuple(channels),
        pitch_min=min(pitches, default=0),
        pitch_max=max(pitches, default=0),
        unique_pitches=len(set(pitches)),
        max_polyphony=max_polyphony(notes),
        has_percussion=DRUM_CHANNEL in channels,
    )


def write_midi(
    notes: list[NoteEvent] | tuple[NoteEvent, ...],
    path: str | Path,
    *,
    ticks_per_beat: int = TICKS_PER_BEAT,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
    time_signature: tuple[int, int] = DEFAULT_TIME_SIGNATURE,
) -> None:
    """Notaları kanonik, deterministik bir MIDI dosyası olarak yazar.

    Format 1: ilk parça yalnızca meta (tempo + ölçü), sonraki parçalar
    (kanal, program) çifti başına birer parça. Aynı girdi her zaman aynı
    baytları üretir (notalar başlangıç/pitch sırasına göre yazılır).
    """
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    meta.append(
        mido.MetaMessage(
            "time_signature",
            numerator=time_signature[0],
            denominator=time_signature[1],
            time=0,
        )
    )
    meta.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(meta)

    groups: dict[tuple[int, int], list[NoteEvent]] = {}
    for note in notes:
        groups.setdefault((note.channel, note.program), []).append(note)

    for (channel, program), group in sorted(groups.items()):
        track = mido.MidiTrack()
        track.append(
            mido.Message("program_change", channel=channel, program=program, time=0)
        )

        events: list[tuple[int, int, int, int]] = []
        for note in group:
            events.append((note.start, 1, note.pitch, note.velocity))
            events.append((note.end, 0, note.pitch, 0))
        # Aynı tick'te önce note_off (0), sonra note_on (1)
        events.sort()

        cursor = 0
        for tick, kind, pitch, velocity in events:
            message = (
                mido.Message("note_on", channel=channel, note=pitch, velocity=velocity)
                if kind
                else mido.Message("note_off", channel=channel, note=pitch, velocity=0)
            )
            message.time = tick - cursor
            cursor = tick
            track.append(message)

        track.append(mido.MetaMessage("end_of_track", time=0))
        midi.tracks.append(track)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(path))
