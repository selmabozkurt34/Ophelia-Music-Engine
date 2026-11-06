"""Test yardımcıları: sentetik MIDI üretimi (ağ/veri bağımlılığı yok)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from ophelia.data.midi_io import NoteEvent, write_midi


def make_notes(
    pitches: Sequence[int] = (60, 62, 64, 65, 67, 69, 71, 72),
    step: int = 480,
    duration: int = 480,
    *,
    channel: int = 0,
    program: int = 0,
    velocity: int = 90,
) -> list[NoteEvent]:
    """Sıralı (monofonik) bir nota dizisi üretir."""
    return [
        NoteEvent(
            start=index * step,
            duration=duration,
            pitch=pitch,
            velocity=velocity,
            channel=channel,
            program=program,
            track=0,
        )
        for index, pitch in enumerate(pitches)
    ]


@pytest.fixture
def midi_file(tmp_path: Path):
    """Tek dosya üreten fabrika: ``midi_file("a.mid", notes)``."""

    def _factory(
        name: str = "ornek.mid",
        notes: list[NoteEvent] | None = None,
        *,
        ticks_per_beat: int = 480,
        tempo_bpm: float = 100.0,
        time_signature: tuple[int, int] = (4, 4),
    ) -> Path:
        path = tmp_path / name
        write_midi(
            notes if notes is not None else make_notes(),
            path,
            ticks_per_beat=ticks_per_beat,
            tempo_bpm=tempo_bpm,
            time_signature=time_signature,
        )
        return path

    return _factory
