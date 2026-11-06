"""Sentez ve stereo miks testleri."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ophelia.audio.ensemble import Part, arrange
from ophelia.audio.render import (
    SAMPLE_RATE,
    MixSettings,
    peak,
    render_file,
    render_parts,
    rms,
    stereo_width,
    synthesize_part,
)
from ophelia.data.midi_io import NoteEvent
from ophelia.theory.instruments import get


def _note(start: int, pitch: int, duration: int = 480, velocity: int = 90) -> NoteEvent:
    return NoteEvent(start, duration, pitch, velocity, 0, 0, 0)


def _parts() -> list[Part]:
    return [
        Part(instrument=get("violin"), notes=[_note(0, 72), _note(480, 74)], channel=0, pan=0.4),
        Part(instrument=get("cello"), notes=[_note(0, 48), _note(480, 45)], channel=1, pan=-0.4),
    ]


def test_synthesize_part_is_stereo_and_non_silent() -> None:
    part = _parts()[0]
    left, right = synthesize_part(part, seconds=2.0, ticks_per_second=800.0)

    assert left.shape == right.shape
    assert left.size == pytest.approx(2.0 * SAMPLE_RATE, rel=0.01)
    assert np.max(np.abs(left)) > 0
    assert not np.allclose(left, right)  # pan uygulanmış


def test_mix_peak_is_below_clipping() -> None:
    mix = render_parts(_parts(), tempo_bpm=100.0)
    assert mix.shape[0] == 2
    assert peak(mix) <= 0.95
    assert rms(mix) > 0.001


def test_mix_duration_follows_notes() -> None:
    mix = render_parts(_parts(), tempo_bpm=100.0)
    # 2 vuruş (960 tick) @100 BPM = 1,2 sn + 0,35 sn kuyruk
    seconds = mix.shape[1] / SAMPLE_RATE
    assert 1.4 <= seconds <= 1.7


def test_silent_parts_are_skipped_safely() -> None:
    mix = render_parts([Part(instrument=get("piano"), notes=[_note(0, 60)])], tempo_bpm=120.0)
    assert peak(mix) <= 0.95


def test_render_without_parts_raises() -> None:
    with pytest.raises(ValueError):
        render_parts([])


def test_wav_file_is_stereo_16bit(tmp_path: Path) -> None:
    from scipy.io import wavfile

    target = tmp_path / "miks.wav"
    render_file(_parts(), str(target), tempo_bpm=100.0)

    rate, data = wavfile.read(target)
    assert rate == SAMPLE_RATE
    assert data.shape[1] == 2
    assert data.dtype == np.int16
    assert np.max(np.abs(data)) > 0


def test_single_part_has_zero_width() -> None:
    part = Part(instrument=get("piano"), notes=[_note(0, 60)], pan=0.0)
    mix = render_parts([part], tempo_bpm=100.0)
    assert stereo_width(mix) < 0.01


def test_render_is_deterministic() -> None:
    first = render_parts(_parts(), tempo_bpm=100.0)
    second = render_parts(_parts(), tempo_bpm=100.0)
    assert np.array_equal(first, second)


def test_mix_settings_change_level() -> None:
    quiet = render_parts(_parts(), tempo_bpm=100.0, settings=MixSettings(master_gain=0.2))
    loud = render_parts(_parts(), tempo_bpm=100.0, settings=MixSettings(master_gain=0.9))
    assert peak(quiet) < peak(loud)


def test_render_uses_musical_role_for_envelope() -> None:
    """Çello bas çalıyorsa tını da bas olmalı; enstrümanın etiketi değil."""
    cello = get("cello")
    note = _note(0, 40)
    implicit = Part(instrument=cello, notes=[note], channel=1, pan=0.0)
    explicit = Part(instrument=cello, notes=[note], channel=1, pan=0.0, role="harmony")
    as_bass = Part(instrument=cello, notes=[note], channel=1, pan=0.0, role="bass")

    implicit_left, _ = synthesize_part(implicit, seconds=2.0, ticks_per_second=800.0)
    explicit_left, _ = synthesize_part(explicit, seconds=2.0, ticks_per_second=800.0)
    bass_left, _ = synthesize_part(as_bass, seconds=2.0, ticks_per_second=800.0)

    # Rol verilmezse enstrümanın doğal rolü kullanılır…
    assert np.array_equal(implicit_left, explicit_left)
    # …ama bas rolü verilirse zarf ve tını değişir.
    assert not np.allclose(bass_left, explicit_left)


def test_arranged_chorale_renders(tmp_path: Path) -> None:
    notes = [
        _note(index * 480, pitch)
        for index, chord in enumerate([(48, 55, 60, 64), (43, 50, 55, 59)])
        for pitch in chord
    ]
    parts = arrange(notes, ("violin", "viola", "cello", "contrabass"))
    target = tmp_path / "koral.wav"
    render_file(parts, str(target), tempo_bpm=90.0)

    from scipy.io import wavfile

    rate, data = wavfile.read(target)
    assert rate == SAMPLE_RATE
    assert data.shape[0] > SAMPLE_RATE  # en az bir saniye
