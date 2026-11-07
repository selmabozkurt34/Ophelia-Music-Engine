"""8. aşama: tını, salon reverb'ü, yükseklik ölçümü ve yayın zinciri."""

from __future__ import annotations

import numpy as np
import pytest
from ophelia.audio import timbre
from ophelia.audio.ensemble import Part
from ophelia.audio.mastering import (
    compress,
    integrated_lufs,
    limit_true_peak,
    master,
    true_peak,
)
from ophelia.audio.render import (
    LEGATO_OVERLAP,
    SAMPLE_RATE,
    MixSettings,
    loudness,
    peak,
    render_parts,
    synthesize_part,
)
from ophelia.audio.reverb import apply_reverb, energy_decay_db, hall_ir
from ophelia.data.midi_io import NoteEvent
from ophelia.theory.instruments import get


def _note(start: int, pitch: int, duration: int = 480, velocity: int = 90) -> NoteEvent:
    return NoteEvent(start, duration, pitch, velocity, 0, 0, 0)


def _sine(frequency: float, seconds: float, amplitude: float) -> np.ndarray:
    time = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    mono = (amplitude * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    return np.vstack([mono, mono])


# --- tını ------------------------------------------------------------------


def test_note_signal_is_rich_and_bounded() -> None:
    voice = timbre.voice_for("melody")
    wave = timbre.note_signal(69, 1.0, voice=voice, sample_rate=SAMPLE_RATE, seed=1)
    assert wave.size == SAMPLE_RATE
    assert np.max(np.abs(wave)) == pytest.approx(1.0, abs=0.01)

    # Saf sinüs değil: temel frekansın üstünde harmonik enerji var.
    spectrum = np.abs(np.fft.rfft(wave))
    fundamental = np.argmax(spectrum)
    above = spectrum[fundamental * 2 : fundamental * 3]
    assert above.max() > spectrum[fundamental] * 0.05


def test_roles_have_different_timbres() -> None:
    melody = timbre.note_signal(69, 0.5, voice=timbre.voice_for("melody"), sample_rate=SAMPLE_RATE)
    bass = timbre.note_signal(45, 0.5, voice=timbre.voice_for("bass"), sample_rate=SAMPLE_RATE)
    assert melody.size == bass.size
    assert not np.allclose(melody, bass)


def test_envelope_sustains_then_releases() -> None:
    voice = timbre.voice_for("harmony")
    envelope = timbre.amplitude_envelope(1.0, 1.5, voice=voice, sample_rate=SAMPLE_RATE)
    held = envelope[int(0.7 * SAMPLE_RATE)]
    released = envelope[int(1.4 * SAMPLE_RATE)]
    assert held > 0.5  # sustain platosu: oyun müziğindeki çarpıcı sönüm yok
    assert released < held * 0.5  # kuyruk sönüyor
    assert envelope[0] < 0.05  # atak sıfırdan başlıyor


# --- reverb ----------------------------------------------------------------


def test_hall_ir_is_deterministic_and_normalized() -> None:
    first = hall_ir(seconds=0.5, sample_rate=SAMPLE_RATE)
    second = hall_ir(seconds=0.5, sample_rate=SAMPLE_RATE)
    for left, right in (first, second):
        assert np.array_equal(left, right) is False  # kanallar farklı (stereo)
    assert np.array_equal(first[0], second[0])
    energy = float(np.sqrt(np.sum(np.square(first[0]))))
    assert energy == pytest.approx(1.0, abs=0.01)


def test_reverb_extends_and_decays() -> None:
    dry = _sine(220.0, 0.5, 0.3)
    wet = apply_reverb(dry, sample_rate=SAMPLE_RATE, wet=0.3, seconds=1.0)
    assert wet.shape[1] > dry.shape[1]
    assert energy_decay_db(wet, sample_rate=SAMPLE_RATE) < -10.0


def test_reverb_off_returns_the_same_signal() -> None:
    dry = _sine(220.0, 0.2, 0.3)
    assert apply_reverb(dry, wet=0.0) is dry


# --- yükseklik / tepe ------------------------------------------------------


def test_lufs_calibration_on_a_reference_sine() -> None:
    """İki kanalda 997 Hz, -20 dBFS tepe sinüs ≈ -20,7 LUFS okunmalı."""
    signal = _sine(997.0, 2.0, 0.1)
    assert integrated_lufs(signal, SAMPLE_RATE) == pytest.approx(-20.7, abs=1.0)


def test_lufs_follows_level() -> None:
    quiet = integrated_lufs(_sine(440.0, 1.0, 0.1), SAMPLE_RATE)
    loud = integrated_lufs(_sine(440.0, 1.0, 0.2), SAMPLE_RATE)
    assert loud - quiet == pytest.approx(6.0, abs=0.5)


def test_true_peak_catches_intersample_overs() -> None:
    signal = _sine(11025.0, 0.1, 0.9)
    sample_peak_db = 20 * np.log10(peak(signal))
    assert true_peak(signal, SAMPLE_RATE) >= sample_peak_db - 0.01


def test_compressor_reduces_crest_factor() -> None:
    signal = _sine(220.0, 1.0, 0.9)
    quiet_part = np.zeros_like(signal)
    quiet_part[:, : SAMPLE_RATE // 2] = signal[:, : SAMPLE_RATE // 2] * 0.05
    quiet_part[:, SAMPLE_RATE // 2 :] = signal[:, SAMPLE_RATE // 2 :]
    squashed = compress(quiet_part, SAMPLE_RATE, threshold_db=-20.0, ratio=4.0)

    def crest(data: np.ndarray) -> float:
        return float(np.max(np.abs(data)) / np.sqrt(np.mean(np.square(data))))

    assert crest(squashed) < crest(quiet_part)


def test_limiter_holds_the_ceiling() -> None:
    loud = _sine(440.0, 0.5, 0.99) * 4.0
    limited = limit_true_peak(loud, SAMPLE_RATE, ceiling_db=-1.0)
    assert true_peak(limited, SAMPLE_RATE) <= -1.0 + 0.1


def test_master_hits_target_loudness_and_ceiling() -> None:
    signal = _sine(220.0, 2.0, 0.2)
    mastered, stats = master(signal, SAMPLE_RATE, target_lufs=-14.0, ceiling_db=-1.0)
    assert stats.lufs_after == pytest.approx(-14.0, abs=1.0)
    assert stats.true_peak_db <= -1.0 + 0.1
    assert stats.gain_db > 0  # sessiz sinyal hedefe çekildi


# --- render zinciri --------------------------------------------------------


def _parts() -> list[Part]:
    return [
        Part(instrument=get("violin"), notes=[_note(0, 72), _note(480, 74)], channel=0, pan=0.4),
        Part(instrument=get("cello"), notes=[_note(0, 48), _note(480, 45)], channel=1, pan=-0.4),
    ]


def test_legato_connects_adjacent_notes() -> None:
    part = Part(instrument=get("violin"), notes=[_note(0, 69), _note(480, 71)], pan=0.0)
    ticks_per_second = 800.0
    joined = synthesize_part(
        part, seconds=2.0, ticks_per_second=ticks_per_second, legato=True
    )[0]
    detached = synthesize_part(
        part, seconds=2.0, ticks_per_second=ticks_per_second, legato=False
    )[0]

    junction = int(0.6 * SAMPLE_RATE)  # 480 tick = 0,6 sn
    window = slice(junction - 2000, junction + 2000)
    joined_rms = float(np.sqrt(np.mean(np.square(joined[window]))))
    detached_rms = float(np.sqrt(np.mean(np.square(detached[window]))))
    assert joined_rms > detached_rms * 1.02
    assert LEGATO_OVERLAP > 0


def test_render_hits_publication_targets() -> None:
    mix = render_parts(_parts(), tempo_bpm=100.0)
    assert peak(mix) <= 0.95
    assert loudness(mix) == pytest.approx(-14.0, abs=1.5)


def test_render_without_mastering_keeps_old_path_available() -> None:
    settings = MixSettings(target_lufs=None, reverb=0.0)
    mix = render_parts(_parts(), tempo_bpm=100.0, settings=settings)
    # Eski yol: tepe, peak × master_gain'e oturur (LUFS hedefi uygulanmaz).
    assert peak(mix) == pytest.approx(settings.peak * settings.master_gain, abs=0.005)


def test_render_is_deterministic_with_new_chain() -> None:
    first = render_parts(_parts(), tempo_bpm=100.0)
    second = render_parts(_parts(), tempo_bpm=100.0)
    assert np.array_equal(first, second)


def test_humanization_shifts_onsets_but_stays_reproducible() -> None:
    part = Part(instrument=get("violin"), notes=[_note(0, 69)], pan=0.0)
    straight = synthesize_part(part, seconds=1.0, ticks_per_second=800.0, humanize=False)[0]
    shifted = synthesize_part(part, seconds=1.0, ticks_per_second=800.0, humanize=True)[0]
    again = synthesize_part(part, seconds=1.0, ticks_per_second=800.0, humanize=True)[0]

    assert np.array_equal(shifted, again)  # tohumlu: aynı giriş aynı çıktı
    assert not np.array_equal(straight, shifted)
    first_sound = int(np.argmax(np.abs(shifted) > 1e-4))
    assert first_sound <= int(0.02 * SAMPLE_RATE)  # kayma 20 ms'yi aşmıyor
