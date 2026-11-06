"""Çok kanallı MIDI'yi stereo WAV'a basar (7. aşama).

Sentez bilinçli olarak basit: her enstrüman için harmonik profil + zarf + pan.
Amaç bir ses kütüphanesi değil, **duyulabilir ve düzenlenebilir** bir çıktı;
8. aşamadaki mixing/mastering bunun üzerine kurulacak.

Sinyal zinciri: nota → harmonikler → zarf → velocity → kanal seviyesi → pan →
toplam → tepe normalizasyonu (tepe ≤ -1 dBFS).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ophelia.audio.ensemble import Part
from ophelia.data.midi_io import NoteEvent
from ophelia.theory.instruments import BASS, HARMONY, MELODY, TEXTURE

SAMPLE_RATE = 44100
TICKS_PER_SECOND = 480 * 100 / 60  # varsayılan: 480 PPQ, 100 BPM

# Rol başına harmonik profili: (harmonik, genlik)
PROFILES: dict[str, tuple[tuple[int, float], ...]] = {
    MELODY: ((1, 1.0), (2, 0.42), (3, 0.22), (4, 0.12), (5, 0.06)),
    HARMONY: ((1, 1.0), (2, 0.55), (3, 0.3), (4, 0.14), (6, 0.05)),
    BASS: ((1, 1.0), (2, 0.28), (3, 0.1)),
    TEXTURE: ((1, 1.0), (2, 0.3), (3, 0.25), (5, 0.08)),
}
DEFAULT_PROFILE = PROFILES[HARMONY]


@dataclass(frozen=True)
class MixSettings:
    """Karıştırma parametreleri."""

    sample_rate: int = SAMPLE_RATE
    peak: float = 0.89  # -1 dBFS
    master_gain: float = 0.8


def _frequency(pitch: int) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def _envelope(length: int, *, role: str, sample_rate: int) -> np.ndarray:
    """Rol başına zarf: yaylılarda yavaş atak, piyanoda hızlı sönüm."""
    time = np.arange(length, dtype=np.float32) / sample_rate
    if role in {TEXTURE, HARMONY}:
        attack_s, decay = 0.06, 0.55
    elif role == BASS:
        attack_s, decay = 0.01, 1.6
    else:
        attack_s, decay = 0.008, 2.4

    attack = max(1, int(attack_s * sample_rate))
    envelope = np.exp(-time * decay)
    envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    release = max(1, int(0.08 * sample_rate))
    if length > release * 2:
        envelope[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
    return envelope


def synthesize_part(
    part: Part,
    *,
    seconds: float,
    ticks_per_second: float,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Bir parçayı stereo dalga formuna çevirir (sol, sağ)."""
    total = max(1, int(seconds * sample_rate))
    mono = np.zeros(total, dtype=np.float32)
    # Zarf ve tını **müzikal işleve** göre seçilir: çello bas çalıyorsa
    # yaylı zarfı değil bas profili kullanılır.
    role = part.part_role
    profile = PROFILES.get(role, DEFAULT_PROFILE)

    for note in part.notes:
        start = int(note.start / ticks_per_second * sample_rate)
        length = max(1, int(note.duration / ticks_per_second * sample_rate))
        stop = min(total, start + length)
        if stop <= start:
            continue

        time = np.arange(stop - start, dtype=np.float32) / sample_rate
        frequency = _frequency(note.pitch)
        wave = np.zeros_like(time)
        for harmonic, weight in profile:
            if frequency * harmonic < sample_rate / 2:
                wave += weight * np.sin(2 * math.pi * frequency * harmonic * time)

        envelope = _envelope(stop - start, role=role, sample_rate=sample_rate)
        gain = (note.velocity / 127.0) ** 1.4 * 0.22
        mono[start:stop] += wave * envelope * gain

    # Eşit güçte pan (sin/cos): merkezde her iki kanal aynı seviyede
    angle = (part.pan + 1.0) * math.pi / 4
    left = mono * math.cos(angle)
    right = mono * math.sin(angle)
    return left, right


def render_parts(
    parts: list[Part],
    *,
    tempo_bpm: float = 100.0,
    settings: MixSettings | None = None,
) -> np.ndarray:
    """Parçaları stereo (2, N) float32 diziye miksler."""
    config = settings or MixSettings()
    if not parts:
        raise ValueError("mikslenecek parça yok")

    ticks_per_second = 480 * tempo_bpm / 60.0
    end_tick = max((note.end for part in parts for note in part.notes), default=0)
    tail = 0.35  # son notanın sönümü için pay
    seconds = end_tick / ticks_per_second + tail

    mix = np.zeros((2, int(seconds * config.sample_rate)), dtype=np.float32)
    for part in parts:
        left, right = synthesize_part(
            part,
            seconds=seconds,
            ticks_per_second=ticks_per_second,
            sample_rate=config.sample_rate,
        )
        mix[0, : left.size] += left
        mix[1, : right.size] += right

    peak = float(np.max(np.abs(mix))) or 1.0
    return (mix / peak * config.peak * config.master_gain).astype(np.float32)


def write_wav(samples: np.ndarray, path: str, *, sample_rate: int = SAMPLE_RATE) -> str:
    """Stereo float diziyi 16-bit PCM WAV olarak yazar."""
    from scipy.io import wavfile

    clipped = np.clip(samples.T, -1.0, 1.0)
    wavfile.write(path, sample_rate, (clipped * 32767).astype(np.int16))
    return path


def render_file(
    parts: list[Part],
    path: str,
    *,
    tempo_bpm: float = 100.0,
    settings: MixSettings | None = None,
) -> str:
    """Parçaları miksleyip WAV yazar; dosya yolunu döner."""
    config = settings or MixSettings()
    samples = render_parts(parts, tempo_bpm=tempo_bpm, settings=config)
    return write_wav(samples, path, sample_rate=config.sample_rate)


def rms(samples: np.ndarray) -> float:
    """Karışımın RMS seviyesi (sessizlik kontrolü için)."""
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def peak(samples: np.ndarray) -> float:
    """Karışımın tepe seviyesi (kırpma kontrolü için)."""
    return float(np.max(np.abs(samples)))


def stereo_width(samples: np.ndarray) -> float:
    """Sol/sağ farkının enerjisi: 0 = tamamen mono."""
    if samples.shape[0] != 2:
        return 0.0
    difference = samples[0] - samples[1]
    return float(np.sqrt(np.mean(np.square(difference, dtype=np.float64))))


def note_seconds(notes: list[NoteEvent], tempo_bpm: float) -> float:
    """Notaların son bitişinin saniye karşılığı."""
    if not notes:
        return 0.0
    return max(note.end for note in notes) / (480 * tempo_bpm / 60.0)
