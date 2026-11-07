"""Ses karakteri: sürekli (yaylı/koa) tını ve nefes alan zarflar (8. aşama).

7. aşamanın sentezi bilinçli olarak iskeletti: saf sinüs harmonikleri ve kısa,
çarpıcı zarf. Koral müzikte bu "oyun müziği / müzik kutusu" etkisi bırakıyordu.
Bu modül aynı notaları **sürekli**, hafif chorus'lu ve vibratolu bir tınıyla
üretir; salonda çalınmış gibi duyulması için zarfın bırakma (release) kuyruğu
nota süresinin dışına taşar (``held`` / ``total`` ayrımı).

Tını hâlâ örnek tabanlı değil; SoundFont 8. aşamanın son adımı.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ophelia.theory.instruments import BASS, HARMONY, MELODY, TEXTURE

#: Yaylı/koa için harmonik seri: 1/n^rolloff, çift harmonikler biraz zayıf.
EVEN_TILT = 0.85
#: Vibratonun oturması için geçen süre (sn) — insan yaylıda anında titremez.
VIBRATO_RAMP = 0.45


@dataclass(frozen=True)
class Voice:
    """Bir müzikal rolün tınısı ve zarfı."""

    harmonics: int
    rolloff: float
    attack: float
    decay: float
    sustain: float
    release: float
    vibrato_hz: float
    vibrato_cents: float
    chorus_cents: float
    breath: float


VOICES: dict[str, Voice] = {
    # Yaylı melodi: zengin spektrum, belirgin vibrato, hızlı ama yumuşak atak.
    MELODY: Voice(12, 1.05, 0.075, 0.30, 0.75, 0.42, 5.1, 9.0, 6.0, 0.0),
    # Koro/viyola dokusu: daha yuvarlak spektrum, nefes gürültüsü.
    HARMONY: Voice(10, 1.18, 0.11, 0.35, 0.80, 0.55, 4.6, 7.0, 8.0, 0.02),
    # Çello/kontrbas: pes harmonikler, kısa atak, uzun sönüm.
    BASS: Voice(8, 1.45, 0.05, 0.45, 0.78, 0.38, 3.6, 3.0, 4.0, 0.0),
    # Uzun pedal/org dokusu: en yavaş atak, en uzun bırakma.
    TEXTURE: Voice(10, 1.22, 0.18, 0.40, 0.85, 0.85, 4.2, 6.0, 9.0, 0.05),
}


def voice_for(role: str) -> Voice:
    """Rolün tınısı; bilinmeyen rolde koro benzeri varsayılan."""
    return VOICES.get(role, VOICES[HARMONY])


def harmonic_weights(voice: Voice) -> np.ndarray:
    """Harmonik genlikleri: tepe 1, 1/n^rolloff ile sönen seri."""
    index = np.arange(1, voice.harmonics + 1, dtype=np.float64)
    tilt = np.where(index % 2 == 0, EVEN_TILT, 1.0)
    return tilt / np.power(index, voice.rolloff)


def _breath_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """Alçak geçiren süzülmüş gürültü: koronun nefes/salon havası."""
    from scipy.signal import lfilter

    noise = rng.normal(0.0, 1.0, length)
    coefficient = 0.02  # ~350 Hz'te -3 dB: nefes, ıslık değil
    smoothed = lfilter([coefficient], [1.0, coefficient - 1.0], noise)
    peak = float(np.max(np.abs(smoothed))) or 1.0
    return smoothed / peak


def note_signal(
    pitch: int,
    seconds: float,
    *,
    voice: Voice,
    sample_rate: int,
    seed: int = 0,
) -> np.ndarray:
    """Tek notanın ham dalga biçimi (zarf uygulanmamış)."""
    length = max(1, int(seconds * sample_rate))
    time = np.arange(length, dtype=np.float64) / sample_rate
    base = 440.0 * 2 ** ((pitch - 69) / 12)
    rng = np.random.default_rng(seed)
    phase0 = float(rng.uniform(0.0, 2 * math.pi))

    # Vibrato gecikmeli girer; faz birikimli hesaplanır ki frekans kayması
    # tıkırtı üretmesin.
    ramp = np.clip(time / VIBRATO_RAMP, 0.0, 1.0)
    depth = voice.vibrato_cents / 1200.0
    lfo = np.sin(2 * math.pi * voice.vibrato_hz * time + phase0)
    phase = (
        2
        * math.pi
        * np.cumsum(base * np.power(2.0, depth * lfo * ramp))
        / sample_rate
        + phase0
    )

    weights = harmonic_weights(voice)
    wave = np.zeros(length, dtype=np.float64)
    for harmonic, weight in enumerate(weights, start=1):
        if base * harmonic >= sample_rate * 0.45:
            break
        wave += weight * np.sin(harmonic * phase + phase0 * 0.7 * harmonic)

    if voice.chorus_cents:
        layers = np.zeros(length, dtype=np.float64)
        for sign in (-1.0, 1.0):
            detuned = base * 2 ** (sign * voice.chorus_cents / 1200)
            chorus_phase = 2 * math.pi * detuned * time + phase0 * (1.0 + sign)
            for harmonic, weight in enumerate(weights, start=1):
                if detuned * harmonic >= sample_rate * 0.45:
                    break
                layers += weight * np.sin(harmonic * chorus_phase)
        wave = (wave + 0.35 * layers) / 1.7

    if voice.breath:
        wave += voice.breath * _breath_noise(length, rng)

    peak = float(np.max(np.abs(wave))) or 1.0
    return (wave / peak).astype(np.float32)


def amplitude_envelope(
    held: float,
    total: float,
    *,
    voice: Voice,
    sample_rate: int,
) -> np.ndarray:
    """ADSR: atak → sönüm → sustain (``held`` boyunca) → üstel bırakma.

    ``total`` > ``held`` olması notanın kuyruğunun duyulması demek; legato ve
    reverb buna dayanır.
    """
    length = max(1, int(total * sample_rate))
    hold = min(max(1, int(held * sample_rate)), length)
    envelope = np.full(length, voice.sustain, dtype=np.float64)

    attack = min(max(1, int(voice.attack * sample_rate)), hold)
    envelope[:attack] = np.linspace(0.0, 1.0, attack)

    decay_start = attack
    decay_end = min(hold, attack + max(1, int(voice.decay * sample_rate)))
    if decay_end > decay_start:
        envelope[decay_start:decay_end] = np.linspace(1.0, voice.sustain, decay_end - decay_start)

    tail = length - hold
    if tail > 0:
        # 60 dB sönüm: bırakma süresi sonunda tepe %2'ye iner.
        rate = -math.log(0.02) / max(voice.release, 1e-3)
        envelope[hold:] = voice.sustain * np.exp(-rate * np.arange(tail) / sample_rate)

    # Son 5 ms'de sıfıra in: bir sonraki notaya geçişte tıkırtı olmasın.
    fade = min(length, max(1, int(0.005 * sample_rate)))
    envelope[-fade:] *= np.linspace(1.0, 0.0, fade)
    return envelope.astype(np.float32)
