"""Salon (hall) reverb'ü: yapay impuls yanıtı + FFT evrişimi (8. aşama).

Gerçek bir mekân kaydı yok; bu yüzden impuls yanıtı **sentetik**: üstel sönen,
alçak geçiren süzülmüş gürültü + birkaç erken yansıma. Kuru miks "oyun müziği"
gibi kulağa geldiği için salon, tını düzeltmesinin en büyük tek parçası.

Tohum sabittir: aynı oturum her çalıştırmada aynı WAV'ı üretir (A/B testinin
adil olması buna bağlı).
"""

from __future__ import annotations

import math

import numpy as np

#: Erken yansımalar: (gecikme sn, genlik)
EARLY_TAPS = ((0.011, 0.50), (0.019, 0.38), (0.031, 0.30), (0.047, 0.22))
DEFAULT_SEED = 20261107


def hall_ir(
    *,
    seconds: float = 2.2,
    sample_rate: int = 44100,
    seed: int = DEFAULT_SEED,
    predelay: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    """Stereo impuls yanıtı (sol, sağ). Enerjisi 1'e normalize edilir."""
    from scipy.signal import lfilter

    length = max(1, int(seconds * sample_rate))
    time = np.arange(length) / sample_rate
    # -60 dB'lik üstel sönüm
    decay = np.exp(-6.9 * time / max(seconds, 1e-3))
    rng = np.random.default_rng(seed)
    channels: list[np.ndarray] = []
    for _ in range(2):
        noise = rng.normal(0.0, 1.0, length)
        # Yüksek frekanslar doğal olarak daha hızlı söner: gövdeyi alçak geçir.
        coefficient = 0.16
        body = lfilter([coefficient], [1.0, coefficient - 1.0], noise)
        impulse = body * decay
        for delay, gain in EARLY_TAPS:
            index = int((predelay + delay) * sample_rate)
            if index < length:
                impulse[index] += gain * decay[index]
        impulse[: int(predelay * sample_rate)] = 0.0
        energy = float(np.sqrt(np.sum(np.square(impulse)))) or 1.0
        channels.append((impulse / energy).astype(np.float32))
    return channels[0], channels[1]


def apply_reverb(
    stereo: np.ndarray,
    *,
    sample_rate: int = 44100,
    wet: float = 0.24,
    seconds: float = 2.2,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Kuru sinyale salon ekler; çıktı, kuyruk kadar uzar."""
    if wet <= 0.0:
        return stereo
    from scipy.signal import fftconvolve

    if stereo.ndim != 2 or stereo.shape[0] != 2:
        raise ValueError("reverb stereo (2, N) dizi bekler")

    left, right = hall_ir(seconds=seconds, sample_rate=sample_rate, seed=seed)
    length = stereo.shape[1] + left.size - 1
    wet_signal = np.zeros((2, length), dtype=np.float32)
    wet_signal[0, :] = fftconvolve(stereo[0], left)[:length]
    wet_signal[1, :] = fftconvolve(stereo[1], right)[:length]

    mix = np.zeros((2, length), dtype=np.float32)
    mix[:, : stereo.shape[1]] += stereo
    mix += wet * wet_signal
    return mix


def tail_seconds(seconds: float, predelay: float = 0.025) -> float:
    """Reverb'ün eklediği kuyruk süresi (ölçüm/testler için)."""
    return max(0.0, seconds) + max(0.0, predelay)


def energy_decay_db(stereo: np.ndarray, *, sample_rate: int = 44100, fraction: float = 0.1) -> float:
    """Son ``fraction``'lık bölümün enerjisi, en yüksek bölüme göre dB — kuyruk ölçütü."""
    if stereo.ndim != 2 or stereo.shape[1] == 0:
        return -math.inf
    window = max(1, int(stereo.shape[1] * fraction))
    head = float(np.mean(np.square(stereo[:, :window], dtype=np.float64)))
    tail = float(np.mean(np.square(stereo[:, -window:], dtype=np.float64)))
    if head <= 0 or tail <= 0:
        return -math.inf if tail <= 0 else math.inf
    return float(10 * math.log10(tail / head))
