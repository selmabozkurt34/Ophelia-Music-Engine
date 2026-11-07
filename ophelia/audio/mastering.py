"""Yayın zinciri: gürültü seviyesi (LUFS), true peak, kompresör, limiter.

8. aşamanın ölçütü \"−14 LUFS, true peak ≤ −1 dBTP\". Bu modül ölçümü ve
işlemeyi birlikte verir; ölçüm olmadan yapılan mastering bir tahmindir.

İki yaklaşıklık bilinçli olarak belgelenir:

* LUFS **gating'siz** hesaplanır (ITU-R BS.1770'nin kısa-vade kapısı yok). Bu,
  sessiz bölümlerin ortalamayı aşağı çekmesi demek; hedef −14 LUFS için pratikte
  birkaç dB fark eder.
* K-ağırlık filtresi BS.1770'nin iki aşamalı filtre setinin yaklaşımıdır:
  38 Hz yüksek geçiren + 1,5 kHz üstünde ~+4 dB raf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CEILING_DB = -1.0
TARGET_LUFS = -14.0
OVERSAMPLE = 4


@dataclass(frozen=True)
class MasterStats:
    """Zincirin ne yaptığının ölçülmüş özeti."""

    lufs_before: float
    lufs_after: float
    true_peak_db: float
    gain_db: float


def _high_shelf(sample_rate: int, *, freq: float = 1500.0, gain_db: float = 4.0, slope: float = 1.0):
    """RBJ tarifine göre yüksek raf süzgeci katsayıları (BS.1770 ikinci aşaması)."""
    amplitude = 10 ** (gain_db / 40)
    omega = 2 * math.pi * freq / sample_rate
    cos_w, sin_w = math.cos(omega), math.sin(omega)
    alpha = sin_w / 2 * math.sqrt((amplitude + 1 / amplitude) * (1 / slope - 1) + 2)
    root = 2 * math.sqrt(amplitude) * alpha

    b0 = amplitude * ((amplitude + 1) + (amplitude - 1) * cos_w + root)
    b1 = -2 * amplitude * ((amplitude - 1) + (amplitude + 1) * cos_w)
    b2 = amplitude * ((amplitude + 1) + (amplitude - 1) * cos_w - root)
    a0 = (amplitude + 1) - (amplitude - 1) * cos_w + root
    a1 = 2 * ((amplitude - 1) - (amplitude + 1) * cos_w)
    a2 = (amplitude + 1) - (amplitude - 1) * cos_w - root
    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]


def k_weight(stereo: np.ndarray, sample_rate: int) -> np.ndarray:
    """BS.1770 yaklaşımı: 38 Hz yüksek geçiren + 1,5 kHz raf (+4 dB)."""
    from scipy.signal import butter, lfilter, sosfilt

    nyquist = sample_rate / 2
    high_pass = butter(2, 38.0 / nyquist, btype="highpass", output="sos")
    filtered = sosfilt(high_pass, stereo.astype(np.float64), axis=-1)
    b, a = _high_shelf(sample_rate)
    return lfilter(b, a, filtered, axis=-1)


def integrated_lufs(stereo: np.ndarray, sample_rate: int) -> float:
    """K-ağırlıklı, gating'siz bütünleşik yükseklik (LUFS)."""
    if stereo.size == 0:
        return -math.inf
    weighted = k_weight(stereo.astype(np.float64), sample_rate)
    power = float(np.sum(np.mean(np.square(weighted), axis=-1)))
    if power <= 0.0:
        return -math.inf
    return -0.691 + 10 * math.log10(power)


def true_peak(stereo: np.ndarray, sample_rate: int, *, oversample: int = OVERSAMPLE) -> float:
    """Örnekler arası tepe: 4× yeniden örnekleyip en büyük genliği ölçer (dBFS)."""
    from scipy.signal import resample_poly

    if stereo.size == 0:
        return -math.inf
    peak = 0.0
    for channel in stereo:
        upsampled = resample_poly(channel, oversample, 1)
        peak = max(peak, float(np.max(np.abs(upsampled))))
    if peak <= 0.0:
        return -math.inf
    return 20 * math.log10(peak)


def db_to_gain(value: float) -> float:
    return float(10 ** (value / 20))


def compress(
    stereo: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -20.0,
    ratio: float = 2.0,
    release: float = 0.2,
) -> np.ndarray:
    """Bus kompresörü: anlık atak (taşma yok), üstel bırakma (pompalanma yok)."""
    if stereo.ndim != 2 or stereo.shape[1] == 0:
        return stereo
    detector = np.max(np.abs(stereo), axis=0)
    coefficient = math.exp(-1.0 / max(release * sample_rate, 1.0))
    smoothed = np.empty_like(detector)
    state = 0.0
    # Vektörel hâli: y[n] = max(x[n], a*y[n-1]) — ani tepe, yavaş sönüm.
    for index, value in enumerate(detector):
        state = value if value > state * coefficient else state * coefficient
        smoothed[index] = state

    threshold = db_to_gain(threshold_db)
    over = np.maximum(smoothed, threshold)
    target = threshold * np.power(over / threshold, 1.0 / max(ratio, 1.0))
    gain = np.where(smoothed > 0, target / np.maximum(smoothed, 1e-9), 1.0)
    return (stereo * gain.astype(np.float32)).astype(np.float32)


def limit_true_peak(
    stereo: np.ndarray,
    sample_rate: int,
    *,
    ceiling_db: float = CEILING_DB,
    oversample: int = OVERSAMPLE,
) -> np.ndarray:
    """Tepeyi tavana indirir; tavan üstünde kalan yoksa dokunmaz."""
    measured = true_peak(stereo, sample_rate, oversample=oversample)
    if not math.isfinite(measured) or measured <= ceiling_db:
        return stereo
    gain = db_to_gain(ceiling_db - measured)
    return (stereo * gain).astype(np.float32)


def master(
    stereo: np.ndarray,
    sample_rate: int,
    *,
    target_lufs: float | None = TARGET_LUFS,
    ceiling_db: float = CEILING_DB,
    compress_bus: bool = True,
) -> tuple[np.ndarray, MasterStats]:
    """Kompresör → hedef yükseklik → true peak tavanı."""
    signal = (stereo.astype(np.float32, copy=True)).copy()
    before = integrated_lufs(signal, sample_rate)
    if compress_bus:
        signal = compress(signal, sample_rate)

    if target_lufs is not None and math.isfinite(before):
        current = integrated_lufs(signal, sample_rate)
        if math.isfinite(current):
            signal = (signal * db_to_gain(target_lufs - current)).astype(np.float32)

    signal = limit_true_peak(signal, sample_rate, ceiling_db=ceiling_db)
    after = integrated_lufs(signal, sample_rate)
    measured_peak = true_peak(signal, sample_rate)
    gain = after - before if math.isfinite(after) and math.isfinite(before) else 0.0
    return signal, MasterStats(
        lufs_before=before,
        lufs_after=after,
        true_peak_db=measured_peak,
        gain_db=gain,
    )
