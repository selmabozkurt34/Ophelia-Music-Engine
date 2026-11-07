"""Çok kanallı MIDI'yi stereo WAV'a basar (7. aşama) + yayın zinciri (8. aşama).

Sinyal zinciri: nota → tını (harmonik seri + chorus + vibrato) → ADSR zarf →
velocity → kanal seviyesi → pan → toplam → salon reverb'ü → kompresör → hedef
LUFS → true peak tavanı.

7. aşamada zincir burada bitiyordu (kuru, tepe normalizasyonlu). Kuru miks
koral müzikte \"oyun müziği\" gibi duyuluyordu: notalar birbirine bağlanmıyor,
ataklar keskin, mekân yoktu. 8. aşama bunları tını (`timbre`), mekân (`reverb`)
ve seviye (`mastering`) katmanlarına ayırır.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ophelia.audio import timbre
from ophelia.audio.ensemble import Part
from ophelia.audio.mastering import CEILING_DB, TARGET_LUFS, MasterStats, integrated_lufs, master
from ophelia.audio.reverb import apply_reverb
from ophelia.data.midi_io import NoteEvent

SAMPLE_RATE = 44100
TICKS_PER_SECOND = 480 * 100 / 60  # varsayılan: 480 PPQ, 100 BPM

#: Sondaki notanın sönümü için pay (sn).
TAIL_SECONDS = 0.35
#: Bu kadar kısa boşluklarda notalar birbirine bağlanır (legato).
LEGATO_GAP = 0.35
#: Legato örtüşmesi: iki nota bir an birlikte tınlar, yay değişimi gibi.
LEGATO_OVERLAP = 0.09
#: İnsanlaştırma: notalar tam ızgaraya değil, ±bu kadar kayarak girer.
HUMANIZE_SECONDS = 0.012
#: Güçlü vuruşta hafif vurgu (koral cümlelemesi).
ACCENT = 1.05


@dataclass(frozen=True)
class MixSettings:
    """Karıştırma ve yayın parametreleri."""

    sample_rate: int = SAMPLE_RATE
    peak: float = 0.89  # -1 dBFS (yalnızca hedef LUFS kapalıyken kullanılır)
    master_gain: float = 0.8
    reverb: float = 0.24  # kuru sinyale göre ıslak oran
    reverb_seconds: float = 2.2
    target_lufs: float | None = TARGET_LUFS
    ceiling_db: float = CEILING_DB
    legato: bool = True
    humanize: bool = True


def _frequency(pitch: int) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def _note_seed(note: NoteEvent) -> int:
    """Nota başına sabit tohum: vibrato/chorus fazı tekrarlanabilir olsun."""
    return int(note.pitch * 7919 + note.start * 31 + note.duration)


def _humanize(note: NoteEvent) -> float:
    """Notanın giriş gecikmesi (sn); tohumlu, yani çıktı yine tekrarlanabilir."""
    rng = np.random.default_rng(_note_seed(note) + 1)
    return float(rng.uniform(-HUMANIZE_SECONDS, HUMANIZE_SECONDS))


def synthesize_part(
    part: Part,
    *,
    seconds: float,
    ticks_per_second: float,
    sample_rate: int = SAMPLE_RATE,
    legato: bool = True,
    humanize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Bir parçayı stereo dalga formuna çevirir (sol, sağ).

    Dönen diziler ``seconds``'tan uzun olabilir: her notanın bırakma kuyruğu
    (ve legato örtüşmesi) süreye sığmak zorunda değil.
    """
    total = max(1, int(seconds * sample_rate))
    role = part.part_role
    voice = timbre.voice_for(role)
    notes = sorted(part.notes, key=lambda item: (item.start, item.pitch))
    onsets = sorted({note.start for note in notes})
    following = {current: nxt for current, nxt in zip(onsets, onsets[1:], strict=False)}

    rendered: list[tuple[int, np.ndarray]] = []
    for note in notes:
        end_tick = note.end
        next_onset = following.get(note.start)
        if legato and next_onset is not None:
            gap = (next_onset - end_tick) / ticks_per_second
            if gap <= LEGATO_GAP:
                end_tick = max(end_tick, next_onset) + int(LEGATO_OVERLAP * ticks_per_second)

        held = max(1.0 / sample_rate, (end_tick - note.start) / ticks_per_second)
        buffer_seconds = held + voice.release
        signal = timbre.note_signal(
            note.pitch,
            buffer_seconds,
            voice=voice,
            sample_rate=sample_rate,
            seed=_note_seed(note),
        )
        envelope = timbre.amplitude_envelope(
            held, buffer_seconds, voice=voice, sample_rate=sample_rate
        )
        accent = ACCENT if note.start % 480 == 0 else 1.0
        gain = (note.velocity / 127.0) ** 1.4 * 0.22 * accent
        offset = _humanize(note) if humanize else 0.0
        start = max(0, int((note.start / ticks_per_second + offset) * sample_rate))
        rendered.append((start, (signal * envelope * gain).astype(np.float32)))

    length = max([total, *(start + wave.size for start, wave in rendered)])
    mono = np.zeros(length, dtype=np.float32)
    for start, wave in rendered:
        if start >= length:
            continue
        stop = min(length, start + wave.size)
        mono[start:stop] += wave[: stop - start]

    # Eşit güçte pan (sin/cos): merkezde her iki kanal aynı seviyede
    angle = (part.pan + 1.0) * math.pi / 4
    left = mono * math.cos(angle)
    right = mono * math.sin(angle)
    return left, right


def _render(
    parts: list[Part],
    *,
    tempo_bpm: float,
    settings: MixSettings,
) -> tuple[np.ndarray, MasterStats | None]:
    if not parts:
        raise ValueError("mikslenecek parça yok")

    ticks_per_second = 480 * tempo_bpm / 60.0
    end_tick = max((note.end for part in parts for note in part.notes), default=0)
    seconds = end_tick / ticks_per_second + TAIL_SECONDS

    channels: list[tuple[np.ndarray, np.ndarray]] = [
        synthesize_part(
            part,
            seconds=seconds,
            ticks_per_second=ticks_per_second,
            sample_rate=settings.sample_rate,
            legato=settings.legato,
            humanize=settings.humanize,
        )
        for part in parts
    ]
    length = max(left.size for left, _ in channels)
    mix = np.zeros((2, length), dtype=np.float32)
    for left, right in channels:
        mix[0, : left.size] += left
        mix[1, : right.size] += right

    if settings.reverb > 0:
        mix = apply_reverb(
            mix,
            sample_rate=settings.sample_rate,
            wet=settings.reverb,
            seconds=settings.reverb_seconds,
        )

    if settings.target_lufs is None:
        level = float(np.max(np.abs(mix))) or 1.0
        return (mix / level * settings.peak * settings.master_gain).astype(np.float32), None

    mastered, stats = master(
        mix,
        settings.sample_rate,
        target_lufs=settings.target_lufs,
        ceiling_db=settings.ceiling_db,
    )
    return mastered, stats


def render_parts(
    parts: list[Part],
    *,
    tempo_bpm: float = 100.0,
    settings: MixSettings | None = None,
) -> np.ndarray:
    """Parçaları stereo (2, N) float32 diziye miksler."""
    mix, _stats = _render(parts, tempo_bpm=tempo_bpm, settings=settings or MixSettings())
    return mix


def render_with_stats(
    parts: list[Part],
    *,
    tempo_bpm: float = 100.0,
    settings: MixSettings | None = None,
) -> tuple[np.ndarray, MasterStats | None]:
    """Miksin yanında yayın zincirinin ölçümünü de döner (CLI raporu için)."""
    return _render(parts, tempo_bpm=tempo_bpm, settings=settings or MixSettings())


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


def channel_correlation(samples: np.ndarray) -> float:
    """Sol/sağ korelasyonu: 1 = mono, 0 = ilgisiz."""
    if samples.shape[0] != 2:
        return 1.0
    left = samples[0].astype(np.float64)
    right = samples[1].astype(np.float64)
    denominator = math.sqrt(float(np.sum(left * left)) * float(np.sum(right * right)))
    if denominator <= 0.0:
        return 1.0
    return float(np.sum(left * right) / denominator)


def loudness(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Karışımın bütünleşik LUFS değeri (rapor için)."""
    return integrated_lufs(samples, sample_rate)


def note_seconds(notes: list[NoteEvent], tempo_bpm: float) -> float:
    """Notaların son bitişinin saniye karşılığı."""
    if not notes:
        return 0.0
    return max(note.end for note in notes) / (480 * tempo_bpm / 60.0)
