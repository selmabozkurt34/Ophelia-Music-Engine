"""Tek kanallı besteyi enstrümanlara dağıtır (7. aşama).

Model tek kanallı, tek programlı müzik üretiyor. Duyulabilir bir düzenek için
önce **seslere ayırmak**, sonra her sesi bir enstrümana vermek gerekiyor.

Ses ayırma, koral müziğin doğal yapısına dayanır: aynı anda başlayan notalar
perde sırasına göre en tizden en pesе doğru seslere dağıtılır (S-A-T-B). Bu,
nota sayısını korur — hiçbir nota kaybolmaz, hiçbir nota uydurulmaz.
"""

from __future__ import annotations

from dataclasses import dataclass

import mido

from ophelia.data.midi_io import NoteEvent
from ophelia.theory.instruments import (
    BASS,
    DRUM_CHANNEL,
    HARMONY,
    MELODY,
    PERCUSSION,
    RHYTHM,
    TEXTURE,
    Instrument,
    get,
    normalize_role,
    pick_for_role,
)

# Koral dokusu için doğal dağılım: tizden pese keman → viyola → çello → kontrbas.
CHORALE_ENSEMBLE: tuple[str, ...] = ("violin", "viola", "cello", "contrabass")

# Rol tabanlı düzenlemede seslerin yukarıdan aşağıya sırası: melodi en tiz,
# bas en pes. Diğer roller arada kalır.
ROLE_ORDER: tuple[str, ...] = (MELODY, HARMONY, TEXTURE, RHYTHM, BASS, PERCUSSION)


def chorale_ensemble() -> tuple[str, ...]:
    """Modelin ürettiği 4 sesli dokuyu karşılayan varsayılan topluluk."""
    return CHORALE_ENSEMBLE


@dataclass(frozen=True)
class Part:
    """Bir enstrümanın çalacağı notalar."""

    instrument: Instrument
    notes: list[NoteEvent]
    channel: int = 0
    pan: float = 0.0  # -1 (sol) … +1 (sağ)
    # Müzikal işlev. Enstrümanın doğal rolünden farklı olabilir (çello bas
    # çalıyorsa zarf ve harmonik profili de "bas" olmalı).
    role: str | None = None

    @property
    def slug(self) -> str:
        return self.instrument.slug

    @property
    def part_role(self) -> str:
        return self.role or self.instrument.role


def split_into_voices(
    notes: list[NoteEvent], parts: int, *, mode: str = "register"
) -> list[list[NoteEvent]]:
    """Notaları ``parts`` sese böler.

    ``mode="register"`` (varsayılan): perdeler çeyreklik dilimlere ayrılır, her
    ses kendi **bölgesinde** kalır. Modelin dokusu çoğu zaman 1-2 eşzamanlı
    notayla ilerlediği için "akor sırası" yöntemi her notayı en tiz sese
    yığıyordu (122 notanın 64'ü sadece keman). Bölge yöntemi sesleri dengeler.

    ``mode="chord"``: aynı başlangıçta notalar tizden pese sırayla dağıtılır
    (SATB koral okuması); dört sesli akorlarda daha "doğru" ama seyrek
    dokularda dengesiz.

    Her iki yöntemde de hiçbir nota atılmaz veya uydurulmaz.
    """
    if parts < 1:
        raise ValueError("ses sayısı en az 1 olmalı")
    if mode not in {"register", "chord"}:
        raise ValueError(f"bilinmeyen ses ayırma yöntemi: {mode!r}")
    if not notes:
        return [[] for _ in range(parts)]

    voices: list[list[NoteEvent]] = [[] for _ in range(parts)]
    if mode == "chord":
        by_start: dict[int, list[NoteEvent]] = {}
        for note in notes:
            by_start.setdefault(note.start, []).append(note)
        for start in sorted(by_start):
            group = sorted(by_start[start], key=lambda note: -note.pitch)
            for index, note in enumerate(group):
                voices[min(index, parts - 1)].append(note)
    else:
        import numpy as np

        pitches = np.array([note.pitch for note in notes])
        edges = np.percentile(pitches, [100 * index / parts for index in range(1, parts)])
        for note in notes:
            # Bölge 0 en tiz olmalı: ilk enstrüman melodiyi çalar.
            # Aynı perde sınırın üstünde kalır (>=), böylece bantlar boş kalmaz.
            band = parts - 1 - int(sum(note.pitch >= edge for edge in edges))
            voices[max(0, band)].append(note)

    for voice in voices:
        voice.sort(key=lambda note: (note.start, note.pitch))
    return voices


def fit_to_range(notes: list[NoteEvent], instrument: Instrument) -> tuple[list[NoteEvent], int]:
    """Notaları enstrümanın aralığına oktavlarla taşır; taşınamayanı kırpar.

    Dönen ikinci değer, oktav kaydırılan nota sayısıdır (raporlama için).
    """
    moved: list[NoteEvent] = []
    shifted = 0
    for note in notes:
        pitch = note.pitch
        if not instrument.in_range(pitch):
            candidate = pitch
            while candidate < instrument.low:
                candidate += 12
            while candidate > instrument.high:
                candidate -= 12
            if candidate != pitch:
                shifted += 1
            pitch = instrument.clamp(candidate)

        moved.append(
            NoteEvent(
                start=note.start,
                duration=note.duration,
                pitch=pitch,
                velocity=note.velocity,
                channel=note.channel,
                program=note.program,
                track=note.track,
            )
        )
    return moved, shifted


def _pan_for(index: int, total: int) -> float:
    """Sesleri sahneye yayar: melodi hafif sağ, bas hafif sol."""
    if total <= 1:
        return 0.0
    span = 0.9
    return round(span / 2 - span * index / (total - 1), 3)


def arrange(
    notes: list[NoteEvent],
    ensemble: list[str] | tuple[str, ...],
) -> list[Part]:
    """Notaları verilen enstrüman topluluğuna dağıtır.

    Tiz sesler topluluğun ilk enstrümanına, pes sesler sonuncusuna gider.
    """
    instruments = [get(slug) for slug in ensemble]
    if not instruments:
        raise ValueError("en az bir enstrüman gerekli")
    # Sesler perde bölgesine göre dağıtıldığı için ilk enstrüman **en tiz**
    # olmalı: komutta "kontrbas ve yaylılar" dense de melodi yaylılara gider.
    instruments.sort(key=lambda instrument: -instrument.low)
    if len(instruments) > 15:
        raise ValueError("en fazla 15 enstrüman (9. kanal davul için ayrılmıştır)")

    voices = split_into_voices(notes, len(instruments), mode="register")
    parts: list[Part] = []
    channel = 0
    for index, (instrument, voice) in enumerate(zip(instruments, voices, strict=True)):
        while channel == DRUM_CHANNEL:
            channel += 1
        fitted, _ = fit_to_range(voice, instrument)
        parts.append(
            Part(
                instrument=instrument,
                notes=fitted,
                channel=channel,
                pan=0.0 if instrument.is_drum else _pan_for(index, len(instruments)),
            )
        )
        channel += 1
    return parts


def arrange_by_role(
    notes: list[NoteEvent],
    roles: list[str] | tuple[str, ...],
    ensemble: list[str] | tuple[str, ...] = (),
) -> list[Part]:
    """Notaları **müzikal rollere** dağıtır ve her role enstrüman seçer.

    ``arrange`` yalnızca perde bölgesine bakar: "melodi, armoni, bas" dense de
    sesleri rastgele dört banda bölerdi. Burada her rol bir ses alır ve rol
    sırası (melodi en tiz … bas en pes) korunur. ``ensemble`` verilirse
    kullanıcının istediği enstrümanlar role göre eşlenir; uymayanlar reddedilmez,
    sıradaki role verilir.
    """
    normalized: list[str] = []
    for role in roles:
        candidate = normalize_role(role)
        if candidate == PERCUSSION:
            raise ValueError(
                "davul rolü henüz üretilmiyor: davul kalıbı 8. aşamada gelecek"
            )
        if candidate in normalized:
            raise ValueError(f"rol iki kez verilemez: {candidate!r}")
        normalized.append(candidate)
    if not normalized:
        raise ValueError("en az bir rol gerekli")
    if len(normalized) > 15:
        raise ValueError("en fazla 15 rol (9. kanal davul için ayrılmıştır)")

    ordered = sorted(normalized, key=ROLE_ORDER.index)
    candidates = [get(slug) for slug in ensemble]
    voices = split_into_voices(notes, len(ordered), mode="register")

    parts: list[Part] = []
    channel = 0
    used: set[str] = set()
    for index, (role, voice) in enumerate(zip(ordered, voices, strict=True)):
        instrument = pick_for_role(role, candidates, used)
        used.add(instrument.slug)
        while channel == DRUM_CHANNEL:
            channel += 1
        if channel > 15:
            raise ValueError("boş MIDI kanalı kalmadı")
        fitted, _ = fit_to_range(voice, instrument)
        parts.append(
            Part(
                instrument=instrument,
                notes=fitted,
                channel=channel,
                pan=_pan_for(index, len(ordered)),
                role=role,
            )
        )
        channel += 1
    return parts


def write_multitrack_midi(
    parts: list[Part],
    path: str,
    *,
    tempo_bpm: float = 100.0,
    ticks_per_beat: int = 480,
    time_signature: tuple[int, int] = (4, 4),
) -> str:
    """Parçaları tek bir tip-1 MIDI dosyasına yazar (parça başına bir iz)."""
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    meta.append(
        mido.MetaMessage(
            "time_signature", numerator=time_signature[0], denominator=time_signature[1], time=0
        )
    )
    meta.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(meta)

    for part in parts:
        track = mido.MidiTrack()
        # MIDI iz adları latin-1 kodlanır; Türkçe karakterler ('ı') patlar.
        # Bu yüzden ASCII kimlik yazılır ("violin"), görünen ad raporlarda kalır.
        track.name = part.instrument.slug
        track.append(mido.Message("program_change", channel=part.channel, program=part.instrument.gm_program, time=0))
        # CC10 = pan; 64 merkez, 0 sol, 127 sağ
        track.append(
            mido.Message(
                "control_change", channel=part.channel, control=10, value=int(64 + part.pan * 63), time=0
            )
        )

        events: list[tuple[int, int, int, int]] = []
        for note in part.notes:
            events.append((note.start, 1, note.pitch, note.velocity))
            events.append((note.end, 0, note.pitch, 0))
        events.sort()

        previous = 0
        for tick, is_on, pitch, velocity in events:
            delta = tick - previous
            previous = tick
            if is_on:
                track.append(
                    mido.Message("note_on", channel=part.channel, note=pitch, velocity=velocity, time=delta)
                )
            else:
                track.append(
                    mido.Message("note_off", channel=part.channel, note=pitch, velocity=0, time=delta)
                )
        track.append(mido.MetaMessage("end_of_track", time=0))
        midi.tracks.append(track)

    midi.save(path)
    return path
