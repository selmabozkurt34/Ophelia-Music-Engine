"""Üretim sonrası düzenleme: tonaliteye oturtma, süreye kırpma.

Modelin çıktısı müzikal olarak tutarlı ama **istenen** tonalitede olmayabilir.
Buradaki adımlar komutun sözünü tutar: eser istenen tonikte başlar/biter ve
istenen süreyi aşmaz. Hepsi notalar üzerinde saf fonksiyon; model olmadan
test edilebilir.
"""

from __future__ import annotations

from dataclasses import dataclass

from ophelia.data.midi_io import NoteEvent
from ophelia.theory.scales import Key, estimate_key

# Perdelerin kalacağı güvenli aralık (piyano sınırlarının içi).
MIN_PITCH = 21
MAX_PITCH = 108


@dataclass(frozen=True)
class AdjustReport:
    """Düzenleme özeti: ne yapıldı, neden."""

    detected: Key | None = None
    target: Key | None = None
    semitones: int = 0
    mode_mismatch: bool = False
    notes_before: int = 0
    notes_after: int = 0
    clipped_at_seconds: float | None = None
    off_scale_before: int = 0
    off_scale_after: int = 0

    @property
    def transposed(self) -> bool:
        return self.semitones != 0

    def scale_fit(self, notes: list[NoteEvent], key: Key | None) -> float:
        """Notaların istenen gamda kalma oranı (0-1); tonalite yoksa 1."""
        if key is None or not notes:
            return 1.0
        inside = sum(1 for note in notes if key.scale.contains(note.pitch))
        return inside / len(notes)


def estimated_key(notes: list[NoteEvent]) -> Key | None:
    """Notaların perde sınıfı histogramından tonalite tahmini."""
    if not notes:
        return None
    histogram: dict[int, float] = {}
    for note in notes:
        weight = note.duration / 480.0  # uzun notalar tonaliteyi daha çok belirler
        histogram[note.pitch % 12] = histogram.get(note.pitch % 12, 0.0) + weight
    return estimate_key(histogram)


def transpose_to_key(
    notes: list[NoteEvent], target: Key | None
) -> tuple[list[NoteEvent], AdjustReport]:
    """Notaları en yakın oktavla istenen toniğe taşır.

    Mod (majör/minör) aktarımla değiştirilemez; uyuşmazlık raporda bildirilir —
    sessizce "düzelttim" demek yanlış olurdu.
    """
    detected = estimated_key(notes)
    if target is None or detected is None or not notes:
        return notes, AdjustReport(detected=detected, target=target, notes_before=len(notes))

    offset = (target.tonic_pc - detected.tonic_pc) % 12
    if offset > 6:
        offset -= 12

    moved = _shift(notes, offset)
    # Aktarım **aralıkları** korur, modu değiştirmez: la minör bir ezgiyi do'dan
    # çalmak do minör verir. Bu yüzden mod farkı gerçek bir uyuşmazlıktır ve
    # ardından gelen diyatonik projeksiyonla (gam dışı perdelerin çekilmesi)
    # giderilir.
    mismatch = detected.mode != target.mode

    return moved, AdjustReport(
        detected=detected,
        target=target,
        semitones=offset,
        mode_mismatch=mismatch,
        notes_before=len(notes),
        notes_after=len(moved),
    )


def _shift(notes: list[NoteEvent], semitones: int) -> list[NoteEvent]:
    """Tüm notaları kaydırır; aralık dışına çıkanları oktavla geri çeker."""
    if semitones == 0:
        return list(notes)

    shifted: list[NoteEvent] = []
    for note in notes:
        pitch = note.pitch + semitones
        while pitch < MIN_PITCH:
            pitch += 12
        while pitch > MAX_PITCH:
            pitch -= 12
        shifted.append(
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
    return shifted


def project_to_scale(
    notes: list[NoteEvent], target: Key | None
) -> tuple[list[NoteEvent], int, int]:
    """Gam dışı perdeleri en yakın gam sesine çeker.

    Neden gerekli: aktarım yalnızca **toniği** değiştirir, **modu** değiştirmez.
    Model majör üretmişse istenen minör ancak perdeleri gamın içine çekerek
    sağlanır (diyatonik projeksiyon). Kesme en fazla bir yarım tondur.

    Dönen: (yeni notalar, önceki gam dışı sayısı, sonraki gam dışı sayısı).
    """
    if target is None or not notes:
        return notes, 0, 0

    scale = target.scale
    scale_pcs = set(scale.pitch_classes)
    before = sum(1 for note in notes if note.pitch % 12 not in scale_pcs)
    if before == 0:
        return notes, 0, 0

    snapped: list[NoteEvent] = []
    for note in notes:
        pitch = note.pitch
        if pitch % 12 not in scale_pcs:
            pitch = _nearest_scale_pitch(pitch, scale_pcs)
        snapped.append(
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

    after = sum(1 for note in snapped if note.pitch % 12 not in scale_pcs)
    return snapped, before, after


def _nearest_scale_pitch(pitch: int, scale_pcs: set[int]) -> int:
    """Gam dışı perdeyi en yakın gam sesine çeker (eşit uzaklıkta aşağıya)."""
    for distance in (1, 2, 3, 4, 5, 6):
        for candidate in (pitch - distance, pitch + distance):
            if candidate % 12 in scale_pcs and MIN_PITCH <= candidate <= MAX_PITCH:
                return candidate
    return pitch  # pragma: no cover - kromatik gam dışında olmaz


def clip_to_duration(
    notes: list[NoteEvent], seconds: float, *, tempo_bpm: float, ticks_per_beat: int = 480
) -> tuple[list[NoteEvent], float | None]:
    """Eseri istenen süreye kırpar; son notayı kesmek yerine bırakır.

    Sınırın **başında** kalan notalar korunur, süreleri sınıra kadar kısaltılır:
    böylece eser ortada kesilmez, sonundaki akor tamamlanır.
    """
    limit = int(seconds * tempo_bpm / 60.0 * ticks_per_beat)
    if not notes:
        return notes, None

    kept: list[NoteEvent] = []
    for note in notes:
        if note.start >= limit:
            continue
        duration = min(note.duration, limit - note.start)
        kept.append(
            NoteEvent(
                start=note.start,
                duration=max(1, duration),
                pitch=note.pitch,
                velocity=note.velocity,
                channel=note.channel,
                program=note.program,
                track=note.track,
            )
        )
    return kept, seconds
