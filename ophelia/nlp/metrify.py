"""Ritmik düzeltme: üretilen notaları müzikal ızgaraya oturtur (6. aşama+).

Neden gerekli: tokenizer 16'lık ızgarada çalışıyor ve model, korallerden çok
daha fazla 16'lık üretiyor (ölçüm: gerçek korallerde %3, kısıtsız modelde %30).
Buna ek olarak ``limit_polyphony`` aynı perdede çakışan notayı kısaltırken
1 ızgaralık (16'lık) kırıntılar bırakıyor. İkisi birlikte çıktıyı koral değil
"oyun müziği" gibi duyuruyordu: kısa, sürekli değişen notalar.

Bu modül üretimden **sonra**, saf bir dönüşüm uygular:

1. Başlangıçlar ızgaraya yuvarlanır (varsayılan 8'lik).
2. Aynı perde+kanal+başlangıçta tek nota kalır (uzun olan).
3. Hiçbir nota ``min_duration``'dan kısa kalmaz; kısaltma yüzünden oluşan
   kırıntılar bir sonraki aynı perde notasına yer açmak için düşürülür.
"""

from __future__ import annotations

from dataclasses import replace

from ophelia.data.midi_io import NoteEvent

#: 480 PPQ'da 8'lik nota.
EIGHTH = 240


def metrify(
    notes: list[NoteEvent],
    *,
    grid: int = EIGHTH,
    min_duration: int = EIGHTH,
) -> list[NoteEvent]:
    """Notaları ``grid``e yuvarlar ve ``min_duration``'dan kısa notaları eler."""
    if not notes:
        return []
    grid = max(1, grid)
    min_duration = max(1, min_duration)

    snapped: list[NoteEvent] = []
    for note in notes:
        # Yarım ızgarayı yukarı yuvarla (120 → 240): nota ileri alınır, geri değil.
        start = int(note.start / grid + 0.5) * grid
        snapped.append(replace(note, start=start, duration=max(min_duration, note.duration)))

    # Aynı perde+kanal+başlangıç: uzun olanı tut (yuvarlama çakışma yaratabilir).
    best: dict[tuple[int, int, int], NoteEvent] = {}
    for note in snapped:
        key = (note.channel, note.pitch, note.start)
        current = best.get(key)
        if current is None or note.duration > current.duration:
            best[key] = note

    by_voice: dict[tuple[int, int], list[NoteEvent]] = {}
    for note in best.values():
        by_voice.setdefault((note.channel, note.pitch), []).append(note)

    out: list[NoteEvent] = []
    for group in by_voice.values():
        group.sort(key=lambda item: item.start)
        for index, note in enumerate(group):
            limit = group[index + 1].start if index + 1 < len(group) else None
            duration = note.duration
            if limit is not None:
                available = limit - note.start
                if available < min_duration:
                    continue  # kırıntı: sonraki notaya yer aç
                duration = min(duration, available)
            out.append(replace(note, duration=duration))

    return sorted(out, key=lambda item: (item.start, item.pitch))
