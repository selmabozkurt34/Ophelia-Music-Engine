"""Enstrümanlar: roller, MIDI program numaraları ve ses aralıkları.

``gm_program`` değerleri **0 tabanlıdır** (MIDI Program Change mesajındaki
değer). General MIDI listelerinde 1 tabanlı numaralar yaygındır; dönüştürme
gerekirse ``gm_program + 1`` kullanılır.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

MELODY: Final = "melody"
HARMONY: Final = "harmony"
BASS: Final = "bass"
RHYTHM: Final = "rhythm"
TEXTURE: Final = "texture"
PERCUSSION: Final = "percussion"

ROLES: tuple[str, ...] = (MELODY, HARMONY, BASS, RHYTHM, TEXTURE, PERCUSSION)

DRUM_CHANNEL = 9


@dataclass(frozen=True)
class Instrument:
    """Tek bir enstrüman tanımı."""

    slug: str
    name: str
    gm_program: int
    role: str
    low: int
    high: int
    velocity: int = 90
    is_drum: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.gm_program <= 127:
            raise ValueError(f"GM program 0-127 olmalı: {self.gm_program}")
        if self.role not in ROLES:
            raise ValueError(f"bilinmeyen rol: {self.role!r}")
        if not 0 <= self.low < self.high <= 127:
            raise ValueError(f"geçersiz ses aralığı: {self.low}-{self.high}")

    @property
    def midi_range(self) -> tuple[int, int]:
        return (self.low, self.high)

    def in_range(self, midi: int) -> bool:
        return self.low <= midi <= self.high

    def clamp(self, midi: int) -> int:
        """Perdeyi enstrümanın çalınabilir aralığına çeker."""
        return max(self.low, min(self.high, midi))

    def __str__(self) -> str:
        return f"{self.name} ({self.role}, GM {self.gm_program})"


INSTRUMENTS: dict[str, Instrument] = {
    instrument.slug: instrument
    for instrument in (
        Instrument("piano", "Piyano", 0, HARMONY, 21, 108, 88),
        Instrument("electric_piano", "Elektrik piyano", 4, HARMONY, 28, 100, 86),
        Instrument("harpsichord", "Klavsen", 6, HARMONY, 29, 89, 82),
        Instrument("music_box", "Müzik kutusu", 10, TEXTURE, 60, 96, 78),
        Instrument("vibraphone", "Vibrafon", 11, TEXTURE, 53, 89, 80),
        Instrument("nylon_guitar", "Klasik gitar", 24, HARMONY, 40, 88, 84),
        Instrument("steel_guitar", "Akustik gitar", 25, HARMONY, 40, 88, 86),
        Instrument("electric_guitar", "Elektrik gitar (clean)", 27, HARMONY, 40, 88, 86),
        Instrument("overdriven_guitar", "Elektrik gitar (distortion)", 29, RHYTHM, 40, 88, 92),
        Instrument("bass", "Bas gitar", 33, BASS, 28, 60, 92),
        Instrument("synth_bass", "Synth bas", 38, BASS, 24, 60, 90),
        Instrument("violin", "Keman", 40, MELODY, 55, 103, 88),
        Instrument("viola", "Viyola", 41, HARMONY, 48, 91, 84),
        Instrument("cello", "Çello", 42, HARMONY, 36, 76, 84),
        Instrument("contrabass", "Kontrbas", 43, BASS, 28, 67, 82),
        Instrument("strings", "Yaylı topluluğu", 48, TEXTURE, 36, 96, 80),
        Instrument("choir", "Koro", 52, TEXTURE, 48, 84, 76),
        Instrument("trumpet", "Trompet", 56, MELODY, 55, 82, 94),
        Instrument("french_horn", "Korno", 60, HARMONY, 41, 77, 84),
        Instrument("flute", "Flüt", 73, MELODY, 60, 96, 82),
        Instrument("synth_lead", "Synth lead", 80, MELODY, 48, 96, 88),
        Instrument("synth_pad", "Synth pad", 89, TEXTURE, 36, 96, 72),
        Instrument("drums", "Davul (kanal 10)", 0, PERCUSSION, 35, 81, 100, is_drum=True),
    )
}

DEFAULT_ENSEMBLE: tuple[str, ...] = (
    "piano",
    "strings",
    "bass",
    "synth_lead",
    "drums",
)


def get(slug: str) -> Instrument:
    """Enstrümanı adıyla getirir; bilinmeyen ad için öneri listeler."""
    try:
        return INSTRUMENTS[slug.strip().lower()]
    except KeyError as exc:
        available = ", ".join(sorted(INSTRUMENTS))
        raise ValueError(f"bilinmeyen enstrüman: {slug!r} (mevcut: {available})") from exc


def all_instruments() -> tuple[Instrument, ...]:
    return tuple(INSTRUMENTS.values())


def by_role(role: str) -> tuple[Instrument, ...]:
    """Belirli roldeki enstrümanlar (melody, harmony, bass, ...)."""
    normalized = role.strip().lower()
    if normalized not in ROLES:
        raise ValueError(f"bilinmeyen rol: {role!r} (mevcut: {', '.join(ROLES)})")
    return tuple(instrument for instrument in INSTRUMENTS.values() if instrument.role == normalized)


def resolve_many(slugs: list[str] | tuple[str, ...]) -> tuple[Instrument, ...]:
    return tuple(get(slug) for slug in slugs)


def default_ensemble() -> tuple[Instrument, ...]:
    return resolve_many(DEFAULT_ENSEMBLE)


# Rol → o rolün varsayılan enstrümanı. Komut yalnızca rol verdiğinde
# ("melodi ve bas") topluluk belirtilmemişse bu eşleme kullanılır.
ROLE_DEFAULTS: dict[str, str] = {
    MELODY: "violin",
    HARMONY: "viola",
    BASS: "contrabass",
    TEXTURE: "strings",
    RHYTHM: "piano",
    PERCUSSION: "drums",
}


# Rol adlarının Türkçe/İngilizce karşılıkları. Kanonik değerler ``ROLES``.
ROLE_ALIASES: dict[str, str] = {
    "melodi": MELODY,
    "melody": MELODY,
    "ezgi": MELODY,
    "solist": MELODY,
    "lead": MELODY,
    "armoni": HARMONY,
    "harmony": HARMONY,
    "eslik": HARMONY,
    "eslikci": HARMONY,
    "akor": HARMONY,
    "doku": TEXTURE,
    "texture": TEXTURE,
    "pedal": TEXTURE,
    "ritim": RHYTHM,
    "rhythm": RHYTHM,
    "bas": BASS,
    "bass": BASS,
    "bas hatti": BASS,
    "perkusyon": PERCUSSION,
    "davul": PERCUSSION,
    "percussion": PERCUSSION,
}


def normalize_role(role: str) -> str:
    """Türkçe/İngilizce rol adını kanonik role çevirir; bilinmeyende hata verir."""
    key = " ".join(role.strip().lower().split())
    if key in ROLE_ALIASES:
        return ROLE_ALIASES[key]
    if key in ROLES:
        return key
    raise ValueError(f"bilinmeyen rol: {role!r} (mevcut: {', '.join(ROLES)})")


def pick_for_role(
    role: str,
    candidates: tuple[Instrument, ...] | list[Instrument] = (),
    used: set[str] | frozenset[str] = frozenset(),
) -> Instrument:
    """Bir rolü çalacak enstrümanı seçer.

    Sıra: (1) adaylar içinde **aynı roldeki** ve henüz kullanılmamış enstrüman,
    (2) kalan adaylardan **orta noktası rolün doğal aralığına düşen** enstrüman,
    (3) hiçbiri uymuyorsa ``ROLE_DEFAULTS``. Böylece kullanıcının açıkça istediği
    enstrümanlar ("keman ve çello") korunur; çello bası çalacaksa bunu yapar,
    ama bas gitar melodiye konmaz.
    """
    normalized = normalize_role(role)

    pool = list(candidates)
    exact = [
        instrument
        for instrument in pool
        if instrument.role == normalized and instrument.slug not in used
    ]
    if exact:
        return exact[0]

    default = get(ROLE_DEFAULTS[normalized])
    if normalized == PERCUSSION:
        return default

    # Kalan aday ancak rolün doğal bölgesine "oturuyorsa" o rolü alabilir:
    # bas gitar (orta nokta 44) melodi çalamaz, çello (56) bas çalabilir.
    spare = [
        instrument
        for instrument in pool
        if instrument.slug not in used
        and not instrument.is_drum
        and default.in_range(round((instrument.low + instrument.high) / 2))
    ]
    if spare:
        center = (default.low + default.high) / 2
        if normalized == BASS:
            return min(spare, key=lambda instrument: instrument.low)
        if normalized == MELODY:
            return max(spare, key=lambda instrument: instrument.low)
        return min(
            spare, key=lambda instrument: abs((instrument.low + instrument.high) / 2 - center)
        )

    return get(ROLE_DEFAULTS[normalized])


def assign_channels(instruments: list[Instrument] | tuple[Instrument, ...]) -> dict[str, int]:
    """Her enstrümana MIDI kanalı atar; davul daima 9. kanaldadır.

    MIDI'de 16 kanal vardır (0-15) ve 9 numaralı kanal perküsyon içindir.
    Aynı enstrüman iki kez verilirse ikincisi yeni kanal alır.
    """
    if len(instruments) > 16:
        raise ValueError("en fazla 16 enstrüman seslendirilebilir")

    channels: dict[str, int] = {}
    used: set[int] = {DRUM_CHANNEL}
    next_channel = 0

    for index, instrument in enumerate(instruments):
        if instrument.is_drum:
            channels[f"{instrument.slug}_{index}"] = DRUM_CHANNEL
            continue
        while next_channel in used:
            next_channel += 1
        if next_channel > 15:
            raise ValueError("boş MIDI kanalı kalmadı")
        used.add(next_channel)
        channels[f"{instrument.slug}_{index}"] = next_channel
        next_channel += 1

    return channels
