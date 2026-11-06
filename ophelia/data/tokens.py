"""MIDI ↔ token temsili (olay tabanlı, REMI benzeri).

Tasarım hedefleri:

* **Kayıpsız**: aynı ızgara üzerinde ``encode → decode`` notaları birebir
  geri verir (başlangıç, süre, perde, velocity, kanal, program).
* **Kompakt**: nota başına 3 token (velocity, pitch, duration); akorlar tek
  ``TIME`` altında gruplanır.
* **Denetlenebilir**: ızgara dışı veri sessizce bozulmaz; ``strict`` modda hata
  verir. Veri hattı zaten 16'lık ızgaraya oturttuğu için kayıp oluşmaz.

Sözlük düzeni (sabit sıra, deterministik)::

    PAD, BOS, EOS,
    TIME_<0..max_time_units>,
    DURATION_<1..max_duration_units>,
    VELOCITY_<0..127>,
    PITCH_<0..127>,
    PROGRAM_<0..127>,
    CHANNEL_<0..15>
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ophelia.data.midi_io import NoteEvent, iter_notes, load_midi

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

SPECIAL_TOKENS: tuple[str, ...] = (PAD, BOS, EOS)

PITCH_COUNT = 128
VELOCITY_COUNT = 128
PROGRAM_COUNT = 128
CHANNEL_COUNT = 16


class TokenError(Exception):
    """Token dizisi çözümlenemedi."""


class OffGridError(TokenError):
    """Nota zamanı ızgaraya oturmuyor (kayıpsız kodlama mümkün değil)."""


@dataclass(frozen=True)
class TokenizerConfig:
    """Tokenizasyon parametreleri."""

    grid_ticks: int = 120  # 480 PPQ'da 16'lık nota
    max_time_units: int = 1024
    max_duration_units: int = 1024
    velocity_bins: int | None = None  # None -> tam velocity (kayıpsız)
    strict: bool = True  # ızgara dışı veride hata ver

    def __post_init__(self) -> None:
        if self.grid_ticks <= 0:
            raise ValueError("grid_ticks pozitif olmalı")
        if self.max_time_units <= 0 or self.max_duration_units <= 0:
            raise ValueError("token sınırları pozitif olmalı")
        if self.velocity_bins is not None and not 2 <= self.velocity_bins <= 128:
            raise ValueError("velocity_bins 2-128 arasında olmalı")


@dataclass(frozen=True)
class Vocabulary:
    """Sıralı token listesi; id ↔ token dönüşümü."""

    tokens: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("sözlükte yinelenen token var")

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: object) -> bool:
        return token in self._index

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def _index(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def id_of(self, token: str) -> int:
        try:
            return self._index[token]
        except KeyError as exc:
            raise TokenError(f"bilinmeyen token: {token!r}") from exc

    def token_of(self, token_id: int) -> str:
        try:
            return self.tokens[token_id]
        except IndexError as exc:
            raise TokenError(f"token id aralık dışında: {token_id}") from exc

    def to_ids(self, tokens: list[str] | tuple[str, ...]) -> list[int]:
        return [self.id_of(token) for token in tokens]

    def to_tokens(self, ids: list[int] | tuple[int, ...]) -> list[str]:
        return [self.token_of(token_id) for token_id in ids]

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"size": self.size, "tokens": list(self.tokens)}
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(payload["tokens"]))


def _velocity_token(value: int) -> str:
    return f"VELOCITY_{value}"


def _parse_token(token: str) -> tuple[str, int]:
    """``"PITCH_60"`` -> ``("PITCH", 60)``."""
    head, _, tail = token.rpartition("_")
    if not head:
        raise TokenError(f"geçersiz token: {token!r}")
    try:
        return head, int(tail)
    except ValueError as exc:
        raise TokenError(f"geçersiz token: {token!r}") from exc


def build_vocabulary(config: TokenizerConfig | None = None) -> Vocabulary:
    """Yapılandırmaya göre deterministik sözlük üretir."""
    cfg = config or TokenizerConfig()
    tokens: list[str] = list(SPECIAL_TOKENS)
    tokens.extend(f"TIME_{index}" for index in range(cfg.max_time_units + 1))
    tokens.extend(f"DURATION_{index}" for index in range(1, cfg.max_duration_units + 1))

    if cfg.velocity_bins is None:
        tokens.extend(_velocity_token(value) for value in range(VELOCITY_COUNT))
    else:
        tokens.extend(_velocity_token(value) for value in range(cfg.velocity_bins))

    tokens.extend(f"PITCH_{value}" for value in range(PITCH_COUNT))
    tokens.extend(f"PROGRAM_{value}" for value in range(PROGRAM_COUNT))
    tokens.extend(f"CHANNEL_{value}" for value in range(CHANNEL_COUNT))
    return Vocabulary(tuple(tokens))


def quantize_velocity(velocity: int, bins: int | None) -> int:
    """Velocity'yi kutu sayısına indirger (``None`` ise dokunmaz)."""
    if bins is None:
        return max(0, min(127, velocity))
    clamped = max(0, min(127, velocity))
    return min(bins - 1, round(clamped * (bins - 1) / 127))


def dequantize_velocity(bucket: int, bins: int | None) -> int:
    """Kutu indeksini velocity'ye geri çevirir (kutulama kayıplıdır)."""
    if bins is None:
        return bucket
    if bins <= 1:
        return 0
    return min(127, round(bucket * 127 / (bins - 1)))


class MidiTokenizer:
    """MIDI notalarını token dizisine çevirir ve geri kurar."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self.vocabulary = build_vocabulary(self.config)

    # --- yardımcı -----------------------------------------------------------
    @property
    def grid(self) -> int:
        return self.config.grid_ticks

    def _units(self, ticks: int, *, what: str) -> int:
        if ticks % self.grid == 0:
            return ticks // self.grid
        if self.config.strict:
            raise OffGridError(
                f"{what} ızgaraya oturmuyor: {ticks} tick, ızgara {self.grid} tick"
            )
        return round(ticks / self.grid)

    def _time_tokens(self, units: int) -> list[str]:
        """Uzun boşlukları birden fazla TIME token'ına böler."""
        tokens: list[str] = []
        remaining = units
        while remaining > self.config.max_time_units:
            tokens.append(f"TIME_{self.config.max_time_units}")
            remaining -= self.config.max_time_units
        tokens.append(f"TIME_{remaining}")
        return tokens

    def _duration_tokens(self, units: int) -> list[str]:
        if units > self.config.max_duration_units:
            raise TokenError(
                f"süre token sınırını aşıyor: {units} birim "
                f"(max {self.config.max_duration_units})"
            )
        return [f"DURATION_{units}"]

    # --- kodlama ------------------------------------------------------------
    def encode_notes(self, notes: list[NoteEvent] | tuple[NoteEvent, ...]) -> list[str]:
        """Notaları token dizisine çevirir (BOS ... EOS)."""
        tokens: list[str] = [BOS]
        if not notes:
            tokens.append(EOS)
            return tokens

        ordered = sorted(notes, key=lambda note: (note.start, note.program, note.channel, note.pitch))
        current_time = 0
        current_program: int | None = None
        current_channel: int | None = None

        index = 0
        while index < len(ordered):
            note = ordered[index]
            group_key = (note.start, note.program, note.channel)
            group: list[NoteEvent] = []
            while index < len(ordered):
                candidate = ordered[index]
                if (candidate.start, candidate.program, candidate.channel) != group_key:
                    break
                group.append(candidate)
                index += 1

            delta_units = self._units(note.start - current_time, what="başlangıç")
            tokens.extend(self._time_tokens(delta_units))
            current_time = note.start

            if note.program != current_program:
                tokens.append(f"PROGRAM_{note.program}")
                current_program = note.program
            if note.channel != current_channel:
                tokens.append(f"CHANNEL_{note.channel}")
                current_channel = note.channel

            for member in group:
                velocity = quantize_velocity(member.velocity, self.config.velocity_bins)
                tokens.append(_velocity_token(velocity))
                tokens.append(f"PITCH_{member.pitch}")
                tokens.extend(
                    self._duration_tokens(self._units(member.duration, what="süre"))
                )

        tokens.append(EOS)
        return tokens

    def decode_notes(
        self, tokens: list[str] | tuple[str, ...], *, strict: bool = True
    ) -> list[NoteEvent]:
        """Token dizisinden notaları geri kurar.

        ``strict=False`` yalnızca **model çıktısı** için: eksik parçalar (``PITCH``
        olmadan ``DURATION``, yarım kalan nota) atlanır. Kodlama doğrulaması
        (4. aşama gidiş-dönüş) her zaman ``strict=True`` kullanır; orada eksik
        parça bir hata olarak görülmelidir.
        """
        notes: list[NoteEvent] = []
        time_units = 0
        program = 0
        channel = 0
        velocity: int | None = None
        pitch: int | None = None

        for token in tokens:
            if token == PAD:
                continue
            if token == BOS:
                time_units, program, channel = 0, 0, 0
                velocity = pitch = None
                notes = []
                continue
            if token == EOS:
                break

            kind, value = _parse_token(token)

            if kind == "TIME":
                if not strict:
                    velocity = pitch = None  # yarım kalan parça zamanla birlikte düşer
                time_units += value
            elif kind == "PROGRAM":
                program = value
            elif kind == "CHANNEL":
                channel = value
            elif kind == "VELOCITY":
                velocity = dequantize_velocity(value, self.config.velocity_bins)
            elif kind == "PITCH":
                pitch = value
            elif kind == "DURATION":
                if pitch is None or velocity is None:
                    if not strict:
                        velocity = pitch = None
                        continue
                    raise TokenError("DURATION öncesinde PITCH/VELOCITY yok")
                notes.append(
                    NoteEvent(
                        start=time_units * self.grid,
                        duration=value * self.grid,
                        pitch=pitch,
                        velocity=velocity,
                        channel=channel,
                        program=program,
                        track=0,
                    )
                )
                velocity = pitch = None
            else:  # pragma: no cover - sözlük dışı token
                raise TokenError(f"bilinmeyen token türü: {kind}")

        return notes

    # --- dosya --------------------------------------------------------------
    def encode_file(self, path: str | Path) -> list[str]:
        return self.encode_notes(iter_notes(load_midi(path)))

    def encode_ids(self, tokens: list[str] | tuple[str, ...]) -> list[int]:
        return self.vocabulary.to_ids(tokens)


def _canonical(notes: list[NoteEvent] | tuple[NoteEvent, ...]) -> list[tuple[int, ...]]:
    """Kodlamanın kullandığı sıraya göre kanonik biçim.

    Aynı başlangıçta farklı program/kanal varsa dosya sırası müziksel bilgi
    taşımaz; karşılaştırma kanonik sırayla yapılır.
    """
    return sorted(
        (
            note.start,
            note.program,
            note.channel,
            note.pitch,
            note.duration,
            note.velocity,
        )
        for note in notes
    )


def tokens_equal(left: list[NoteEvent], right: list[NoteEvent]) -> bool:
    """İki nota listesi aynı notaları içeriyor mu (kanonik sırayla)?"""
    if len(left) != len(right):
        return False
    return _canonical(left) == _canonical(right)
