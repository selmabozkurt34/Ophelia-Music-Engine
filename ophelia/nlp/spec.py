"""Beste isteği: doğal dil komutunun çözümlenmiş, doğrulanmış hâli.

``CompositionSpec`` komutun **anlamıdır**: tonalite, tempo, süre, doku. Üretim
hattı yalnızca bu nesneyi görür; doğal dil cümlesi ile model arasındaki tek
sözleşme budur. Bu ayrım sayesinde komut çözümlemesi model olmadan test
edilebiliyor ve ileride komut kaynağı değişse (LLM, ses, arayüz) hat
değişmiyor.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ophelia.theory.instruments import normalize_role
from ophelia.theory.scales import Key

# Komutun hiçbir sözcük vermediği durumda kullanılan varsayılanlar.
DEFAULT_TEMPO = 96.0
DEFAULT_DURATION = 45.0
# Modelin doğal dokusu 4-6 sesli; varsayılan sınır bunu kırpmamalı.
DEFAULT_POLYPHONY = 6


@dataclass(frozen=True)
class CompositionSpec:
    """Üretim için gereken her şey; hepsi isteğe bağlı olarak boş bırakılabilir."""

    command: str = ""
    key: Key | None = None
    tempo_bpm: float = DEFAULT_TEMPO
    duration_seconds: float = DEFAULT_DURATION
    max_polyphony: int = DEFAULT_POLYPHONY
    seed: int | None = None
    # Boşsa tek kanallı (modelin kendi dokusu); doluysa enstrümanlara dağıtılır.
    ensemble: tuple[str, ...] = ()
    # Boşsa düzenleme perde bölgesine göre; doluysa roller seslere atanır.
    roles: tuple[str, ...] = ()
    # Komutun hangi sözcükten ne çıkardığı (``--explain`` ve testler için).
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 30.0 <= self.tempo_bpm <= 300.0:
            raise ValueError(f"tempo 30-300 BPM arasında olmalı: {self.tempo_bpm}")
        if not 5.0 <= self.duration_seconds <= 600.0:
            raise ValueError(f"süre 5-600 sn arasında olmalı: {self.duration_seconds}")
        if not 1 <= self.max_polyphony <= 8:
            raise ValueError(f"polifoni 1-8 arasında olmalı: {self.max_polyphony}")
        if len(self.ensemble) > 15:
            raise ValueError("en fazla 15 enstrüman (9. kanal davul için ayrılmıştır)")
        if len(self.roles) > 15:
            raise ValueError("en fazla 15 rol (9. kanal davul için ayrılmıştır)")
        # Türkçe adlar da kabul edilir, saklanan değer her zaman kanoniktir.
        object.__setattr__(
            self, "roles", tuple(normalize_role(role) for role in self.roles)
        )

    def with_notes(self, *messages: str) -> CompositionSpec:
        return replace(self, notes=self.notes + tuple(messages))

    @property
    def beats(self) -> float:
        return self.duration_seconds * self.tempo_bpm / 60.0

    @property
    def bars(self) -> float:
        """4/4 varsayımıyla ölçü sayısı (veri seti %100 4/4)."""
        return self.beats / 4.0

    def to_dict(self) -> dict[str, object]:
        return {
            "command": self.command,
            "key": self.key.name if self.key else None,
            "tempo_bpm": self.tempo_bpm,
            "duration_seconds": self.duration_seconds,
            "max_polyphony": self.max_polyphony,
            "seed": self.seed,
            "ensemble": list(self.ensemble),
            "roles": list(self.roles),
            "notes": list(self.notes),
        }
