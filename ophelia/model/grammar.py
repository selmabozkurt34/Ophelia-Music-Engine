"""Token dilbilgisi: üretimde yalnızca geçerli token türlerine izin verir.

Neden gerekli: model her adımda serbest örnekleme yaparsa ``DURATION`` yerine
``PITCH``, ya da ``PITCH`` olmadan ``DURATION`` üretebilir. Bu diziler
eğitimde hiç görülmediği için modelin kaybı yüksektir ve çözümleyici hata verir
— yani "MIDI'ye çevrilemeyen beste".

Dilbilgisi, 4. aşamadaki kodlayıcının ürettiği dilin tamamını kapsar::

    BOS → TIME → [PROGRAM] [CHANNEL] → (VELOCITY PITCH DURATION)+ → EOS
                  ↑ TIME her nota grubundan önce tekrar edebilir

Üretim sırasında her adımda geçersiz token'ların olasılığı ``-inf`` yapılır;
model yine kendi dağılımından örnekler, sadece dilbilgisi dışına çıkamaz.
Bu, çözümleyiciyi "kurtarma" mantığından daha temizdir: hatalı dizi hiç doğmaz.
"""

from __future__ import annotations

import torch

from ophelia.data.tokens import EOS, PAD, Vocabulary

# Durumlar (token türleriyle karışmasın diye ``after_`` ön ekli)
START = "start"  # BOS'tan sonra, henüz TIME yok
AFTER_TIME = "after_time"  # bir TIME'dan sonra, notalar gelebilir
AFTER_PROGRAM = "after_program"
AFTER_CHANNEL = "after_channel"
AFTER_VELOCITY = "after_velocity"
AFTER_PITCH = "after_pitch"
AFTER_DURATION = "after_duration"

# Token türleri (sözlükteki ön ekler)
TIME = "TIME"
DURATION = "DURATION"
VELOCITY = "VELOCITY"
PITCH = "PITCH"
PROGRAM = "PROGRAM"
CHANNEL = "CHANNEL"


class TokenGrammar:
    """Sözlüğe bağlı dilbilgisi; durum geçişleri ve izin maskeleri."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        *,
        min_duration_steps: int = 0,
        time_step_steps: int = 0,
    ) -> None:
        self.vocabulary = vocabulary
        self._by_kind = _ids_by_kind(vocabulary)
        self._masks: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}
        # Ritim politikası: ızgara birimi (1 = 16'lık). 0 = kısıtsız.
        self.min_duration_steps = max(0, min_duration_steps)
        self.time_step_steps = max(0, time_step_steps)

    # --- kimlik kümeleri ----------------------------------------------------
    def ids_of(self, kind: str) -> set[int]:
        """Türün izinli kimlikleri (ritim politikası uygulanmış)."""
        ids = self._by_kind.get(kind, set())
        if kind == "DURATION" and self.min_duration_steps:
            return {i for i in ids if self._value_of(i) >= self.min_duration_steps}
        if kind == "TIME" and self.time_step_steps:
            return {i for i in ids if self._value_of(i) % self.time_step_steps == 0}
        return ids

    def _value_of(self, token_id: int) -> int:
        """``DURATION_4`` → 4; sayı yoksa 0."""
        token = self.vocabulary.token_of(token_id)
        _, _, raw = token.partition("_")
        return int(raw) if raw.isdigit() else 0

    def _union(self, *kinds: str) -> list[int]:
        collected: set[int] = set()
        for kind in kinds:
            collected |= self.ids_of(kind)
        return sorted(collected)

    def _kind_of(self, token_id: int) -> str:
        token = self.vocabulary.token_of(token_id)
        return token.split("_", 1)[0] if "_" in token else token

    # --- dilbilgisi ---------------------------------------------------------
    def allowed_ids(self, state: str, *, notes: int, min_notes: int = 1) -> list[int]:
        """Durumda izinli token kimlikleri.

        ``notes < min_notes`` iken ``EOS`` yasak: ne boş eser ne de birkaç
        notalık kırıntı üretilebilir. Eşik, modelin değil bestecinin kararı.
        """
        if state == START:
            return self._union(TIME)
        if state == AFTER_TIME:
            allowed = self._union(TIME, PROGRAM, CHANNEL, VELOCITY)
        elif state == AFTER_PROGRAM:
            allowed = self._union(CHANNEL, VELOCITY, TIME)
        elif state == AFTER_CHANNEL:
            allowed = self._union(VELOCITY, TIME)
        elif state == AFTER_VELOCITY:
            allowed = self._union(PITCH)
        elif state == AFTER_PITCH:
            allowed = self._union(DURATION)
        elif state == AFTER_DURATION:
            allowed = self._union(VELOCITY, TIME)
        else:  # pragma: no cover - savunma
            raise ValueError(f"bilinmeyen durum: {state}")

        if notes >= max(1, min_notes):
            allowed.append(self.vocabulary.id_of(EOS))
        return sorted(allowed)

    def next_state(self, state: str, token_id: int) -> str:
        """Bir token üretildikten sonraki durum."""
        kind = self._kind_of(token_id)

        transitions = {
            TIME: AFTER_TIME,
            PROGRAM: AFTER_PROGRAM,
            CHANNEL: AFTER_CHANNEL,
            VELOCITY: AFTER_VELOCITY,
            PITCH: AFTER_PITCH,
            DURATION: AFTER_DURATION,
        }
        return transitions.get(kind, state)

    def state_after(self, generated: list[int] | tuple[int, ...]) -> tuple[str, int]:
        """Üretilmiş kimlik dizisinden (durum, nota sayısı) çıkarır."""
        state: str = START
        notes = 0
        for token_id in generated:
            kind = self._kind_of(token_id)
            if kind in (EOS, PAD):
                continue
            if kind == DURATION and state == AFTER_PITCH:
                notes += 1
            state = self.next_state(state, token_id)
        return state, notes

    def mask_for(
        self, generated: list[int] | tuple[int, ...], *, min_notes: int = 1
    ) -> torch.Tensor:
        """Dizinin devamı için izin maskesi (örnekleme döngüsüne verilir)."""
        state, notes = self.state_after(generated)
        return self.mask(state, notes=notes, min_notes=min_notes)

    def mask(self, state: str, *, notes: int, min_notes: int = 1) -> torch.Tensor:
        """Tüm sözlük üzerinde ``bool`` maske (True = izinli, önbellekli)."""
        allowed = self.allowed_ids(state, notes=notes, min_notes=min_notes)
        key = (state, tuple(allowed))
        cached = self._masks.get(key)
        if cached is not None:
            return cached

        mask = torch.zeros(self.vocabulary.size, dtype=torch.bool)
        if allowed:
            mask[torch.tensor(allowed, dtype=torch.long)] = True
        self._masks[key] = mask
        return mask


def _ids_by_kind(vocabulary: Vocabulary) -> dict[str, set[int]]:
    """Token adının ön ekine göre kimlik kümeleri (``PITCH_60`` → ``PITCH``)."""
    groups: dict[str, set[int]] = {}
    for index, token in enumerate(vocabulary.tokens):
        kind = token.split("_", 1)[0] if "_" in token else token
        groups.setdefault(kind, set()).add(index)
    return groups


def grammar_for(
    vocabulary: Vocabulary, *, min_duration_steps: int = 0, time_step_steps: int = 0
) -> TokenGrammar:
    """Sözlükten dilbilgisi kurar (isteğe bağlı ritim politikasıyla)."""
    return TokenGrammar(
        vocabulary,
        min_duration_steps=min_duration_steps,
        time_step_steps=time_step_steps,
    )
