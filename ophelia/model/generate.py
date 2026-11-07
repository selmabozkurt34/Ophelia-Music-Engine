"""Token üretimi ve üretilen token'ların MIDI'ye çevrilmesi.

Akış: ``<bos>`` → otoregresif örnekleme → token dizisi → ``decode_notes`` →
``write_midi``. Üretim, modelin öğrendiği sözlükle **birebir aynı** tokenizer'ı
kullanmak zorundadır; uyumsuzluk hata verir.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch

from ophelia.core.logging import get_logger
from ophelia.data.midi_io import NoteEvent, write_midi
from ophelia.data.tokens import BOS, EOS, MidiTokenizer, TokenError
from ophelia.model.config import SampleConfig
from ophelia.model.grammar import grammar_for
from ophelia.model.transformer import OpheliaGPT

log = get_logger(__name__)


class GenerationError(Exception):
    """Üretilen token dizisi çözümlenemedi."""


@dataclass
class GenerationResult:
    """Üretim çıktısı."""

    tokens: list[str]
    notes: list[NoteEvent]
    midi_path: Path | None = None
    stopped_at_eos: bool = False

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def n_notes(self) -> int:
        return len(self.notes)

    @property
    def duration_ticks(self) -> int:
        return max((note.end for note in self.notes), default=0)


def limit_polyphony(notes: list[NoteEvent], limit: int) -> list[NoteEvent]:
    """Eşzamanlı nota sayısını sınırlar (düşük velocity'li notalar düşer).

    Kurallar:

    * ``limit``'ten fazla nota seslendirilmez.
    * Aynı kanal+perde için iki nota **üst üste çalamaz**: MIDI'de ikinci
      ``note_on`` ilk notayı belirsizleştirir (hangi ``note_off`` hangi notayı
      kapatıyor?) ve okuyucular notayı kaybeder. Bu yüzden önceki nota,
      yenisinin başlangıcına kadar **kısaltılır** — nota düşürülmez. Notayı
      atmak melodiyi delik deşik ediyordu (185 notanın 85'i kayboluyordu).
    * Birebir aynı anda, aynı perdede başlayan notalardan yalnızca biri kalır.
    """
    if not notes:
        return notes

    ordered = sorted(notes, key=lambda note: (note.start, -note.velocity, note.pitch))
    active: list[NoteEvent] = []
    kept: list[NoteEvent] = []

    for note in ordered:
        active = [candidate for candidate in active if candidate.end > note.start]
        if limit > 0 and len(active) >= limit:
            continue

        conflict = None
        for index, candidate in enumerate(kept):
            if candidate.pitch != note.pitch or candidate.channel != note.channel:
                continue
            if candidate.start == note.start:
                conflict = "duplicate"
                break
            if candidate.end > note.start:
                kept[index] = replace(
                    candidate, duration=max(1, note.start - candidate.start)
                )

        if conflict == "duplicate":
            continue

        active = [candidate for candidate in active if candidate.end > note.start]
        kept.append(note)
        active.append(note)

    return sorted(kept, key=lambda note: (note.start, note.pitch))


def _prompt_ids(tokenizer: MidiTokenizer) -> torch.Tensor:
    bos_id = tokenizer.vocabulary.id_of(BOS)
    return torch.tensor([[bos_id]], dtype=torch.long)


def generate_tokens(
    model: OpheliaGPT,
    tokenizer: MidiTokenizer,
    config: SampleConfig,
    *,
    device: str = "cpu",
    prompt_tokens: list[int] | None = None,
) -> tuple[list[str], bool]:
    """``<bos>``'tan (ya da verilen tohumdan) başlayıp token üretir.

    ``prompt_tokens`` verilirse üretim onun devamıdır (5. aşamada tonalite
    tohumu, 6. aşamada komutun tonalitesi için kullanılır).
    """
    model.eval()
    generator = torch.Generator(device=device)
    if config.seed is not None:
        generator.manual_seed(config.seed)
    else:  # pragma: no cover - rastgele tohum
        generator.seed()

    if prompt_tokens:
        prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    else:
        prompt = _prompt_ids(tokenizer).to(device)

    # Tohumun ürettiği durum, üretimin başlangıç durumudur: aksi hâlde dilbilgisi
    # tohumun içine düşmüş gibi davranıp yanlış token'ları yasaklardı.
    seed_state = [int(value) for value in prompt[0].tolist()]
    while seed_state and seed_state[0] == tokenizer.vocabulary.id_of(BOS):
        seed_state = seed_state[1:]

    eos_id = tokenizer.vocabulary.id_of(EOS)

    allowed = None
    if config.constrained:
        grammar = grammar_for(
            tokenizer.vocabulary,
            min_duration_steps=config.min_duration_steps,
            time_step_steps=config.time_step_steps,
        )
        min_notes = config.min_notes
        allowed = lambda generated: grammar.mask_for(  # noqa: E731
            seed_state + generated, min_notes=min_notes
        )

    with torch.no_grad():
        output = model.generate(
            prompt,
            config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            stop_id=eos_id,
            generator=generator,
            allowed=allowed,
        )

    ids = [int(value) for value in output[0].tolist()]
    tokens = tokenizer.vocabulary.to_tokens(ids)
    stopped = bool(tokens) and tokens[-1] == EOS
    return tokens, stopped


def tokens_to_notes(tokens: list[str], tokenizer: MidiTokenizer) -> list[NoteEvent]:
    """Token dizisini notalara çevirir.

    Model dilbilgisi kısıtsız da çalışabilir (``constrained=False``); o durumda
    eksik nota parçaları (``PITCH`` olmadan ``DURATION``) hata değil, atlanır —
    aksi hâlde üretim çöker ve hiç MIDI yazılmaz.
    """
    trimmed = list(tokens)
    if EOS in trimmed:
        trimmed = trimmed[: trimmed.index(EOS) + 1]

    try:
        return tokenizer.decode_notes(trimmed)
    except TokenError as exc:
        log.warning("bozuk token parçası atlandı: %s", exc)

    notes = tokenizer.decode_notes(trimmed, strict=False)
    if not notes:
        raise GenerationError("üretilen token dizisi hiç nota içermiyor")
    return notes


def generate_piece(
    model: OpheliaGPT,
    tokenizer: MidiTokenizer,
    config: SampleConfig,
    *,
    midi_path: str | Path | None = None,
    tempo_bpm: float = 100.0,
    device: str = "cpu",
    prompt_tokens: list[int] | None = None,
) -> GenerationResult:
    """Eser üretir; ``midi_path`` verilirse MIDI olarak yazar."""
    tokens, stopped = generate_tokens(
        model, tokenizer, config, device=device, prompt_tokens=prompt_tokens
    )
    notes = tokens_to_notes(tokens, tokenizer)
    notes = limit_polyphony(notes, config.max_polyphony)

    result = GenerationResult(
        tokens=tokens, notes=notes, stopped_at_eos=stopped
    )

    if midi_path is not None:
        target = Path(midi_path)
        write_midi(notes, target, tempo_bpm=tempo_bpm)
        result.midi_path = target

    return result
