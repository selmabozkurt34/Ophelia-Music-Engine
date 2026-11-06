"""Örnekleme ve MIDI üretimi testleri."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ophelia.data.midi_io import NoteEvent, analyze, iter_notes, load_midi
from ophelia.data.tokens import MidiTokenizer, TokenError
from ophelia.model.config import ModelConfig, SampleConfig
from ophelia.model.generate import (
    GenerationError,
    generate_piece,
    generate_tokens,
    limit_polyphony,
    tokens_to_notes,
)
from ophelia.model.transformer import OpheliaGPT


def _note(start: int, duration: int, pitch: int, velocity: int = 90) -> NoteEvent:
    return NoteEvent(start, duration, pitch, velocity, 0, 0, 0)


def test_limit_polyphony_caps_concurrent_notes() -> None:
    notes = [_note(0, 480, 60 + index, 90) for index in range(6)]
    limited = limit_polyphony(notes, 3)
    assert len(limited) == 3

    assert limit_polyphony(notes, 0) == notes
    assert limit_polyphony([], 3) == []


def test_limit_polyphony_frees_slots_after_release() -> None:
    notes = [
        _note(0, 120, 60),
        _note(0, 120, 64),
        _note(0, 120, 67),
        _note(120, 120, 72),
    ]
    limited = limit_polyphony(notes, 2)
    assert len(limited) == 3
    assert any(note.pitch == 72 for note in limited)


def test_limit_polyphony_prefers_louder_notes() -> None:
    notes = [_note(0, 480, 60, 10), _note(0, 480, 64, 120)]
    limited = limit_polyphony(notes, 1)
    assert [note.pitch for note in limited] == [64]


def test_tokens_to_notes_roundtrip() -> None:
    tokenizer = MidiTokenizer()
    notes = [_note(0, 120, 60), _note(120, 240, 64)]
    tokens = tokenizer.encode_notes(notes)

    rebuilt = tokens_to_notes(tokens, tokenizer)
    assert [(note.start, note.duration, note.pitch) for note in rebuilt] == [
        (0, 120, 60),
        (120, 240, 64),
    ]


def test_tokens_to_notes_truncates_after_eos() -> None:
    tokenizer = MidiTokenizer()
    tokens = tokenizer.encode_notes([_note(0, 120, 60)])
    noisy = tokens + ["PITCH_99", "DURATION_1"]

    rebuilt = tokens_to_notes(noisy, tokenizer)
    assert len(rebuilt) == 1


def test_tokens_to_notes_skips_malformed_fragments() -> None:
    """Bozuk parçalar üretimi çökertmez: atlanır, sağlam notalar korunur."""
    tokenizer = MidiTokenizer()
    tokens = ["<bos>", "TIME_0", "DURATION_3", "VELOCITY_80", "PITCH_60", "DURATION_1", "<eos>"]

    rebuilt = tokens_to_notes(tokens, tokenizer)
    assert [(note.pitch, note.duration) for note in rebuilt] == [(60, 120)]

    # Katı çözümleme hâlâ hata verir: kodlama doğrulaması gevşetilmedi.
    with pytest.raises(TokenError):
        tokenizer.decode_notes(tokens)


def test_tokens_to_notes_raises_when_nothing_decodes() -> None:
    tokenizer = MidiTokenizer()
    with pytest.raises(GenerationError):
        tokens_to_notes(["<bos>", "DURATION_2", "<eos>"], tokenizer)


class ScriptedModel(OpheliaGPT):
    """Testler için: sıradaki token'ı sabit bir senaryodan seçer."""

    def __init__(self, config: ModelConfig, script: list[int]) -> None:
        super().__init__(config)
        self.script = script

    def forward(self, idx, targets=None, pad_id=-100):  # type: ignore[override]
        batch, length = idx.shape
        position = min(length - 1, len(self.script) - 1)
        target = self.script[position]
        logits = torch.full((batch, length, self.config.vocab_size), -30.0)
        logits[..., target] = 30.0
        return logits, None


def _scripted_setup() -> tuple[ScriptedModel, MidiTokenizer, list[NoteEvent]]:
    tokenizer = MidiTokenizer()
    notes = [_note(0, 120, 60), _note(120, 120, 64), _note(240, 240, 67)]
    tokens = tokenizer.encode_notes(notes)

    vocab = tokenizer.vocabulary
    script = vocab.to_ids(tokens[1:])  # BOS'tan sonraki tüm token'lar
    config = ModelConfig(
        vocab_size=vocab.size, context=64, d_model=32, n_layer=1, n_head=2, dropout=0.0
    )
    return ScriptedModel(config, script), tokenizer, notes


def test_generate_tokens_follows_script_and_stops_at_eos() -> None:
    model, tokenizer, _ = _scripted_setup()
    tokens, stopped = generate_tokens(
        model, tokenizer, SampleConfig(max_new_tokens=64, min_notes=1)
    )

    assert stopped
    assert tokens[-1] == "<eos>"
    assert "PITCH_60" in tokens


def test_generate_piece_writes_playable_midi(tmp_path: Path) -> None:
    model, tokenizer, _ = _scripted_setup()
    target = tmp_path / "eser.mid"

    result = generate_piece(
        model,
        tokenizer,
        SampleConfig(max_new_tokens=64),
        midi_path=target,
    )

    assert result.midi_path == target
    assert target.is_file()
    assert result.n_notes >= 1

    stats = analyze(target)
    assert stats.n_notes == len(result.notes)
    reloaded = iter_notes(load_midi(target))
    assert {note.pitch for note in reloaded} == {note.pitch for note in result.notes}


def test_generation_is_reproducible_with_seed(tmp_path: Path) -> None:
    model, tokenizer, _ = _scripted_setup()
    config = SampleConfig(max_new_tokens=32, seed=7)

    first = generate_piece(model, tokenizer, config, midi_path=tmp_path / "a.mid")
    second = generate_piece(model, tokenizer, config, midi_path=tmp_path / "b.mid")

    assert first.tokens == second.tokens
    assert (tmp_path / "a.mid").read_bytes() == (tmp_path / "b.mid").read_bytes()


def test_generate_piece_without_writing(tmp_path: Path) -> None:
    model, tokenizer, _ = _scripted_setup()
    result = generate_piece(model, tokenizer, SampleConfig(max_new_tokens=16))
    assert result.midi_path is None
    assert result.n_tokens > 1
    assert result.duration_ticks >= 0
