"""Token dilbilgisi (dilbilgisi kısıtlı örnekleme) testleri."""

from __future__ import annotations

import pytest
import torch
from ophelia.data.midi_io import NoteEvent
from ophelia.data.tokens import EOS, MidiTokenizer
from ophelia.model.config import ModelConfig, SampleConfig
from ophelia.model.generate import generate_tokens, tokens_to_notes
from ophelia.model.grammar import (
    AFTER_PITCH,
    AFTER_TIME,
    AFTER_VELOCITY,
    START,
    grammar_for,
)
from ophelia.model.transformer import OpheliaGPT


def _note(start: int, duration: int, pitch: int) -> NoteEvent:
    return NoteEvent(start, duration, pitch, 90, 0, 0, 0)


def test_grammar_blocks_notes_before_time() -> None:
    tokenizer = MidiTokenizer()
    grammar = grammar_for(tokenizer.vocabulary)
    vocab = tokenizer.vocabulary

    allowed = set(grammar.allowed_ids(START, notes=0))
    assert vocab.id_of("TIME_0") in allowed
    assert vocab.id_of("VELOCITY_90") not in allowed
    assert vocab.id_of(EOS) not in allowed  # boş eser yasak


def test_grammar_requires_velocity_pitch_duration_order() -> None:
    tokenizer = MidiTokenizer()
    grammar = grammar_for(tokenizer.vocabulary)
    vocab = tokenizer.vocabulary

    after_time = set(grammar.allowed_ids(AFTER_TIME, notes=0))
    assert vocab.id_of("VELOCITY_64") in after_time
    assert vocab.id_of("PITCH_60") not in after_time

    after_velocity = set(grammar.allowed_ids(AFTER_VELOCITY, notes=0))
    assert after_velocity == {vocab.id_of(f"PITCH_{value}") for value in range(128)}

    after_pitch = grammar.allowed_ids(AFTER_PITCH, notes=0)
    assert all(vocab.token_of(token_id).startswith("DURATION_") for token_id in after_pitch)


def test_grammar_allows_eos_only_after_min_notes() -> None:
    tokenizer = MidiTokenizer()
    grammar = grammar_for(tokenizer.vocabulary)
    vocab = tokenizer.vocabulary

    assert vocab.id_of(EOS) not in grammar.allowed_ids(AFTER_TIME, notes=0)
    assert vocab.id_of(EOS) in grammar.allowed_ids(AFTER_PITCH, notes=1)
    assert vocab.id_of(EOS) not in grammar.allowed_ids(AFTER_PITCH, notes=5, min_notes=16)
    assert vocab.id_of(EOS) in grammar.allowed_ids(AFTER_PITCH, notes=16, min_notes=16)


def test_grammar_accepts_every_encoded_file() -> None:
    """Kodlayıcının ürettiği her dizi dilbilgisi için geçerli olmalı."""
    tokenizer = MidiTokenizer()
    grammar = grammar_for(tokenizer.vocabulary)
    notes = [
        _note(0, 120, 60),
        _note(0, 120, 64),
        _note(240, 480, 62),
        _note(720, 120, 67),
    ]
    ids = tokenizer.vocabulary.to_ids(tokenizer.encode_notes(notes))

    for index, token_id in enumerate(ids[1:], start=1):
        state, notes = grammar.state_after(ids[1:index])
        allowed = grammar.allowed_ids(state, notes=notes)
        assert token_id in allowed, tokenizer.vocabulary.token_of(token_id)


def test_grammar_mask_matches_size() -> None:
    tokenizer = MidiTokenizer()
    grammar = grammar_for(tokenizer.vocabulary)

    mask = grammar.mask(START, notes=0)
    assert mask.dtype == torch.bool
    assert mask.shape == (tokenizer.vocabulary.size,)
    assert int(mask.sum()) == 1025  # yalnız TIME_0..1024


class InvalidScriptModel(OpheliaGPT):
    """Dilbilgisi dışı token'ları zorlayan model: TIME yerine PITCH ister."""

    def __init__(self, config: ModelConfig, target: int) -> None:
        super().__init__(config)
        self.target = target

    def forward(self, idx, targets=None, pad_id=-100):  # type: ignore[override]
        batch, length = idx.shape
        logits = torch.full((batch, length, self.config.vocab_size), -30.0)
        logits[..., self.target] = 30.0
        return logits, None


def test_constrained_generation_stays_inside_grammar() -> None:
    tokenizer = MidiTokenizer()
    vocab = tokenizer.vocabulary
    config = ModelConfig(
        vocab_size=vocab.size, context=32, d_model=16, n_layer=1, n_head=2, dropout=0.0
    )
    # Model PITCH_60'ı istiyor; başlangıçta bu token yasak olduğu için TIME gelir.
    model = InvalidScriptModel(config, vocab.id_of("PITCH_60"))

    tokens, _ = generate_tokens(
        model, tokenizer, SampleConfig(max_new_tokens=8, seed=1, constrained=True)
    )
    assert tokens[0] == "<bos>"
    assert tokens[1].startswith("TIME_")

    notes = tokens_to_notes(tokens, tokenizer)
    assert notes == [] or all(note.pitch >= 0 for note in notes)


def test_unconstrained_generation_can_break_grammar() -> None:
    tokenizer = MidiTokenizer()
    vocab = tokenizer.vocabulary
    config = ModelConfig(
        vocab_size=vocab.size, context=32, d_model=16, n_layer=1, n_head=2, dropout=0.0
    )
    model = InvalidScriptModel(config, vocab.id_of("PITCH_60"))

    tokens, _ = generate_tokens(
        model, tokenizer, SampleConfig(max_new_tokens=4, seed=1, constrained=False)
    )
    assert tokens[1] == "PITCH_60"  # kısıt yokken model dilbilgisi dışına çıkar


def test_sample_config_exposes_constraint() -> None:
    assert SampleConfig().constrained is True
    assert SampleConfig(constrained=False).to_dict()["constrained"] is False
    assert SampleConfig().min_notes == 120
    with pytest.raises(ValueError):
        SampleConfig(min_notes=0)


def test_min_notes_keeps_piece_going() -> None:
    """EOS isteyen modele rağmen parça min_notes'a kadar devam eder."""
    tokenizer = MidiTokenizer()
    vocab = tokenizer.vocabulary
    config = ModelConfig(
        vocab_size=vocab.size, context=64, d_model=16, n_layer=1, n_head=2, dropout=0.0
    )
    model = InvalidScriptModel(config, vocab.id_of(EOS))

    tokens, stopped = generate_tokens(
        model,
        tokenizer,
        SampleConfig(max_new_tokens=64, seed=2, min_notes=8),
    )
    notes = tokens_to_notes(tokens, tokenizer)
    assert not stopped or len(notes) >= 8


def test_grammar_unknown_state_rejected() -> None:
    grammar = grammar_for(MidiTokenizer().vocabulary)
    with pytest.raises(ValueError):
        grammar.allowed_ids("bilinmeyen", notes=0)
