"""Tokenizasyon testleri: kayıpsızlık, ızgara, sözlük, hatalar."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ophelia.data.midi_io import NoteEvent
from ophelia.data.tokens import (
    BOS,
    EOS,
    MidiTokenizer,
    OffGridError,
    TokenError,
    TokenizerConfig,
    Vocabulary,
    build_vocabulary,
    dequantize_velocity,
    quantize_velocity,
    tokens_equal,
)


def _note(start: int = 0, duration: int = 120, pitch: int = 60, velocity: int = 90) -> NoteEvent:
    return NoteEvent(start, duration, pitch, velocity, 0, 0, 0)


def test_vocabulary_is_deterministic_and_unique() -> None:
    first = build_vocabulary(TokenizerConfig())
    second = build_vocabulary(TokenizerConfig())
    assert first.tokens == second.tokens
    assert len(set(first.tokens)) == len(first)
    assert BOS in first and EOS in first
    assert first.id_of(BOS) == 1


def test_vocabulary_size_matches_config() -> None:
    cfg = TokenizerConfig(max_time_units=16, max_duration_units=8)
    vocab = build_vocabulary(cfg)
    # 3 özel + 17 TIME + 8 DURATION + 128 VELOCITY + 128 PITCH + 128 PROGRAM + 16 CHANNEL
    assert vocab.size == 3 + 17 + 8 + 128 + 128 + 128 + 16


def test_vocabulary_roundtrip_and_errors() -> None:
    vocab = build_vocabulary()
    token_id = vocab.id_of("PITCH_60")
    assert vocab.token_of(token_id) == "PITCH_60"

    with pytest.raises(TokenError):
        vocab.id_of("YOK_1")
    with pytest.raises(TokenError):
        vocab.token_of(999_999)
    with pytest.raises(ValueError):
        Vocabulary(("A", "A"))


def test_vocabulary_save_and_load(tmp_path: Path) -> None:
    vocab = build_vocabulary(TokenizerConfig(max_time_units=4, max_duration_units=4))
    path = vocab.save(tmp_path / "vocab.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["size"] == vocab.size

    loaded = Vocabulary.load(path)
    assert loaded.tokens == vocab.tokens


def test_single_note_roundtrip() -> None:
    tokenizer = MidiTokenizer()
    notes = [_note(start=480, duration=240, pitch=64, velocity=77)]

    tokens = tokenizer.encode_notes(notes)
    assert tokens[0] == BOS and tokens[-1] == EOS
    assert "PITCH_64" in tokens
    assert "VELOCITY_77" in tokens
    assert "DURATION_2" in tokens
    assert tokens_equal(notes, tokenizer.decode_notes(tokens))


def test_chord_shares_one_time_token() -> None:
    tokenizer = MidiTokenizer()
    notes = [_note(0, 120, pitch) for pitch in (60, 64, 67)]

    tokens = tokenizer.encode_notes(notes)
    assert tokens.count("TIME_0") == 1
    assert tokens.count("VELOCITY_90") == 3
    assert tokens_equal(notes, tokenizer.decode_notes(tokens))


def test_empty_notes_yield_bos_eos() -> None:
    tokenizer = MidiTokenizer()
    tokens = tokenizer.encode_notes([])
    assert tokens == [BOS, EOS]
    assert tokenizer.decode_notes(tokens) == []


def test_multiple_notes_and_gaps() -> None:
    tokenizer = MidiTokenizer()
    notes = [
        _note(0, 120, 60),
        _note(120, 120, 62),
        _note(1200, 120, 64),
    ]
    tokens = tokenizer.encode_notes(notes)
    assert "TIME_1" in tokens
    assert "TIME_9" in tokens
    assert tokens_equal(notes, tokenizer.decode_notes(tokens))


def test_long_gap_splits_into_multiple_time_tokens() -> None:
    tokenizer = MidiTokenizer(TokenizerConfig(max_time_units=4))
    notes = [_note(0, 120, 60), _note(4 * 120 + 2 * 120, 120, 62)]  # 6 birim boşluk

    tokens = tokenizer.encode_notes(notes)
    assert tokens.count("TIME_4") == 1
    assert "TIME_2" in tokens
    assert tokens_equal(notes, tokenizer.decode_notes(tokens))


def test_program_and_channel_changes() -> None:
    tokenizer = MidiTokenizer()
    notes = [
        NoteEvent(0, 120, 60, 90, 0, 0, 0),
        NoteEvent(0, 120, 64, 90, 0, 42, 0),
        NoteEvent(120, 120, 67, 90, 3, 42, 0),
    ]
    tokens = tokenizer.encode_notes(notes)
    assert "PROGRAM_42" in tokens
    assert "CHANNEL_3" in tokens
    assert tokens_equal(notes, tokenizer.decode_notes(tokens))


def test_repeated_pitch_at_same_time_is_preserved() -> None:
    tokenizer = MidiTokenizer()
    notes = [_note(0, 120, 60, 50), _note(0, 240, 60, 100)]

    tokens = tokenizer.encode_notes(notes)
    rebuilt = tokenizer.decode_notes(tokens)
    assert len(rebuilt) == 2
    assert tokens_equal(notes, rebuilt)


def test_off_grid_start_raises_in_strict_mode() -> None:
    tokenizer = MidiTokenizer()
    with pytest.raises(OffGridError):
        tokenizer.encode_notes([_note(start=50, duration=120, pitch=60)])


def test_off_grid_duration_raises_in_strict_mode() -> None:
    tokenizer = MidiTokenizer()
    with pytest.raises(OffGridError):
        tokenizer.encode_notes([_note(start=0, duration=61, pitch=60)])


def test_non_strict_mode_rounds_to_grid() -> None:
    tokenizer = MidiTokenizer(TokenizerConfig(strict=False))
    notes = [_note(start=50, duration=70, pitch=60)]
    rebuilt = tokenizer.decode_notes(tokenizer.encode_notes(notes))

    assert rebuilt[0].start % 120 == 0
    assert rebuilt[0].duration % 120 == 0


def test_duration_over_limit_raises() -> None:
    tokenizer = MidiTokenizer(TokenizerConfig(max_duration_units=2))
    with pytest.raises(TokenError):
        tokenizer.encode_notes([_note(0, 3 * 120, 60)])


def test_velocity_bins_are_lossy_but_monotonic() -> None:
    assert quantize_velocity(90, None) == 90
    assert quantize_velocity(0, 4) == 0
    assert quantize_velocity(127, 4) == 3
    assert quantize_velocity(120, 4) >= quantize_velocity(10, 4)
    assert dequantize_velocity(0, 4) == 0
    assert dequantize_velocity(3, 4) == 127

    tokenizer = MidiTokenizer(TokenizerConfig(velocity_bins=4))
    notes = [_note(0, 120, 60, 127)]
    rebuilt = tokenizer.decode_notes(tokenizer.encode_notes(notes))
    assert rebuilt[0].velocity == 127


def test_decode_rejects_malformed_sequence() -> None:
    tokenizer = MidiTokenizer()
    with pytest.raises(TokenError):
        tokenizer.decode_notes([BOS, "DURATION_1", EOS])
    with pytest.raises(TokenError):
        tokenizer.decode_notes([BOS, "BOZUK"])


def test_ids_roundtrip() -> None:
    tokenizer = MidiTokenizer()
    tokens = tokenizer.encode_notes([_note(0, 120, 60)])
    ids = tokenizer.encode_ids(tokens)
    assert tokenizer.vocabulary.to_tokens(ids) == tokens


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        TokenizerConfig(grid_ticks=0)
    with pytest.raises(ValueError):
        TokenizerConfig(max_time_units=0)
    with pytest.raises(ValueError):
        TokenizerConfig(velocity_bins=1)


def test_tokens_equal_ignores_order() -> None:
    first = [_note(0, 120, 60), _note(0, 120, 64)]
    second = [_note(0, 120, 64), _note(0, 120, 60)]
    assert tokens_equal(first, second)
    assert not tokens_equal(first, [_note(0, 120, 60)])


def test_encode_file(tmp_path: Path, midi_file) -> None:
    path = midi_file("a.mid", [_note(0, 120, 60), _note(120, 120, 62)])
    tokens = MidiTokenizer().encode_file(path)
    assert tokens[0] == BOS
    assert "PITCH_62" in tokens
