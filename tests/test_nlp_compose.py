"""Tohum motif ve komut→eser hattı testleri (sahte model ile, checkpoint gerekmez)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from ophelia.data.midi_io import analyze
from ophelia.data.tokens import BOS, MidiTokenizer
from ophelia.model.config import SampleConfig
from ophelia.nlp.compose import compose, generate_until, min_notes_for, tokens_for
from ophelia.nlp.parse import parse_command
from ophelia.nlp.seeding import prompt_ids, seed_notes, seed_tokens
from ophelia.nlp.spec import CompositionSpec
from ophelia.theory.scales import Key


class StubModel:
    """``OpheliaGPT`` yerine geçen sahte model: sabit bir cümlecik üretip susar."""

    config = type("Cfg", (), {"vocab_size": 0, "context": 256})()

    def __init__(self, tokenizer: MidiTokenizer, *, notes: int = 2, repeat: int = 2) -> None:
        self.tokenizer = tokenizer
        self.notes = notes
        self.repeat = repeat
        self.calls = 0

    def eval(self) -> StubModel:
        return self

    def generate(self, idx, max_new_tokens, *, stop_id=None, **_: object):
        self.calls += 1
        vocab = self.tokenizer.vocabulary
        chunk = ["TIME_0"]
        for index in range(self.notes):
            chunk += ["VELOCITY_90", "PITCH_60", "DURATION_1"]
            if index + 1 < self.notes:
                chunk += ["TIME_4"]
        ids = vocab.to_ids(chunk) + [vocab.id_of("<eos>")]
        extra = torch.tensor([ids], dtype=torch.long)
        return torch.cat([idx, extra], dim=1)


def _tokenizer() -> MidiTokenizer:
    return MidiTokenizer()


def test_seed_notes_give_tonic_triad() -> None:
    spec = CompositionSpec(key=Key(9, "natural_minor"))
    notes = seed_notes(spec)
    assert [note.pitch % 12 for note in notes] == [9, 0, 4, 9]  # A, C, E, A
    assert len(notes) == 4


def test_seed_notes_major_triad() -> None:
    notes = seed_notes(CompositionSpec(key=Key(0, "major")))
    assert [note.pitch % 12 for note in notes] == [0, 4, 7, 0]


def test_seed_tokens_roundtrip() -> None:
    tokenizer = _tokenizer()
    spec = CompositionSpec(key=Key(2, "major"))
    tokens = seed_tokens(tokenizer, spec)

    assert BOS not in tokens
    assert "<eos>" not in tokens
    rebuilt = tokenizer.decode_notes(tokens)
    assert [note.pitch for note in rebuilt] == [note.pitch for note in seed_notes(spec)]


def test_prompt_ids_start_with_bos() -> None:
    tokenizer = _tokenizer()
    ids = prompt_ids(tokenizer, CompositionSpec(key=Key(0, "major")))
    assert ids[0] == tokenizer.vocabulary.id_of(BOS)
    assert len(ids) > 5


def test_generate_until_keeps_going_past_eos() -> None:
    tokenizer = _tokenizer()
    model = StubModel(tokenizer, notes=2)
    spec = CompositionSpec(key=Key(0, "major"), tempo_bpm=100.0, duration_seconds=30.0)
    sample = SampleConfig(max_new_tokens=64, min_notes=1, seed=1)

    tokens, _, passes = generate_until(model, tokenizer, spec, sample=sample, max_passes=3)

    assert passes == 3  # her geçiş 8 nota ekliyor, 30 sn için yetmiyor
    assert "<eos>" not in tokens  # cümle sonları akışa yazılmaz
    notes = tokenizer.decode_notes(tokens)
    assert len(notes) > 8


def test_generate_until_stops_when_long_enough() -> None:
    tokenizer = _tokenizer()
    model = StubModel(tokenizer, notes=40)
    spec = CompositionSpec(key=Key(0, "major"), tempo_bpm=100.0, duration_seconds=8.0)
    sample = SampleConfig(max_new_tokens=256, min_notes=1, seed=1)

    tokens, _, passes = generate_until(model, tokenizer, spec, sample=sample, max_passes=4)
    assert passes == 1
    assert len(tokenizer.decode_notes(tokens)) >= 40


def test_token_budget_scales_with_duration() -> None:
    assert tokens_for(10) < tokens_for(60)
    assert tokens_for(600) <= 4096
    assert min_notes_for(60) > min_notes_for(10)


def test_compose_writes_midi_respecting_request(tmp_path: Path) -> None:
    tokenizer = _tokenizer()
    model = StubModel(tokenizer, notes=40)
    command = "la minör 120 bpm 20 saniye"
    target = tmp_path / "eser.mid"

    result = compose(command, model, tokenizer, out_path=target, seed=3)

    assert target.is_file()
    assert result.spec.key is not None
    assert result.spec.key.name == "A natural_minor"
    assert result.n_notes > 0

    stats = analyze(target)
    assert stats.n_notes == result.n_notes
    assert stats.tempos[0] == pytest.approx(500000)  # 120 BPM
    assert result.duration_seconds <= result.spec.duration_seconds + 1.0

    # İstenen gamda kalma oranı raporlanır ve garanti edilir.
    assert result.adjust.scale_fit(result.notes, result.spec.key) == 1.0


def test_compose_without_key_does_not_transpose(tmp_path: Path) -> None:
    tokenizer = _tokenizer()
    model = StubModel(tokenizer, notes=20)
    spec = CompositionSpec(tempo_bpm=100.0, duration_seconds=20.0)

    result = compose("", model, tokenizer, out_path=tmp_path / "a.mid", spec=spec)

    assert result.spec.key is None
    assert result.adjust.semitones == 0
    assert result.adjust.scale_fit(result.notes, None) == 1.0


def test_compose_is_reproducible(tmp_path: Path) -> None:
    tokenizer = _tokenizer()
    command = "do majör 100 bpm 15 saniye"

    first = compose(command, StubModel(tokenizer, notes=12), tokenizer, seed=5)
    second = compose(command, StubModel(tokenizer, notes=12), tokenizer, seed=5)

    assert [note.pitch for note in first.notes] == [note.pitch for note in second.notes]
    assert len(first.tokens) == len(second.tokens)


def test_parse_then_compose_keeps_tempo(tmp_path: Path) -> None:
    tokenizer = _tokenizer()
    result = compose(
        "neşeli bir do majör parça 140 bpm 12 saniye",
        StubModel(tokenizer, notes=30),
        tokenizer,
        out_path=tmp_path / "c.mid",
        seed=1,
    )
    assert result.spec.tempo_bpm == 140.0
    assert result.stats is not None
    assert result.stats.tempos[0] == pytest.approx(60_000_000 / 140.0, rel=1e-3)


def test_wav_request_arranges_midi_too(tmp_path: Path) -> None:
    """WAV istenince .mid de çok kanallı olmalı: iki dosya aynı eseri anlatsın."""
    import mido

    tokenizer = _tokenizer()
    target = tmp_path / "eser.mid"
    result = compose(
        "do majör 100 bpm 12 saniye",
        StubModel(tokenizer, notes=20),
        tokenizer,
        out_path=target,
        wav_path=tmp_path / "eser.wav",
        seed=2,
    )

    assert result.parts  # enstrümanlar atandı
    assert result.wav_path is not None and result.wav_path.is_file()

    midi = mido.MidiFile(str(target))
    assert len(midi.tracks) == len(result.parts) + 1


def test_compose_without_wav_stays_single_track(tmp_path: Path) -> None:
    import mido

    tokenizer = _tokenizer()
    target = tmp_path / "tek.mid"
    result = compose(
        "do majör 100 bpm 12 saniye",
        StubModel(tokenizer, notes=20),
        tokenizer,
        out_path=target,
        seed=2,
    )

    assert result.parts == []
    assert len(mido.MidiFile(str(target)).tracks) == 2  # yalnızca meta + tek iz


def test_compose_reports_request_details() -> None:
    tokenizer = _tokenizer()
    spec = parse_command("hüzünlü la minör 90 bpm 30 saniye")
    result = compose("", StubModel(tokenizer, notes=10), tokenizer, spec=spec)

    assert result.spec.notes  # çözümleme gerekçeleri taşınır
    assert result.adjust.detected is not None
