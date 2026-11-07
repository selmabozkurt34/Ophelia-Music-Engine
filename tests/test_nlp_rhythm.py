"""Ritim politikası: dilbilgisi kısıtı, metrik düzeltme ve bölüm eşlemesi."""

from __future__ import annotations

from dataclasses import replace

from ophelia.data.midi_io import NoteEvent
from ophelia.data.tokens import MidiTokenizer
from ophelia.model.grammar import grammar_for
from ophelia.nlp.metrify import EIGHTH, metrify
from ophelia.nlp.parse import parse_command


def _note(start: int, duration: int, pitch: int = 60) -> NoteEvent:
    return NoteEvent(start, duration, pitch, 90, 0, 0, 0)


def _value(vocabulary, token_id: int) -> int:
    return int(vocabulary.token_of(token_id).split("_", 1)[1])


# --- dilbilgisi ------------------------------------------------------------


def test_grammar_can_forbid_sixteenths() -> None:
    vocabulary = MidiTokenizer().vocabulary
    loose = grammar_for(vocabulary)
    strict = grammar_for(vocabulary, min_duration_steps=2, time_step_steps=2)

    assert 1 in {_value(vocabulary, token) for token in loose.ids_of("DURATION")}
    assert min(_value(vocabulary, token) for token in strict.ids_of("DURATION")) >= 2
    assert all(_value(vocabulary, token) % 2 == 0 for token in strict.ids_of("TIME"))


def test_grammar_policy_does_not_touch_other_kinds() -> None:
    vocabulary = MidiTokenizer().vocabulary
    strict = grammar_for(vocabulary, min_duration_steps=2, time_step_steps=2)
    assert strict.ids_of("PITCH") == grammar_for(vocabulary).ids_of("PITCH")
    assert strict.ids_of("VELOCITY") == grammar_for(vocabulary).ids_of("VELOCITY")


# --- metrik düzeltme -------------------------------------------------------


def test_metrify_snaps_starts_to_the_grid() -> None:
    notes = [_note(120, EIGHTH), _note(370, 480), _note(610, 240)]
    snapped = metrify(notes, grid=EIGHTH, min_duration=EIGHTH)
    assert [note.start for note in snapped] == [240, 480, 720]
    assert all(note.start % EIGHTH == 0 for note in snapped)


def test_metrify_never_leaves_fragments_shorter_than_the_grid() -> None:
    """16'lık kırıntılar ya ızgaraya taşınır ya da düşer; kısa nota kalmaz."""
    notes = [_note(0, 480), _note(120, 120), _note(130, 60)]
    cleaned = metrify(notes, grid=EIGHTH, min_duration=EIGHTH)

    assert cleaned
    assert all(note.duration >= EIGHTH for note in cleaned)
    assert all(note.start % EIGHTH == 0 for note in cleaned)
    for first, second in zip(cleaned, cleaned[1:], strict=False):
        assert first.start + first.duration <= second.start


def test_metrify_keeps_one_note_per_pitch_and_start() -> None:
    long_note = _note(480, 480, pitch=62)
    short_note = replace(long_note, duration=240)
    cleaned = metrify([short_note, long_note], grid=EIGHTH, min_duration=EIGHTH)
    assert len(cleaned) == 1
    assert cleaned[0].duration == 480


def test_metrify_never_overlaps_the_same_pitch() -> None:
    cleaned = metrify([_note(0, 480), _note(240, 480)], grid=EIGHTH, min_duration=EIGHTH)
    assert len(cleaned) == 2
    assert cleaned[0].start + cleaned[0].duration <= cleaned[1].start


def test_metrify_is_a_noop_on_clean_input() -> None:
    notes = [_note(0, 480), _note(480, 240, pitch=64)]
    assert metrify(notes, grid=EIGHTH, min_duration=EIGHTH) == notes


def test_metrify_handles_empty_input() -> None:
    assert metrify([], grid=EIGHTH, min_duration=EIGHTH) == []


# --- "yaylılar" bir bölüm, tek pad değil ------------------------------------


def test_strings_request_expands_to_a_section() -> None:
    spec = parse_command("hüzünlü la minör, 20 saniye, yaylılar")
    assert spec.ensemble == ("violin", "viola", "cello", "contrabass")


def test_single_instrument_request_stays_single() -> None:
    spec = parse_command("la minör, 20 saniye, keman")
    assert spec.ensemble == ("violin",)
