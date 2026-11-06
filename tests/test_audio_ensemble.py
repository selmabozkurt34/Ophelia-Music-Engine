"""Enstrüman dağıtımı ve çok kanallı MIDI testleri."""

from __future__ import annotations

from pathlib import Path

import mido
import pytest
from ophelia.audio.ensemble import (
    Part,
    arrange,
    arrange_by_role,
    chorale_ensemble,
    fit_to_range,
    split_into_voices,
    write_multitrack_midi,
)
from ophelia.data.midi_io import NoteEvent, iter_notes, load_midi
from ophelia.theory.instruments import DRUM_CHANNEL, get, pick_for_role


def _note(start: int, pitch: int, duration: int = 240) -> NoteEvent:
    return NoteEvent(start, duration, pitch, 90, 0, 0, 0)


def _chords() -> list[NoteEvent]:
    """Dört sesli, iki akor: C majör ve G majör."""
    notes: list[NoteEvent] = []
    for index, chord in enumerate([(48, 55, 60, 64), (43, 50, 55, 59)]):
        notes.extend(_note(index * 480, pitch, 480) for pitch in chord)
    return notes


def test_split_preserves_every_note() -> None:
    notes = _chords()
    for mode in ("register", "chord"):
        voices = split_into_voices(notes, 4, mode=mode)
        assert sum(len(voice) for voice in voices) == len(notes)


def test_register_split_orders_high_to_low() -> None:
    voices = split_into_voices(_chords(), 2)
    high, low = voices
    assert min(note.pitch for note in high) >= max(note.pitch for note in low)


def test_chord_split_follows_satb() -> None:
    voices = split_into_voices(_chords(), 4, mode="chord")
    # İlk akor: 64 (soprano) 60 (alto) 55 (tenor) 48 (bas)
    assert [voice[0].pitch for voice in voices] == [64, 60, 55, 48]


def test_split_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        split_into_voices(_chords(), 2, mode="rastgele")
    with pytest.raises(ValueError):
        split_into_voices(_chords(), 0)


def test_split_of_empty_is_empty() -> None:
    assert split_into_voices([], 3) == [[], [], []]


def test_fit_to_range_shifts_by_octaves() -> None:
    violin = get("violin")  # 55-103
    moved, shifted = fit_to_range([_note(0, 48)], violin)
    assert shifted == 1
    assert moved[0].pitch == 60
    assert violin.in_range(moved[0].pitch)


def test_fit_to_range_leaves_in_range_untouched() -> None:
    piano = get("piano")
    notes = [_note(0, 60), _note(480, 72)]
    moved, shifted = fit_to_range(notes, piano)
    assert shifted == 0
    assert [note.pitch for note in moved] == [60, 72]


def test_fit_to_range_clamps_when_octaves_do_not_help() -> None:
    flute = get("flute")  # 60-96
    moved, shifted = fit_to_range([_note(0, 20)], flute)
    assert shifted == 1
    assert flute.low <= moved[0].pitch <= flute.high


def test_arrange_sorts_by_register() -> None:
    parts = arrange(_chords(), ("contrabass", "violin"))
    # Girdi sırası ters verilse de melodi tiz enstrümana gider.
    assert parts[0].instrument.slug == "violin"
    assert parts[1].instrument.slug == "contrabass"


def test_arrange_assigns_unique_channels_without_drums() -> None:
    parts = arrange(_chords(), ("piano", "drums"))
    channels = [part.channel for part in parts]
    assert len(set(channels)) == len(channels)
    assert DRUM_CHANNEL not in channels  # davul 9. kanalda değil, melodi çalmıyor


def test_arrange_pans_spread_across_stage() -> None:
    parts = arrange(_chords(), chorale_ensemble())
    pans = [part.pan for part in parts]
    assert pans == sorted(pans, reverse=True)
    assert all(-1.0 <= pan <= 1.0 for pan in pans)


def test_arrange_rejects_empty_ensemble() -> None:
    with pytest.raises(ValueError):
        arrange(_chords(), ())


def test_arrange_rejects_too_many_instruments() -> None:
    with pytest.raises(ValueError):
        arrange(_chords(), tuple(["piano"] * 16))


def test_multitrack_midi_roundtrip(tmp_path: Path) -> None:
    parts = arrange(_chords(), ("violin", "cello"))
    target = tmp_path / "cok.mid"
    write_multitrack_midi(parts, str(target), tempo_bpm=90.0)

    midi = mido.MidiFile(str(target))
    assert midi.type == 1
    assert len(midi.tracks) == len(parts) + 1  # meta + parça başına iz

    for index, part in enumerate(parts, start=1):
        track = midi.tracks[index]
        program = next(msg.program for msg in track if msg.type == "program_change")
        channel = next(msg.channel for msg in track if msg.type == "program_change")
        assert program == part.instrument.gm_program
        assert channel == part.channel

    notes = iter_notes(load_midi(target))
    assert len(notes) == sum(len(part.notes) for part in parts)


def test_track_names_are_ascii_safe(tmp_path: Path) -> None:
    """Türkçe enstrüman adları MIDI'nin latin-1 kodlamasında patlamamalı."""
    parts = arrange(_chords(), ("cello",))  # adı: "Çello"
    target = tmp_path / "ad.mid"
    write_multitrack_midi(parts, str(target))
    midi = mido.MidiFile(str(target))
    assert midi.tracks[1].name == "cello"


def test_part_pan_default_center() -> None:
    part = Part(instrument=get("piano"), notes=[])
    assert part.pan == 0.0
    assert part.slug == "piano"


def test_part_role_follows_instrument_by_default() -> None:
    assert Part(instrument=get("cello"), notes=[]).part_role == "harmony"
    assert Part(instrument=get("cello"), notes=[], role="bass").part_role == "bass"


# --- rol tabanlı düzenleme ---------------------------------------------------


def test_arrange_by_role_orders_melody_to_bass() -> None:
    parts = arrange_by_role(_chords(), ("bas", "melodi", "armoni"))
    # Girdi sırası ne olursa olsun melodi en tizde, bas en pestte.
    assert [part.part_role for part in parts] == ["melody", "harmony", "bass"]
    melody = [note.pitch for note in parts[0].notes]
    bass = [note.pitch for note in parts[2].notes]
    assert min(melody) > max(bass)


def test_arrange_by_role_preserves_every_note() -> None:
    notes = _chords()
    parts = arrange_by_role(notes, ("melodi", "armoni", "bas"))
    assert sum(len(part.notes) for part in parts) == len(notes)


def test_arrange_by_role_uses_default_instruments() -> None:
    parts = arrange_by_role(_chords(), ("melodi", "bas"))
    assert parts[0].instrument.slug == "violin"
    assert parts[1].instrument.slug == "contrabass"


def test_arrange_by_role_respects_requested_ensemble() -> None:
    parts = arrange_by_role(_chords(), ("melodi", "bas"), ("violin", "cello"))
    # Çello bas rolünde: rolü "harmony" olsa da enstrüman korunur.
    assert parts[0].instrument.slug == "violin"
    assert parts[1].instrument.slug == "cello"
    assert parts[1].part_role == "bass"


def test_arrange_by_role_keeps_out_of_register_instrument_out_of_role() -> None:
    """Bas gitarı melodiye koymak yerine rolün varsayılanı keman seçilir."""
    parts = arrange_by_role(_chords(), ("melodi", "bas"), ("bass",))
    assert parts[0].instrument.slug == "violin"
    assert parts[1].instrument.slug == "bass"


def test_arrange_by_role_assigns_unique_channels_without_drums() -> None:
    parts = arrange_by_role(_chords(), ("melodi", "armoni", "bas"))
    channels = [part.channel for part in parts]
    assert len(set(channels)) == len(channels)
    assert DRUM_CHANNEL not in channels


def test_arrange_by_role_pans_from_high_to_low() -> None:
    parts = arrange_by_role(_chords(), ("melodi", "armoni", "bas"))
    pans = [part.pan for part in parts]
    assert pans == sorted(pans, reverse=True)


def test_arrange_by_role_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="rol"):
        arrange_by_role(_chords(), ("melodi", "rastgele"))


def test_arrange_by_role_rejects_duplicate_role() -> None:
    with pytest.raises(ValueError, match="iki kez"):
        arrange_by_role(_chords(), ("melodi", "melodi"))


def test_arrange_by_role_rejects_empty_roles() -> None:
    with pytest.raises(ValueError):
        arrange_by_role(_chords(), ())


def test_arrange_by_role_rejects_percussion() -> None:
    """Davul kalıbı üretilmediği için perküsyon rolü sessizce çöp üretmemeli."""
    with pytest.raises(ValueError, match="davul"):
        arrange_by_role(_chords(), ("melodi", "perkusyon"))


def test_pick_for_role_defaults() -> None:
    assert pick_for_role("melody").slug == "violin"
    assert pick_for_role("bass").slug == "contrabass"
    assert pick_for_role("harmony").slug == "viola"


def test_pick_for_role_skips_used_instruments() -> None:
    assert pick_for_role("harmony", (get("viola"), get("cello")), {"viola"}).slug == "cello"
