"""Üretim sonrası düzenleme testleri: aktarım, gama oturtma, kırpma."""

from __future__ import annotations

from ophelia.data.midi_io import NoteEvent
from ophelia.nlp.arrange import (
    AdjustReport,
    _nearest_scale_pitch,
    clip_to_duration,
    estimated_key,
    project_to_scale,
    transpose_to_key,
)
from ophelia.theory.scales import Key


def _note(start: int, pitch: int, duration: int = 480, velocity: int = 90) -> NoteEvent:
    return NoteEvent(start, duration, pitch, velocity, 0, 0, 0)


def _chord(root: int, *, offsets=(0, 4, 7), repeats: int = 4) -> list[NoteEvent]:
    notes: list[NoteEvent] = []
    for index in range(repeats):
        for offset in offsets:
            notes.append(_note(index * 480, root + offset))
    return notes


def test_estimated_key_of_c_major_chord() -> None:
    key = estimated_key(_chord(60))
    assert key is not None
    assert (key.tonic_pc, key.mode) == (0, "major")


def test_transpose_moves_to_requested_tonic() -> None:
    notes = _chord(60)  # C major
    moved, report = transpose_to_key(notes, Key(2, "major"))

    assert report.semitones == 2
    assert report.detected is not None
    assert {note.pitch for note in moved} == {62, 66, 69}
    assert not report.mode_mismatch


def test_transpose_uses_shortest_path() -> None:
    """C → A: yukarı 9 değil, aşağı 3 yarım ton (en kısa yol)."""
    moved, report = transpose_to_key(_chord(60), Key(9, "major"))
    assert report.semitones == -3
    assert {note.pitch for note in moved} == {57, 61, 64}


def test_transpose_folds_instead_of_leaving_range() -> None:
    low = [_note(0, 25)]
    moved, _ = transpose_to_key(low, Key(11, "major"))
    assert all(21 <= note.pitch <= 108 for note in moved)


def test_transpose_without_target_is_noop() -> None:
    notes = _chord(60)
    moved, report = transpose_to_key(notes, None)
    assert moved == notes
    assert report.semitones == 0


def test_mode_mismatch_is_reported_not_hidden() -> None:
    # C majör → la minör: aynı perdeler değil (üçlü farklı) → uyuşmazlık
    _, report = transpose_to_key(_chord(60), Key(9, "natural_minor"))
    assert report.mode_mismatch


def test_minor_piece_transposed_to_major_keeps_mismatch_flag() -> None:
    """Aktarım aralıkları korur: la minör +3 = do minör, do majör değil."""
    minor = _chord(57, offsets=(0, 3, 7))  # A minor triad
    moved, report = transpose_to_key(minor, Key(0, "major"))

    assert report.detected is not None
    assert report.detected.name == "A natural_minor"
    assert report.semitones == 3
    assert report.mode_mismatch

    # Aktarım sonrası perdeler do minör kümesi: projeksiyon gerçekten gerekli.
    assert {note.pitch % 12 for note in moved} == {0, 3, 7}
    _, before, after = project_to_scale(moved, Key(0, "major"))
    assert before == 4  # 12 notanın 4'ü mi bemol (3); do majörde yok
    assert after == 0


def test_project_to_scale_snaps_off_scale_notes() -> None:
    notes = [_note(0, 61), _note(480, 63), _note(960, 66)]  # C# D# F#: C majör gamında yok
    projected, before, after = project_to_scale(notes, Key(0, "major"))

    assert before == 3
    assert after == 0
    # Eşit uzaklıkta aşağıya çekilir: C# → C, D# → D, F# → F
    assert [note.pitch for note in projected] == [60, 62, 65]


def test_project_leaves_scale_notes_untouched() -> None:
    notes = _chord(60)
    projected, before, after = project_to_scale(notes, Key(0, "major"))
    assert (before, after) == (0, 0)
    assert projected == notes


def test_project_without_key_is_noop() -> None:
    notes = [_note(0, 61)]
    projected, before, after = project_to_scale(notes, None)
    assert projected == notes
    assert (before, after) == (0, 0)


def test_nearest_scale_pitch_prefers_downward_on_tie() -> None:
    assert _nearest_scale_pitch(61, {0, 2, 4, 5, 7, 9, 11}) == 60
    assert _nearest_scale_pitch(66, {0, 2, 4, 5, 7, 9, 11}) == 65


def test_clip_to_duration_drops_and_truncates() -> None:
    notes = [_note(0, 60, duration=480), _note(4320, 64, duration=480)]
    # 100 BPM'de 4 sn = 320 tick/beat... 4*100/60*480 = 3200 tick
    clipped, _ = clip_to_duration(notes, 4.0, tempo_bpm=100.0, ticks_per_beat=480)
    assert [note.start for note in clipped] == [0]
    assert clipped[0].duration == 480


def test_clip_shortens_note_crossing_the_limit() -> None:
    notes = [_note(3000, 60, duration=480)]
    clipped, _ = clip_to_duration(notes, 4.0, tempo_bpm=100.0, ticks_per_beat=480)
    assert clipped[0].start == 3000
    assert clipped[0].duration == 200


def test_clip_keeps_everything_when_short_enough() -> None:
    notes = _chord(60, repeats=2)
    clipped, _ = clip_to_duration(notes, 60.0, tempo_bpm=100.0)
    assert len(clipped) == 6


def test_scale_fit_reports_fraction_inside_key() -> None:
    notes = [_note(0, 60), _note(480, 61)]
    report = AdjustReport(target=Key(0, "major"))
    assert report.scale_fit(notes, Key(0, "major")) == 0.5
    assert report.scale_fit(notes, None) == 1.0
