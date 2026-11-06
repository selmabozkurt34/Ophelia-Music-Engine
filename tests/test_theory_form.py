"""Motif dönüşümleri ve form şablonları testleri."""

from __future__ import annotations

import pytest
from ophelia.theory.form import (
    FORM_TEMPLATES,
    Motif,
    Section,
    build_form,
    resolve_form_name,
    transform,
)


def test_motif_from_midi() -> None:
    motif = Motif.from_midi([60, 62, 64], [1, 1, 2])
    assert motif.intervals == (0, 2, 4)
    assert motif.durations == (1, 1, 2)
    assert motif.length == 3
    assert motif.span == pytest.approx(4.0)
    assert motif.midi(67) == (67, 69, 71)


def test_motif_validation() -> None:
    with pytest.raises(ValueError):
        Motif((0, 2), (1,))
    with pytest.raises(ValueError):
        Motif((), ())
    with pytest.raises(ValueError):
        Motif((0, 2), (1, 0))


def test_motif_contour() -> None:
    assert Motif((0, 2, 5, 9), (1, 1, 1, 1)).contour == "+++"
    assert Motif((0, 2, 5, 1), (1, 1, 1, 1)).contour == "++-"
    assert Motif((0, 0, 2), (1, 1, 1)).contour == "0+"


def test_motif_transformations() -> None:
    motif = Motif((0, 2, 4), (1, 1, 2))
    assert motif.inverted().intervals == (0, -2, -4)
    assert motif.retrograde().intervals == (4, 2, 0)
    assert motif.retrograde().durations == (2, 1, 1)
    assert motif.retrograde_inversion().intervals == (-4, -2, 0)
    assert motif.transposed(5).intervals == (5, 7, 9)
    assert motif.augmented(2).durations == (2, 2, 4)
    assert motif.diminished(2).durations == (0.5, 0.5, 1.0)


def test_motif_sequence() -> None:
    motif = Motif((0, 2), (1, 1))
    assert motif.sequence(60, steps=2, semitones=2) == (60, 62, 62, 64)
    with pytest.raises(ValueError):
        motif.sequence(60, steps=0)


def test_motif_notes_timing() -> None:
    motif = Motif((0, 2, 4), (1, 1, 2))
    events = motif.notes(start_pitch=60, start_beat=4.0)
    assert [event.start for event in events] == [4.0, 5.0, 6.0]
    assert [event.pitch for event in events] == [60, 62, 64]
    assert events[-1].end == pytest.approx(8.0)

    repeated = motif.notes(repeat=2)
    assert len(repeated) == 6
    assert repeated[3].start == pytest.approx(4.0)
    assert repeated[-1].end == pytest.approx(8.0)

    with pytest.raises(ValueError):
        motif.notes(repeat=0)


def test_section_validation() -> None:
    assert Section("A", 8).bars == 8
    with pytest.raises(ValueError):
        Section("A", 0)


def test_build_form() -> None:
    form = build_form("AABA")
    assert form.name == "AABA"
    assert form.total_bars == 32
    assert form.labels == ("A", "A", "B", "A")
    assert form.unique_labels == ("A", "B")
    assert form.offsets == (0, 8, 16, 24)
    assert form.repetition_counts() == {"A": 3, "B": 1}


def test_form_section_lookup() -> None:
    form = build_form("verse_chorus")
    assert form.section_at(0).label == "intro"
    assert form.section_at(5).label == "verse"
    assert form.section_at(form.total_bars - 1).label == "outro"
    with pytest.raises(ValueError):
        form.section_at(form.total_bars)


def test_form_aliases_and_errors() -> None:
    assert resolve_form_name("pop") == "verse_chorus"
    assert resolve_form_name("blues") == "twelve_bar_blues"
    with pytest.raises(ValueError):
        resolve_form_name("olmayan_form")


def test_all_templates_build() -> None:
    for name in FORM_TEMPLATES:
        form = build_form(name)
        assert form.total_bars == sum(bars for _, bars in FORM_TEMPLATES[name])
        assert len(form.offsets) == len(form.sections)


def test_form_to_dict() -> None:
    payload = build_form("ABAB").to_dict()
    assert payload["total_bars"] == 32
    assert isinstance(payload["sections"], list)
    assert "ABAB" in build_form("ABAB").describe()


def test_transform_dispatch() -> None:
    motif = Motif((0, 2, 4), (1, 1, 1))
    assert transform(motif, "invert").intervals == (0, -2, -4)
    assert transform(motif, "transpose", semitones=3).intervals == (3, 5, 7)
    assert transform(motif, "augment", factor=2).durations == (2, 2, 2)
    with pytest.raises(ValueError):
        transform(motif, "yok_boyle_islem")
