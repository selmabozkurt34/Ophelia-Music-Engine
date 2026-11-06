"""MusicXML dışa aktarımı: yapı, nota sadakati ve CLI davranışı."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from ophelia.cli.main import app
from ophelia.data.midi_io import NoteEvent, write_midi
from ophelia.notation import (
    XmlPart,
    detect_fifths,
    fifths_for_key,
    midi_name,
    notes_to_musicxml,
    parts_from_midi,
    write_musicxml,
)
from typer.testing import CliRunner

runner = CliRunner()

MEASURE = 4 * 480  # 4/4, 480 PPQ


def _scale_notes(pitches: tuple[int, ...] = (60, 62, 64, 65, 67, 69, 71, 72)) -> list[NoteEvent]:
    return [
        NoteEvent(start=index * 480, duration=480, pitch=pitch, velocity=90, channel=0, program=0, track=0)
        for index, pitch in enumerate(pitches)
    ]


def _parse(text: str) -> ET.Element:
    return ET.fromstring(text.split("?>", 1)[1].split("\n", 1)[1])


def test_midi_name_spells_sharps_and_flats() -> None:
    assert midi_name(60) == ("C", 0, 4)
    assert midi_name(61) == ("C", 1, 4)
    assert midi_name(61, fifths=-3) == ("D", -1, 4)
    assert midi_name(58) == ("A", 1, 3)


def test_fifths_for_key_follows_circle_of_fifths() -> None:
    assert fifths_for_key(0) == 0  # C majör
    assert fifths_for_key(7) == 1  # G majör
    assert fifths_for_key(5) == -1  # F majör
    assert fifths_for_key(9, "natural_minor") == 0  # A minör → C majör
    assert fifths_for_key(2) == 2  # D majör


def test_score_declares_parts_attributes_and_tempo() -> None:
    text = notes_to_musicxml(
        [XmlPart("Keman", _scale_notes(), program=40, channel=0)],
        title="Deneme",
        tempo_bpm=96.0,
    )
    root = _parse(text)

    assert root.tag == "score-partwise"
    assert root.findtext("work/work-title") == "Deneme"
    score_part = root.find("part-list/score-part")
    assert score_part is not None
    assert score_part.findtext("part-name") == "Keman"
    assert score_part.findtext("midi-instrument/midi-program") == "41"  # 1 tabanlı

    measure = root.find("part/measure")
    assert measure is not None
    assert measure.findtext("attributes/divisions") == "480"
    assert measure.findtext("attributes/time/beats") == "4"
    assert measure.findtext("attributes/clef/sign") == "G"
    assert measure.find("direction/direction-type/metronome") is not None
    assert measure.find("direction").find("sound").get("tempo") == "96"  # type: ignore[union-attr]


def test_key_signature_is_detected_from_notes() -> None:
    # D majör (F#, C#) içerikli bir dizi → 2 diyez
    major = _scale_notes((62, 64, 66, 67, 69, 71, 73, 74))
    parts = [XmlPart("Piyano", major)]
    assert detect_fifths(parts) == 2
    assert "<fifths>2</fifths>" in notes_to_musicxml(parts)


def test_gap_is_filled_with_rests_and_measures_are_complete() -> None:
    notes = [
        NoteEvent(start=0, duration=480, pitch=60, velocity=90, channel=0, program=0, track=0),
        NoteEvent(start=3 * 480, duration=480, pitch=64, velocity=90, channel=0, program=0, track=0),
    ]
    root = _parse(notes_to_musicxml([XmlPart("Piyano", notes)]))
    measure = root.find("part/measure")
    assert measure is not None
    rests = measure.findall("note/rest/..")
    assert len(rests) == 1
    total = sum(int(item.text or 0) for item in measure.findall("note/duration"))
    assert total == MEASURE


def test_note_across_barline_is_split_with_tie() -> None:
    notes = [NoteEvent(start=3 * 480, duration=3 * 480, pitch=67, velocity=90, channel=0, program=0, track=0)]
    text = notes_to_musicxml([XmlPart("Piyano", notes)])

    assert text.index('type="start"') < text.index('type="stop"')
    root = _parse(text)
    measures = root.findall("part/measure")
    assert len(measures) == 2
    first = measures[0].find("note[tie]")
    assert first is not None
    assert first.find("tie").get("type") == "start"  # type: ignore[union-attr]
    assert first.findtext("duration") == "480"  # ölçü sonuna kadar
    second = measures[1].find("note[tie]")
    assert second is not None
    assert second.find("tie").get("type") == "stop"  # type: ignore[union-attr]
    assert second.findtext("duration") == "960"


def test_simultaneous_notes_become_chord() -> None:
    notes = [
        NoteEvent(start=0, duration=480, pitch=60, velocity=90, channel=0, program=0, track=0),
        NoteEvent(start=0, duration=480, pitch=64, velocity=90, channel=0, program=0, track=0),
        NoteEvent(start=0, duration=480, pitch=67, velocity=90, channel=0, program=0, track=0),
    ]
    root = _parse(notes_to_musicxml([XmlPart("Piyano", notes)]))
    measure = root.find("part/measure")
    assert measure is not None
    assert len(measure.findall("note/chord/..")) == 2  # akorun ilk notası chord taşımaz
    # Akor notaları süre ilerletmez: yalnızca chord taşımayanlar + suslar sayılır.
    advancing = [
        note.findtext("duration") for note in measure.findall("note") if note.find("chord") is None
    ]
    assert sum(int(item or 0) for item in advancing) == MEASURE


def test_output_is_deterministic() -> None:
    parts = [XmlPart("Keman", _scale_notes())]
    assert notes_to_musicxml(parts) == notes_to_musicxml(parts)


def test_empty_parts_rejected() -> None:
    with pytest.raises(ValueError, match="en az bir parti"):
        notes_to_musicxml([])


def test_parts_from_midi_groups_channels(tmp_path: Path) -> None:
    notes = [
        NoteEvent(start=0, duration=480, pitch=60, velocity=90, channel=0, program=40, track=0),
        NoteEvent(start=480, duration=480, pitch=48, velocity=90, channel=1, program=42, track=0),
    ]
    path = tmp_path / "girdi.mid"
    write_midi(notes, path, tempo_bpm=90.0)

    parts = parts_from_midi(path)
    assert len(parts) == 2
    assert [part.channel for part in parts] == [0, 1]
    assert [part.program for part in parts] == [40, 42]


def test_write_musicxml_reads_back_in_notation_library(tmp_path: Path) -> None:
    music21 = pytest.importorskip("music21")
    notes = _scale_notes() + [
        NoteEvent(start=7 * 480, duration=480, pitch=48, velocity=80, channel=1, program=42, track=1),
        NoteEvent(start=7 * 480, duration=480, pitch=55, velocity=80, channel=1, program=42, track=1),
    ]
    path = tmp_path / "eser.musicxml"
    write_musicxml([XmlPart("Keman", notes[:8]), XmlPart("Çello", notes[8:], 42, 1)], path)

    score = music21.converter.parse(str(path))
    assert len(score.parts) == 2
    for element in score.parts:
        element.stripTies(inPlace=True)

    def sounding(element: object) -> list[tuple[int, float]]:
        """music21 akoru tek nesnede toplar; perde bazında açalım."""
        pairs: list[tuple[int, float]] = []
        # Ölçü içi göreli ofset yerine partinin tam zaman çizgisi gerekir.
        for note in element.flatten().notes:  # type: ignore[attr-defined]
            offset = round(float(note.offset), 4)
            if hasattr(note, "pitches"):
                pairs.extend((pitch.midi, offset) for pitch in note.pitches)
            else:
                pairs.append((note.pitch.midi, offset))
        return sorted(pairs)

    assert sounding(score.parts[0]) == sorted(
        (pitch, float(index)) for index, pitch in enumerate((60, 62, 64, 65, 67, 69, 71, 72))
    )
    assert sounding(score.parts[1]) == [(48, 7.0), (55, 7.0)]


STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _read_part(part: ET.Element, measure_ticks: int = MEASURE) -> list[tuple[int, int, int]]:
    """MusicXML'i bağımsız olarak okur: akoru, geri sarmayı ve bağları çözer.

    Notasyon programlarının yaptığı işin asgarisi: ``<chord/>`` önceki nota ile
    aynı anda başlar, ``<backup>`` imleci geri alır, bağlı parçalar tek notadır.
    """
    events: list[dict[str, object]] = []
    clock = 0
    for measure in part.findall("measure"):
        position = 0
        last_start = 0
        for element in measure:
            if element.tag == "note":
                duration = int(element.findtext("duration") or 0)
                if element.find("chord") is not None:
                    start = last_start
                else:
                    start = position
                    last_start = position
                    position += duration
                if element.find("rest") is not None:
                    continue
                pitch = element.find("pitch")
                assert pitch is not None
                midi = (
                    STEPS[pitch.findtext("step") or "C"]
                    + int(pitch.findtext("alter") or 0)
                    + (int(pitch.findtext("octave") or 4) + 1) * 12
                )
                tie = element.find("tie")
                events.append(
                    {
                        "pitch": midi,
                        "voice": element.findtext("voice"),
                        "start": clock + start,
                        "duration": duration,
                        "tie": tie.get("type") if tie is not None else None,
                    }
                )
            elif element.tag == "backup":
                position -= int(element.findtext("duration") or 0)
        clock += measure_ticks

    notes: list[tuple[int, int, int]] = []
    open_ties: dict[tuple[object, object], dict[str, object]] = {}
    for event in sorted(events, key=lambda item: (item["start"], str(item["voice"]))):
        key = (event["voice"], event["pitch"])
        if event["tie"] == "start":
            open_ties[key] = dict(event)
        elif event["tie"] == "stop":
            head = open_ties.pop(key, None)
            assert head is not None, "bağın başı yok"
            notes.append(
                (
                    int(head["pitch"]),
                    int(head["start"]),
                    int(head["duration"]) + int(event["duration"]),
                )
            )
        else:
            notes.append((int(event["pitch"]), int(event["start"]), int(event["duration"])))
    assert not open_ties, "kapanmamış bağ"
    return sorted(notes)


def test_polyphony_and_barline_crossing_round_trip_exactly() -> None:
    """Üst üste binen sesler, akorlar ve ölçü taşan notalar bozulmadan dönmeli."""
    notes = [
        # bas hattı: ölçüyü aşan uzun nota (bağ gerekir)
        NoteEvent(start=1440, duration=1440, pitch=48, velocity=80, channel=0, program=0, track=0),
        # üstüne binen melodik çizgi
        NoteEvent(start=1440, duration=480, pitch=60, velocity=90, channel=0, program=0, track=0),
        NoteEvent(start=1560, duration=240, pitch=64, velocity=90, channel=0, program=0, track=0),
        # akor (aynı başlangıç + süre)
        NoteEvent(start=2400, duration=480, pitch=55, velocity=90, channel=0, program=0, track=0),
        NoteEvent(start=2400, duration=480, pitch=59, velocity=90, channel=0, program=0, track=0),
        NoteEvent(start=2400, duration=480, pitch=62, velocity=90, channel=0, program=0, track=0),
        # akorun üstüne binen ince bir nota
        NoteEvent(start=2640, duration=120, pitch=67, velocity=90, channel=0, program=0, track=0),
    ]
    text = notes_to_musicxml([XmlPart("Piyano", notes)])
    root = _parse(text)
    part = root.find("part")
    assert part is not None

    expected = sorted((note.pitch, note.start, note.duration) for note in notes)
    assert _read_part(part) == expected


def test_cli_export_writes_score(tmp_path: Path) -> None:
    midi = tmp_path / "girdi.mid"
    write_midi(_scale_notes(), midi, tempo_bpm=90.0)
    out = tmp_path / "eser.musicxml"

    result = runner.invoke(app, ["notation", "export", str(midi), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "MusicXML" in out.read_text(encoding="utf-8")
    assert "Keman" in result.output or "nota" in result.output


def test_cli_export_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["notation", "export", str(tmp_path / "yok.mid")])
    assert result.exit_code == 1
    assert "MIDI yok" in result.output
