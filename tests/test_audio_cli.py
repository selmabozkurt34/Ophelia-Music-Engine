"""``ophelia audio ...`` komutlarının davranışı."""

from __future__ import annotations

from pathlib import Path

from ophelia.cli.main import app
from ophelia.data.midi_io import NoteEvent, write_midi
from typer.testing import CliRunner

runner = CliRunner()


def _midi(tmp_path: Path) -> Path:
    notes = [
        NoteEvent(index * 480, 480, pitch, 90, 0, 0, 0)
        for index, chord in enumerate([(48, 55, 60, 64), (43, 50, 55, 59)])
        for pitch in chord
    ]
    path = tmp_path / "girdi.mid"
    write_midi(notes, path, tempo_bpm=90.0)
    return path


def test_audio_instruments_lists_catalog() -> None:
    result = runner.invoke(app, ["audio", "instruments"])
    assert result.exit_code == 0, result.output
    assert "violin" in result.output
    assert "contrabass" in result.output


def test_audio_instruments_rejects_bad_role() -> None:
    result = runner.invoke(app, ["audio", "instruments", "--role", "olmayan"])
    assert result.exit_code == 1
    assert "bilinmeyen rol" in result.output


def test_audio_render_writes_wav_and_multitrack(tmp_path: Path) -> None:
    midi = _midi(tmp_path)
    out = tmp_path / "ses.wav"

    result = runner.invoke(
        app, ["audio", "render", str(midi), "-o", str(out), "--ensemble", "violin,cello"]
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert out.with_suffix(".mid").is_file()
    assert "Keman" in result.output


def test_audio_render_rejects_unknown_instrument(tmp_path: Path) -> None:
    midi = _midi(tmp_path)
    result = runner.invoke(
        app, ["audio", "render", str(midi), "-o", str(tmp_path / "a.wav"), "-e", "uzayli"]
    )
    assert result.exit_code == 1
    assert "bilinmeyen enstrüman" in result.output


def test_audio_render_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["audio", "render", str(tmp_path / "yok.mid")])
    assert result.exit_code == 1
    assert "MIDI yok" in result.output
