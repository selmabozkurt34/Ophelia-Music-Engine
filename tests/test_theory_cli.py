"""``ophelia theory`` komutunun uçtan uca davranışı."""

from __future__ import annotations

from ophelia.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_theory_default_output() -> None:
    result = runner.invoke(app, ["theory"])
    assert result.exit_code == 0, result.output
    assert "C major" in result.output
    assert "Diyatonik akorlar" in result.output
    assert "C" in result.output and "G" in result.output
    assert "Kadans" in result.output


def test_theory_accepts_preset_name() -> None:
    result = runner.invoke(app, ["theory", "-k", "A minor", "-p", "pop", "--seventh"])
    assert result.exit_code == 0, result.output
    assert "A natural_minor" in result.output


def test_theory_accepts_numeric_degrees() -> None:
    result = runner.invoke(app, ["theory", "-p", "1-5-6-4"])
    assert result.exit_code == 0, result.output
    assert "Am" in result.output


def test_theory_rejects_bad_key() -> None:
    result = runner.invoke(app, ["theory", "-k", "H major"])
    assert result.exit_code == 1
    assert "Geçersiz tonalite" in result.output


def test_theory_rejects_bad_progression() -> None:
    result = runner.invoke(app, ["theory", "-p", "1-X-5"])
    assert result.exit_code == 1
    assert "Geçersiz progresyon" in result.output
