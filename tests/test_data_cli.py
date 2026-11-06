"""``ophelia data ...`` komutlarının davranışı."""

from __future__ import annotations

from ophelia.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_data_sources_lists_verdicts() -> None:
    result = runner.invoke(app, ["data", "sources"])
    assert result.exit_code == 0, result.output
    assert "pdmx" in result.output
    assert "maestro" in result.output
    assert "approved" in result.output
    assert "rejected" in result.output


def test_data_sources_filters_approved() -> None:
    result = runner.invoke(app, ["data", "sources", "-s", "approved"])
    assert result.exit_code == 0, result.output
    assert "pdmx" in result.output
    assert "maestro" not in result.output


def test_data_sources_rejects_bad_filter() -> None:
    result = runner.invoke(app, ["data", "sources", "-s", "yok"])
    assert result.exit_code == 1
    assert "Geçersiz durum" in result.output


def test_data_build_rejects_noncommercial_source() -> None:
    result = runner.invoke(app, ["data", "build", "-s", "maestro"])
    assert result.exit_code == 1
    assert "onaylı değil" in result.output


def test_data_build_unknown_source() -> None:
    result = runner.invoke(app, ["data", "build", "-s", "olmayan-kaynak"])
    assert result.exit_code == 1
    assert "Hata" in result.output
