"""``ophelia compose`` komutunun davranışı."""

from __future__ import annotations

from pathlib import Path

from ophelia.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_compose_without_checkpoint_explains(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["compose", "la minör 30 saniye", "--model", str(tmp_path / "yok.pt")]
    )
    assert result.exit_code == 1
    assert "Checkpoint bulunamadı" in result.output


def test_compose_explain_shows_parsed_request(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "compose",
            "hüzünlü la minör 30 saniye",
            "--model",
            str(tmp_path / "yok.pt"),
            "--explain",
        ],
    )
    assert "tonalite" in result.output
    assert "A natural_minor" in result.output
    assert "30 sn" in result.output


def test_compose_help_lists_options() -> None:
    result = runner.invoke(app, ["compose", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--explain" in result.output
