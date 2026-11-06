"""1. aşama kabul testleri: ortam, yapılandırma ve CLI."""

from __future__ import annotations

import importlib
import sys

import pytest
from ophelia import __version__
from ophelia.cli.main import app
from ophelia.core.config import Config
from ophelia.core.envcheck import (
    MIN_PYTHON,
    REQUIRED_MODULES,
    hardware_report,
    run_all_checks,
    summarize,
)
from ophelia.core.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT
from typer.testing import CliRunner


def test_python_version_is_supported() -> None:
    assert tuple(sys.version_info[:2]) >= MIN_PYTHON


@pytest.mark.parametrize("module_name,package_name", REQUIRED_MODULES)
def test_required_package_importable(module_name: str, package_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None, f"{package_name} içe aktarılamadı"


def test_project_layout_exists() -> None:
    for directory in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        assert directory.is_dir(), f"eksik dizin: {directory}"
    assert (PROJECT_ROOT / "configs" / "default.yaml").is_file()


def test_default_config_loads() -> None:
    config = Config.load()
    assert config.project.seed == 42
    assert config.midi.time_signature == (4, 4)
    assert 20 <= config.midi.tempo_bpm <= 300
    assert config.audio.sample_rate in (22050, 44100, 48000)


def test_config_falls_back_to_defaults_when_missing(tmp_path) -> None:
    config = Config.load(tmp_path / "yok.yaml")
    assert config.project.name == "Ophelia Music Engine"


def test_hardware_report_has_core_fields() -> None:
    report = hardware_report()
    for key in ("os", "cpu", "cpu_count", "python", "torch"):
        assert key in report
    assert report["cpu_count"] >= 1


def test_all_environment_checks_pass() -> None:
    checks = run_all_checks()
    ok_count, total = summarize(checks)
    failing = [c.name for c in checks if not c.ok and not c.name.startswith("Ses fontu")]
    assert not failing, f"başarısız kontroller: {failing}"
    assert ok_count >= total - 1  # ses fontu 7. aşamaya kadar opsiyonel


def test_cli_info_command() -> None:
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output


def test_cli_env_command() -> None:
    result = CliRunner().invoke(app, ["env"])
    assert result.exit_code == 0, result.output
    assert "Ortam Kontrolleri" in result.output
