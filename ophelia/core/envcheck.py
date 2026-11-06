"""1. aşama doğrulaması: donanım tespiti ve geliştirme ortamı kontrolü.

Bu modül hem ``scripts/check_env.py`` hem de ``ophelia env`` komutu tarafından
kullanılır; tüm kontroller saf Python ile yapılır ve torch opsiyoneldir.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from typing import Any

from ophelia.core.paths import (
    CONFIG_DIR,
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    ensure_dirs,
)

# (import adı, pip paket adı)
REQUIRED_MODULES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("torch", "torch"),
    ("mido", "mido"),
    ("pretty_midi", "pretty_midi"),
    ("music21", "music21"),
    ("soundfile", "soundfile"),
    ("yaml", "PyYAML"),
    ("typer", "typer"),
    ("rich", "rich"),
    ("tqdm", "tqdm"),
)

MIN_PYTHON: tuple[int, int] = (3, 11)


@dataclass(frozen=True)
class Check:
    """Tek bir doğrulama sonucu."""

    name: str
    ok: bool
    detail: str

    @property
    def symbol(self) -> str:
        return "[OK]" if self.ok else "[FAIL]"


def _total_ram_gb() -> float | None:
    """Toplam fiziksel belleği GB cinsinden döndürür (bağımlılıksız)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return round(status.ullTotalPhys / (1024**3), 1)
            return None

        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        kib = float(line.split()[1])
                        return round(kib / (1024**2), 1)
            return None
    except Exception:  # pragma: no cover - donanım sorgusu asla çökmemeli
        return None
    return None


def hardware_report() -> dict[str, Any]:
    """Makinenin model eğitimi açısından önemli özelliklerini toplar."""
    report: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": platform.processor() or "bilinmiyor",
        "cpu_count": os.cpu_count() or 1,
        "ram_gb": _total_ram_gb(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "torch": None,
        "cuda": False,
        "gpu": "yok / bilinmiyor",
    }

    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda"] = bool(torch.cuda.is_available())
        if report["cuda"]:
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
            )
        else:
            report["gpu"] = "CUDA aygıtı yok (CPU ile çalışılacak)"
    except Exception:
        report["torch"] = None

    return report


def check_python() -> Check:
    """Python sürümü en az 3.11 mi?"""
    version = sys.version_info
    ok = tuple(version[:2]) >= MIN_PYTHON
    wanted = ".".join(str(part) for part in MIN_PYTHON)
    return Check(
        "Python sürümü",
        ok,
        f"{version.major}.{version.minor}.{version.micro} (gerekli >= {wanted})",
    )


def check_venv() -> Check:
    """Betik bir sanal ortam içinden mi çalışıyor?"""
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    return Check("Sanal ortam (venv)", in_venv, sys.prefix)


def check_git() -> Check:
    """Git kurulu mu?"""
    path = shutil.which("git")
    return Check("Git", path is not None, path or "git bulunamadı")


def check_directories() -> Check:
    """Proje dizin yapısı hazır mı?"""
    ensure_dirs()
    missing = [
        str(directory)
        for directory in (CONFIG_DIR, DATA_DIR, MODELS_DIR, OUTPUTS_DIR)
        if not directory.is_dir()
    ]
    return Check(
        "Proje dizin yapısı",
        not missing,
        f"kök: {PROJECT_ROOT}" if not missing else f"eksik: {', '.join(missing)}",
    )


def check_modules() -> list[Check]:
    """Gerekli Python paketlerinin içe aktarılabilirliğini kontrol eder."""
    results: list[Check] = []
    for module_name, package_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "?")
            results.append(Check(package_name, True, str(version)))
        except Exception as exc:  # noqa: BLE001 - her hatayı raporlamalıyız
            results.append(Check(package_name, False, f"{type(exc).__name__}: {exc}"))
    return results


def check_fonts() -> Check:
    """Ses üretimi için sesfontu (SF2) bulunuyor mu?"""
    fonts_dir = PROJECT_ROOT / "assets" / "soundfonts"
    found = sorted(fonts_dir.glob("*.sf2")) if fonts_dir.is_dir() else []
    return Check(
        "Ses fontu (SF2)",
        bool(found),
        f"{len(found)} dosya: assets/soundfonts" if found else "henüz yok (7. aşamada gerekli)",
    )


def run_all_checks() -> list[Check]:
    """Tüm ortam kontrollerini çalıştırır."""
    checks = [
        check_python(),
        check_venv(),
        check_git(),
        check_directories(),
        *check_modules(),
        check_fonts(),
    ]
    return checks


def summarize(checks: list[Check]) -> tuple[int, int]:
    """(başarılı, toplam) sayılarını döndürür."""
    return sum(1 for check in checks if check.ok), len(checks)
