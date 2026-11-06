"""Konsol yardımcıları.

Windows konsolları varsayılan olarak yerel kod sayfasını (örn. cp1254) kullanır ve
Türkçe karakterler bozulur. Rich'in renkli çıktısını korurken akışları UTF-8'e
geçirmek için tüm giriş noktaları :func:`make_console` kullanır.
"""

from __future__ import annotations

import contextlib
import sys

from rich.console import Console

_configured = False


def setup_utf8_output() -> None:
    """stdout/stderr akışlarını UTF-8'e geçirir (birden fazla çağrıda güvenli)."""
    global _configured
    if _configured:
        return

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # Bazı akışlar salt okunur olabilir; sessizce yok sayılır.
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")

    _configured = True


def make_console() -> Console:
    """UTF-8 çıktısı ayarlanmış bir Rich konsolu döndürür."""
    setup_utf8_output()
    return Console()
