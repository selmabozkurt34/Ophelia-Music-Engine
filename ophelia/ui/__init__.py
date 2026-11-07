"""Kullanıcı arayüzü (9. aşama): yerel web uygulaması.

Tek sayfa: Türkçe komut → model → dinlenebilir WAV + MIDI + MusicXML. Sunucu
`stdlib` ile yazıldı (yeni bağımlılık yok), varsayılan olarak yalnız
`127.0.0.1`'i dinler.
"""

from __future__ import annotations

from ophelia.ui.server import (
    ComposeRequest,
    UiError,
    default_composer,
    make_ui_server,
    serve,
)

__all__ = [
    "ComposeRequest",
    "UiError",
    "default_composer",
    "make_ui_server",
    "serve",
]
