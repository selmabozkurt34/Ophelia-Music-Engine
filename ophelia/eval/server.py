"""Kör test sunucusu: sayfa + sesler + oy kaydı, tek yerel süreçte.

Tarayıcı yalnızca üç şeyi görür: dinleyici sayfası, opak adlı ses dosyaları ve
deneme listesi. **Kaynak etiketleri (gerçek/üretim) sunulmaz**; ``key.json``
yalnızca diskte kalır. Sunucu varsayılan olarak yalnızca ``127.0.0.1``e bağlanır:
oylar başka bir makineden gelmez, ağ üzerinden test dağıtılmaz.
"""

from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

from ophelia.eval.abtest import (
    AUDIO_DIR,
    PAGE_FILE,
    TRIALS_FILE,
    AbTestError,
    Session,
    Vote,
    load_session,
)
from ophelia.eval.report import (
    REPORT_ROUTE,
    live_report_html,
    report_dict,
    report_visible_to,
)

MAX_BODY = 4096
LOOPBACK = "127.0.0.1"
LAN = "0.0.0.0"


class AbTestServer(ThreadingHTTPServer):
    """Oturumu ve oy sayacını taşıyan sunucu."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], session: Session) -> None:
        self.session = session
        self.lock = threading.Lock()
        self.votes = 0
        super().__init__(address, VoteHandler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/"


class VoteHandler(BaseHTTPRequestHandler):
    """GET: sayfa/liste/ses · POST: oy kaydı."""

    server_version = "OpheliaAB/1.0"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - imza sabit
        """Her isteği basmayalım; dinleyicinin konsolu temiz kalsın."""

    # --- yardımcılar -------------------------------------------------------

    @property
    def ab_server(self) -> AbTestServer:
        return cast(AbTestServer, self.server)

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _error(self, message: str, status: HTTPStatus) -> None:
        self._json({"error": message}, status)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self._error("dosya yok", HTTPStatus.NOT_FOUND)
            return
        self._send(body, content_type)

    # --- yönlendirme -------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server API'si
        session = self.ab_server.session
        route = self.path.split("?", 1)[0].lstrip("/")

        if route in ("", PAGE_FILE):
            self._send_file(session.directory / PAGE_FILE, "text/html; charset=utf-8")
            return
        if route == TRIALS_FILE:
            self._json(session.public_trials())
            return
        if route.startswith(f"{AUDIO_DIR}/"):
            name = route.split("/", 1)[1]
            allowed = {trial.a for trial in session.trials} | {trial.b for trial in session.trials}
            if name not in allowed:
                self._error("bilinmeyen ses", HTTPStatus.NOT_FOUND)
                return
            self._send_file(session.audio_dir / name, "audio/wav")
            return
        if route in (REPORT_ROUTE, f"{REPORT_ROUTE}.json"):
            # Rapor yalnız sunucunun kendi bilgisayarından: canlı sonuç dinleyiciyi
            # yanıltmasın (oy verecek kişi sonucu görmemeli).
            if not report_visible_to(self.client_address[0]):
                self._error("kaynakta rapor yalnız sunucu bilgisayarından açılır", HTTPStatus.FORBIDDEN)
                return
            if route.endswith(".json"):
                from ophelia.eval.report import build_report

                self._json(report_dict(build_report(session.directory)))
                return
            self._send(live_report_html(session).encode("utf-8"), "text/html; charset=utf-8")
            return
        if route == "favicon.ico":
            self._send(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
            return
        self._error("yol yok", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - http.server API'si
        session = self.ab_server.session
        route = self.path.split("?", 1)[0].lstrip("/")
        if route != "vote":
            self._error("yol yok", HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._error("boş gövde", HTTPStatus.BAD_REQUEST)
            return
        if length > MAX_BODY:
            self._error("gövde çok büyük", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return

        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._error("geçersiz JSON", HTTPStatus.BAD_REQUEST)
            return
        if not isinstance(payload, dict):
            self._error("gövde nesne olmalı", HTTPStatus.BAD_REQUEST)
            return

        try:
            vote = Vote.from_payload(payload, known_trials=[trial.id for trial in session.trials])
        except AbTestError as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST)
            return

        with self.ab_server.lock:
            self.ab_server.votes += 1
            with session.votes_path.open("a", encoding="utf-8") as handle:
                handle.write(vote.to_json() + "\n")
            total = self.ab_server.votes

        self._json({"ok": True, "votes": total})


def lan_addresses() -> list[str]:
    """Bu bilgisayarın yerel ağ (LAN) adresleri; dinleyicilere verilecek URL'ler için."""
    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            # Paket gönderilmez: yalnızca işletim sisteminin seçtiği yerel adres öğrenilir.
            probe.connect(("8.8.8.8", 80))
            found.append(probe.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if address not in found:
                found.append(address)
    except OSError:
        pass
    return [address for address in found if not address.startswith("127.")]


def listener_urls(host: str, port: int) -> list[str]:
    """Dinleyicilerin açacağı adresler: yerel ağa açıksa makinenin IP'leri."""
    if host in {LOOPBACK, "localhost"}:
        return [f"http://{LOOPBACK}:{port}/"]
    if host == LAN:
        return [f"http://{address}:{port}/" for address in lan_addresses()] or [
            f"http://{host}:{port}/"
        ]
    return [f"http://{host}:{port}/"]


def make_server(
    session: Session, *, host: str = LOOPBACK, port: int = 8800
) -> AbTestServer:
    """Dinlemeye hazır (ama başlatılmamış) sunucuyu kurar. ``port=0`` boş port seçer."""
    return AbTestServer((host, port), session)


def serve(
    directory: str | Path,
    *,
    host: str = LOOPBACK,
    port: int = 8800,
    open_browser: bool = True,
) -> None:
    """Sunucuyu çalıştırır; dinleyiciler sayfayı tarayıcıda açar."""
    session = load_session(directory)
    httpd = make_server(session, host=host, port=port)
    urls = listener_urls(host, port)
    print(f"Kör test hazır: {urls[0]}")
    if len(urls) > 1:
        print("  diğer adresler : " + " · ".join(urls[1:]))
    print(f"  oturum : {session.session_id} · {len(session.trials)} deneme")
    print(f"  oylar  : {session.votes_path}")
    print(f"  sonuçlar: {urls[0]}{REPORT_ROUTE} (yalnız bu bilgisayarda; kendini yeniler)")
    if host not in {LOOPBACK, "localhost"}:
        print("  ağ     : yerel ağa açık — aynı ağdaki herkes oy verebilir (kimlik doğrulama yok)")
        print("           aynı bilgisayarda ikinci kişi için adresin sonuna ?yeni=1 ekleyin")
        print("           Windows ilk bağlantıda güvenlik duvarı izni isteyebilir")
    print("  bitirmek için Ctrl+C")
    if open_browser:
        webbrowser.open(urls[0])
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nSunucu kapatıldı.")
    finally:
        httpd.server_close()
