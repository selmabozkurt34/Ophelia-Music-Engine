"""Yerel web arayüzü sunucusu (9. aşama).

Tek sayfa + üç uç:

============================  ===================================================
``GET /``                     arayüz (``index.html``)
``GET /api/status``           model hazır mı, hangi checkpoint
``POST /compose``             komut → eser; WAV/MIDI/MusicXML yollarını döner
``GET /audio|midi|score/…``   üretilen dosyalar (yalnız çıktı klasörlerinden)
============================  ===================================================

Sentezleyici (composer) enjekte edilebilir: testler gerçek modeli yüklemeden
HTTP katmanını sınar. Varsayılan composer modeli tembel yükler ve üretimi
kilitler — CPU'da aynı anda iki üretim makineyi boğardı.
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

from ophelia.core.paths import MIDI_OUT_DIR, MODELS_DIR, SCORE_OUT_DIR, WAV_OUT_DIR

MAX_BODY = 8192
LOOPBACK = "127.0.0.1"
LAN = "0.0.0.0"
INDEX_FILE = "index.html"
PAGE = Path(__file__).with_name(INDEX_FILE)

Composer = Callable[["ComposeRequest"], dict[str, object]]


class UiError(ValueError):
    """Kullanıcıya gösterilebilir hata (400)."""


@dataclass(frozen=True)
class ComposeRequest:
    """Arayüzden gelen üretim isteği."""

    command: str
    seed: int | None = None
    temperature: float = 0.95
    wav: bool = True
    xml: bool = False

    @classmethod
    def from_payload(cls, payload: object) -> ComposeRequest:
        if not isinstance(payload, dict):
            raise UiError("gövde bir JSON nesnesi olmalı")
        command = str(payload.get("command") or "").strip()
        if not command:
            raise UiError("komut boş olamaz")
        if len(command) > 400:
            raise UiError("komut çok uzun (en fazla 400 karakter)")

        raw_seed = payload.get("seed")
        seed: int | None = None
        if raw_seed not in (None, "", "auto"):
            try:
                seed = int(raw_seed)
            except (TypeError, ValueError) as exc:
                raise UiError("tohum bir tam sayı olmalı") from exc
            if seed < 0:
                raise UiError("tohum negatif olamaz")

        try:
            temperature = float(payload.get("temperature", 0.95))
        except (TypeError, ValueError) as exc:
            raise UiError("sıcaklık sayı olmalı") from exc
        if not 0.1 <= temperature <= 2.0:
            raise UiError("sıcaklık 0,1–2,0 arasında olmalı")

        return cls(
            command=command,
            seed=seed,
            temperature=temperature,
            wav=bool(payload.get("wav", True)),
            xml=bool(payload.get("xml", False)),
        )


# --- varsayılan composer ---------------------------------------------------

_MODELS: dict[str, tuple[object, object]] = {}
_MODEL_LOCK = threading.Lock()


def latest_checkpoint() -> Path | None:
    """En yeni eğitilmiş model ağırlığı."""
    candidates = sorted(MODELS_DIR.glob("*/best.pt"), key=lambda item: item.stat().st_mtime)
    return candidates[-1] if candidates else None


def _load_models(checkpoint: Path) -> tuple[object, object]:
    from ophelia.nlp import load_models

    key = str(checkpoint)
    with _MODEL_LOCK:
        cached = _MODELS.get(key)
        if cached is None:
            cached = load_models(checkpoint)
            _MODELS[key] = cached
        return cached


def _measure(wav_path: Path) -> dict[str, float]:
    """Yayın zincirinin sonucunu arayüzde göstermek için ölç."""
    from scipy.io import wavfile

    from ophelia.audio import loudness
    from ophelia.audio.mastering import true_peak

    rate, data = wavfile.read(wav_path)
    stereo = data.T.astype("float32") / 32768.0
    return {
        "lufs": round(loudness(stereo, rate), 1),
        "true_peak": round(true_peak(stereo, rate), 1),
    }


def _payload(result: object) -> dict[str, object]:
    """``ComposeResult``'ı JSON'a çevrilir hâle getirir."""
    spec = result.spec  # type: ignore[attr-defined]
    parts = result.parts  # type: ignore[attr-defined]
    wav_path = result.wav_path  # type: ignore[attr-defined]
    xml_path = result.xml_path  # type: ignore[attr-defined]
    midi_path = result.midi_path  # type: ignore[attr-defined]

    payload: dict[str, object] = {
        "command": spec.command,
        "key": spec.key.name if spec.key else None,
        "detected_key": result.final_key.name if result.final_key else None,  # type: ignore[attr-defined]
        "tempo_bpm": round(spec.tempo_bpm, 1),
        "duration_seconds": round(result.duration_seconds, 1),  # type: ignore[attr-defined]
        "n_notes": result.n_notes,  # type: ignore[attr-defined]
        "seed": spec.seed,
        "explanations": [str(note) for note in spec.notes],
        "instruments": [
            {
                "name": part.instrument.name,
                "role": part.part_role,
                "notes": len(part.notes),
            }
            for part in parts
        ],
        "midi": f"/midi/{Path(midi_path).name}" if midi_path else None,
        "wav": f"/audio/{Path(wav_path).name}" if wav_path else None,
        "xml": f"/score/{Path(xml_path).name}" if xml_path else None,
    }
    if wav_path and Path(wav_path).is_file():
        payload.update(_measure(Path(wav_path)))
    return payload


def default_composer(request: ComposeRequest) -> dict[str, object]:
    """Gerçek modelle üretir (modeli ilk istekte yükler)."""
    from ophelia.nlp import compose as compose_engine

    checkpoint = latest_checkpoint()
    if checkpoint is None:
        raise UiError("model ağırlığı yok — önce: ophelia model train")

    model, tokenizer = _load_models(checkpoint)
    digest = hashlib.sha1(
        f"{request.command}|{request.seed}|{request.temperature}".encode()
    ).hexdigest()[:8]
    label = request.seed if request.seed is not None else "auto"
    target = MIDI_OUT_DIR / f"ui_{label}_{digest}.mid"

    try:
        result = compose_engine(  # type: ignore[arg-type]
            request.command,
            model,
            tokenizer,
            out_path=target,
            wav_path=target.with_suffix(".wav") if request.wav else None,
            xml_path=target.with_suffix(".musicxml") if request.xml else None,
            seed=request.seed,
            temperature=request.temperature,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise UiError(str(exc)) from exc
    return _payload(result)


# --- sunucu ----------------------------------------------------------------


class UiServer(ThreadingHTTPServer):
    """Arayüz sunucusu: composer + üretim kilidi taşır."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], composer: Composer) -> None:
        self.composer = composer
        self.generation_lock = threading.Lock()
        super().__init__(address, UiHandler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/"


#: Klasör listesi: ``model train`` çıktıları ``outputs/wav``, ``compose`` çıktıları
#: MIDI'nin yanına yazılır; iki yer de geçerli.
FILE_ROUTES: dict[str, tuple[tuple[Path, ...], str]] = {
    "audio": ((WAV_OUT_DIR, MIDI_OUT_DIR), "audio/wav"),
    "midi": ((MIDI_OUT_DIR,), "audio/midi"),
    "score": ((SCORE_OUT_DIR, MIDI_OUT_DIR), "application/vnd.recordare.musicxml+xml"),
}


class UiHandler(BaseHTTPRequestHandler):
    """GET: sayfa/dosya/durum · POST: üretim."""

    server_version = "OpheliaUI/1.0"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - imza sabit
        """İstek günlüğü kapalı; sayfa kendi durumunu gösteriyor."""

    @property
    def ui_server(self) -> UiServer:
        return cast(UiServer, self.server)

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

    def do_GET(self) -> None:  # noqa: N802 - http.server API'si
        route = self.path.split("?", 1)[0].lstrip("/")

        if route in ("", INDEX_FILE):
            self._send_file(PAGE, "text/html; charset=utf-8")
            return
        if route == "api/status":
            checkpoint = latest_checkpoint()
            self._json(
                {
                    "ready": checkpoint is not None,
                    "checkpoint": str(checkpoint) if checkpoint else None,
                }
            )
            return
        if route == "favicon.ico":
            self._send(b"", "image/x-icon", HTTPStatus.NO_CONTENT)
            return

        kind, _, name = route.partition("/")
        if kind in FILE_ROUTES and name:
            folders, content_type = FILE_ROUTES[kind]
            for folder in folders:
                if folder.exists() and name in {item.name for item in folder.iterdir()}:
                    self._send_file(folder / name, content_type)
                    return
            self._error("bilinmeyen dosya", HTTPStatus.NOT_FOUND)
            return

        self._error("yol yok", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - http.server API'si
        route = self.path.split("?", 1)[0].lstrip("/")
        if route != "compose":
            self._error("yol yok", HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            self._error("gövde boyutu geçersiz", HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            request = ComposeRequest.from_payload(payload)
        except json.JSONDecodeError:
            self._error("gövde geçerli JSON değil", HTTPStatus.BAD_REQUEST)
            return
        except UiError as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST)
            return

        try:
            with self.ui_server.generation_lock:
                result = self.ui_server.composer(request)
        except UiError as exc:
            self._error(str(exc), HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:  # noqa: BLE001 - arayüz çökmemeli
            self._error(f"üretim başarısız: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._json(result)


def make_ui_server(
    *, composer: Composer | None = None, host: str = LOOPBACK, port: int = 8810
) -> UiServer:
    """Sunucuyu kurar (başlatmaz); ``port=0`` boş port seçer."""
    return UiServer((host, port), composer or default_composer)


def _urls(host: str, port: int) -> list[str]:
    if host not in {LOOPBACK, "localhost"}:
        found: list[str] = []
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("8.8.8.8", 80))
                found.append(probe.getsockname()[0])
        except OSError:
            pass
        if found:
            return [f"http://{address}:{port}/" for address in found]
    return [f"http://{LOOPBACK}:{port}/"]


def serve(*, host: str = LOOPBACK, port: int = 8810, open_browser: bool = True) -> None:
    """Arayüzü çalıştırır; Ctrl+C ile kapanır."""
    httpd = make_ui_server(host=host, port=port)
    urls = _urls(host, port)
    checkpoint = latest_checkpoint()
    print(f"Ophelia arayüzü hazır: {urls[0]}")
    if len(urls) > 1:
        print("  diğer adresler : " + " · ".join(urls[1:]))
    print(f"  model          : {checkpoint or 'YOK — önce: ophelia model train'}")
    if host not in {LOOPBACK, "localhost"}:
        print("  ağ             : yerel ağa açık (kimlik doğrulama yok)")
    print("  bitirmek için Ctrl+C")
    if open_browser:
        webbrowser.open(urls[0])
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nArayüz kapatıldı.")
    finally:
        httpd.server_close()
