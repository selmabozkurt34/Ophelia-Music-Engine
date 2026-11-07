"""9. aşama: yerel web arayüzü — HTTP katmanı (gerçek model yüklenmez)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from ophelia.ui import ComposeRequest, UiError, make_ui_server
from ophelia.ui import server as server_module


def _payload(**overrides) -> dict:
    base = {
        "command": "hüzünlü la minör, yavaş, 20 saniye",
        "key": "A minor",
        "detected_key": "A minor",
        "tempo_bpm": 68.0,
        "duration_seconds": 20.0,
        "n_notes": 42,
        "seed": 7,
        "explanations": ["tonalite: A minor"],
        "instruments": [{"name": "Keman", "role": "melody", "notes": 20}],
        "midi": "/midi/eser.mid",
        "wav": "/audio/eser.wav",
        "xml": None,
        "lufs": -14.0,
        "true_peak": -3.2,
    }
    base.update(overrides)
    return base


@pytest.fixture
def fake_outputs(tmp_path: Path, monkeypatch):
    """Çıktı klasörlerini geçici dizine yönlendirir."""
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    (wav_dir / "eser.wav").write_bytes(b"RIFF" + b"\x00" * 64)
    monkeypatch.setitem(server_module.FILE_ROUTES, "audio", ((wav_dir,), "audio/wav"))
    monkeypatch.setitem(server_module.FILE_ROUTES, "midi", ((wav_dir,), "audio/midi"))
    monkeypatch.setitem(server_module.FILE_ROUTES, "score", ((wav_dir,), "application/xml"))
    return tmp_path


@pytest.fixture
def running_ui(fake_outputs):
    seen: list[ComposeRequest] = []

    def composer(request: ComposeRequest) -> dict:
        seen.append(request)
        return _payload(seed=request.seed, command=request.command)

    httpd = make_ui_server(composer=composer, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd, seen
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - yerel test
        return response.status, response.read()


def _post(url: str, payload: dict | bytes) -> tuple[int, dict]:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - yerel test
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_page_is_served(running_ui) -> None:
    httpd, _ = running_ui
    status, body = _get(httpd.url)
    assert status == 200
    page = body.decode("utf-8")
    assert "Ophelia" in page
    assert 'id="generate"' in page
    assert "Üret" in page


def test_status_reports_model_availability(running_ui) -> None:
    httpd, _ = running_ui
    status, body = _get(httpd.url + "api/status")
    assert status == 200
    data = json.loads(body)
    assert set(data) == {"ready", "checkpoint"}


def test_compose_returns_playable_payload(running_ui) -> None:
    httpd, seen = running_ui
    status, data = _post(
        httpd.url + "compose",
        {"command": "neşeli do majör, 25 saniye", "seed": 12, "temperature": 0.9, "xml": True},
    )
    assert status == 200
    assert data["wav"] == "/audio/eser.wav"
    assert data["xml"] is None  # composer ne döndürdüyse o
    assert data["lufs"] == -14.0

    request = seen[0]
    assert request.command == "neşeli do majör, 25 saniye"
    assert request.seed == 12
    assert request.temperature == pytest.approx(0.9)
    assert request.xml is True


def test_compose_accepts_empty_seed(running_ui) -> None:
    httpd, seen = running_ui
    status, _ = _post(httpd.url + "compose", {"command": "la minör 10 saniye", "seed": None})
    assert status == 200
    assert seen[0].seed is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"command": "   "}, "komut boş"),
        ({"command": "la minör", "seed": "abc"}, "tohum"),
        ({"command": "la minör", "seed": -3}, "negatif"),
        ({"command": "la minör", "temperature": 5}, "sıcaklık"),
        ({"command": "la minör", "temperature": "sıcak"}, "sıcaklık"),
        ({"command": "x" * 500}, "çok uzun"),
    ],
)
def test_compose_rejects_bad_payloads(running_ui, payload, message) -> None:
    httpd, _ = running_ui
    status, data = _post(httpd.url + "compose", payload)
    assert status == 400
    assert message in data["error"]


def test_compose_rejects_oversized_body(running_ui) -> None:
    httpd, _ = running_ui
    status, data = _post(httpd.url + "compose", b"{" + b"a" * server_module.MAX_BODY + b"}")
    assert status == 400
    assert "boyut" in data["error"]


def test_compose_reports_engine_errors_as_400(fake_outputs) -> None:
    def composer(request: ComposeRequest) -> dict:
        raise UiError("model ağırlığı yok")

    httpd = make_ui_server(composer=composer, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, data = _post(httpd.url + "compose", {"command": "la minör"})
        assert status == 400
        assert "model ağırlığı" in data["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_unexpected_errors_are_500_and_keep_server_alive(fake_outputs) -> None:
    def composer(request: ComposeRequest) -> dict:
        raise RuntimeError("patladı")

    httpd = make_ui_server(composer=composer, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, data = _post(httpd.url + "compose", {"command": "la minör"})
        assert status == 500
        assert "patladı" in data["error"]
        assert _get(httpd.url)[0] == 200  # sunucu ayakta
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_generated_files_are_served_but_nothing_else(running_ui) -> None:
    httpd, _ = running_ui
    status, body = _get(httpd.url + "audio/eser.wav")
    assert status == 200
    assert body[:4] == b"RIFF"

    for route in ("audio/yok.wav", "audio/../index.html", "midi/../../configs/default.yaml"):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _get(httpd.url + route)
        assert excinfo.value.code == 404


def test_unknown_routes_are_404(running_ui) -> None:
    httpd, _ = running_ui
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(httpd.url + "bilinmeyen")
    assert excinfo.value.code == 404
