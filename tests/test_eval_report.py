"""Sonuç raporu: HTML çıktısı, canlı görünüm ve sunucu rotası."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from ophelia.data.midi_io import write_midi
from ophelia.eval import (
    Vote,
    build_session,
    make_server,
    report_dict,
    report_visible_to,
    summarize,
    write_report,
)
from ophelia.eval.report import live_report_html, render_report_html

from conftest import make_notes


def _midi(path: Path, *, base: int = 60) -> Path:
    notes = make_notes(tuple(base + step for step in (0, 2, 4, 5, 7)), step=240, duration=240)
    write_midi(notes, path, tempo_bpm=90.0)
    return path


@pytest.fixture
def session(tmp_path: Path):
    real = [_midi(tmp_path / f"gercek{i}.mid", base=55 + i) for i in range(4)]
    generated = [_midi(tmp_path / f"uretim{i}.mid", base=67 + i) for i in range(4)]
    return build_session(real, generated, tmp_path / "oturum", trials=2, seconds=1.0, seed=7)


def _vote(trial_id: str, *, preference: str, listener: str = "L-1", guess=None) -> Vote:
    return Vote.from_payload(
        {"trial": trial_id, "preference": preference, "listener": listener, "human_guess": guess}
    )


def test_report_html_shows_empty_state_without_votes(session) -> None:
    html = render_report_html(summarize(session, []))
    assert "Henüz kararlı oy yok" in html
    assert "dinleyicilere gösterilmemeli" in html
    assert session.session_id in html


def test_report_html_shows_rates_and_tables(session) -> None:
    first, second = session.trials
    votes = [
        _vote(first.id, preference=first.real_side, guess=first.real_side),
        _vote(second.id, preference="tie", listener="L-2"),
    ]
    report = summarize(session, votes)
    html = render_report_html(report)

    assert "Gerçek koral: 100.0%" in html
    assert "%95 güven aralığı" in html
    assert "Deneme bazında" in html
    assert "Dinleyici bazında" in html
    assert "L-2" in html


def test_live_report_refreshes_itself_and_reads_votes(session) -> None:
    session.votes_path.write_text(
        json.dumps(
            {
                "trial": session.trials[0].id,
                "preference": session.trials[0].real_side,
                "listener": "L-1",
                "human_guess": None,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    html = live_report_html(session, refresh_seconds=20)

    assert 'http-equiv="refresh" content="20"' in html
    assert "Gerçek koral: 100.0%" in html


def test_live_report_without_vote_file_does_not_fail(session) -> None:
    assert not session.votes_path.exists()
    assert "Henüz kararlı oy yok" in live_report_html(session)


def test_write_report_creates_a_file(session, tmp_path: Path) -> None:
    path = write_report(session.directory)

    assert path == session.directory / "rapor.html"
    assert "Ophelia" in path.read_text(encoding="utf-8")


def test_report_dict_is_json_serialisable(session) -> None:
    payload = json.loads(json.dumps(report_dict(summarize(session, []))))
    assert payload["trials"] == 2
    assert payload["votes"] == 0
    assert payload["verdict"] == "yeterli karar yok"


def test_report_is_only_visible_to_the_host_machine() -> None:
    assert report_visible_to("127.0.0.1")
    assert not report_visible_to("10.9.9.9", local_ips=["192.168.1.42"])
    assert report_visible_to("192.168.1.42", local_ips=["192.168.1.42"])


# --- sunucu rotası ---------------------------------------------------------


@pytest.fixture
def running_server(session):
    httpd = make_server(session, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield session, httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - yerel test
        return response.status, response.read()


def test_server_serves_the_live_report_to_the_host(running_server) -> None:
    session, httpd = running_server
    session.votes_path.write_text(
        json.dumps(
            {
                "trial": session.trials[0].id,
                "preference": session.trials[0].real_side,
                "listener": "L-1",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    status, body = _get(httpd.url + "rapor")
    assert status == 200
    assert "Gerçek koral: 100.0%" in body.decode("utf-8")

    status, body = _get(httpd.url + "rapor.json")
    payload = json.loads(body)
    assert payload["decisive"] == 1
    assert payload["rate_real"] == pytest.approx(1.0)


def test_server_refuses_report_for_foreign_ip(running_server, monkeypatch) -> None:
    _, httpd = running_server
    monkeypatch.setattr("ophelia.eval.server.report_visible_to", lambda ip: False)

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(httpd.url + "rapor")
    assert excinfo.value.code == 403
