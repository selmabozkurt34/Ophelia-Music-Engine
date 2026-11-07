"""Kör A/B testi: kurulum, körlük, doğrulama ve istatistik."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from ophelia.data.midi_io import write_midi
from ophelia.eval import (
    AbTestError,
    Vote,
    binomial_p_value,
    build_session,
    default_real_pool,
    load_session,
    load_votes,
    make_server,
    required_votes,
    summarize,
    wilson_interval,
)

from conftest import make_notes


def _midi(path: Path, *, base: int = 60, tempo: float = 90.0) -> Path:
    notes = make_notes(tuple(base + step for step in (0, 2, 4, 5, 7)), step=240, duration=240)
    write_midi(notes, path, tempo_bpm=tempo)
    return path


@pytest.fixture
def pools(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    real = [_midi(tmp_path / f"gercek{i}.mid", base=55 + i) for i in range(4)]
    generated = [_midi(tmp_path / f"uretim{i}.mid", base=67 + i) for i in range(4)]
    return real, generated


@pytest.fixture
def session(tmp_path: Path, pools: tuple[list[Path], list[Path]]):
    real, generated = pools
    return build_session(real, generated, tmp_path / "oturum", trials=2, seconds=1.0, seed=7)


# --- istatistik ------------------------------------------------------------


def test_wilson_interval_matches_known_values() -> None:
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.404, abs=0.001)
    assert high == pytest.approx(0.596, abs=0.001)

    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


def test_binomial_p_value_hits_extremes() -> None:
    assert binomial_p_value(5, 10) == pytest.approx(1.0, abs=0.01)
    assert binomial_p_value(10, 10) < 0.01
    assert binomial_p_value(0, 0) == 1.0


def test_required_votes_for_ten_points() -> None:
    assert required_votes(0.1) == 97  # 1,96² · 0,25 / 0,1² = 96,04 → 97
    assert required_votes(0.05) > required_votes(0.1)
    with pytest.raises(AbTestError):
        required_votes(1.5)


# --- kurulum ve körlük -----------------------------------------------------


def test_session_renders_opaque_audio_and_keeps_key_server_side(session) -> None:
    assert len(session.trials) == 2
    wavs = sorted(session.audio_dir.glob("*.wav"))
    assert len(wavs) == 4
    assert all(wav.stat().st_size > 1000 for wav in wavs)

    # Dosya adlarında kaynak sızdırmayan tek iz: opak kimlikler
    assert {wav.name for wav in wavs} == {"t01_a.wav", "t01_b.wav", "t02_a.wav", "t02_b.wav"}

    public = (session.directory / "trials.json").read_text(encoding="utf-8")
    assert "real_side" not in public
    assert "gercek" not in public
    assert "uretim" not in public

    key = json.loads((session.directory / "key.json").read_text(encoding="utf-8"))
    assert {item["real_side"] for item in key["trials"]} <= {"a", "b"}
    assert key["trials"][0]["real_source"].startswith("gercek")

    # Sayfa deneme kimliklerini sunucudan alır, etiketleri taşımaz.
    page = (session.directory / "index.html").read_text(encoding="utf-8")
    assert "key.json" not in page


def test_blinding_is_deterministic_for_a_seed(tmp_path: Path, pools) -> None:
    real, generated = pools
    first = build_session(real, generated, tmp_path / "a", trials=2, seconds=0.5, seed=3)
    second = build_session(real, generated, tmp_path / "b", trials=2, seconds=0.5, seed=3)
    assert [trial.real_side for trial in first.trials] == [trial.real_side for trial in second.trials]
    assert [trial.real_source for trial in first.trials] == [
        trial.real_source for trial in second.trials
    ]


def test_same_pitch_sources_do_not_repeat(tmp_path: Path, pools) -> None:
    real, generated = pools
    session = build_session(real, generated, tmp_path / "o", trials=4, seconds=0.5, seed=1)
    assert len({trial.real_source for trial in session.trials}) == 4
    assert len({trial.generated_source for trial in session.trials}) == 4


def test_small_pool_is_rejected_with_reason(tmp_path: Path, pools) -> None:
    real, generated = pools
    with pytest.raises(AbTestError, match="gerçek havuz yetersiz"):
        build_session(real[:1], generated, tmp_path / "o", trials=2, seconds=0.5)
    with pytest.raises(AbTestError, match="üretim havuzu yetersiz"):
        build_session(real, generated[:1], tmp_path / "o", trials=2, seconds=0.5)


def test_default_real_pool_uses_test_split(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed" / "music21-bach-chorales"
    processed.mkdir(parents=True)
    for stem in ("test-a", "test-b", "train-x"):
        _midi(processed / f"{stem}.mid")
    splits = tmp_path / "data" / "splits" / "splits.json"
    splits.parent.mkdir(parents=True)
    splits.write_text(
        json.dumps({"splits": {"test": ["test-a", "test-b"], "train": ["train-x"]}}),
        encoding="utf-8",
    )

    pool = default_real_pool(tmp_path)
    assert [path.stem for path in pool] == ["test-a", "test-b"]


def test_load_session_round_trip(session) -> None:
    loaded = load_session(session.directory)
    assert loaded.session_id == session.session_id
    assert [trial.id for trial in loaded.trials] == ["t01", "t02"]
    assert loaded.seconds == pytest.approx(1.0)
    with pytest.raises(AbTestError, match="oturum bulunamadı"):
        load_session(session.directory / "yok")


# --- oy doğrulama ----------------------------------------------------------


def test_vote_payload_validation() -> None:
    ok = Vote.from_payload(
        {"trial": "t01", "preference": "a", "listener": "L-1", "human_guess": "B", "confidence": 4},
        known_trials=["t01"],
    )
    assert (ok.preference, ok.human_guess, ok.confidence) == ("a", "b", 4)

    with pytest.raises(AbTestError, match="tercih"):
        Vote.from_payload({"trial": "t01", "preference": "x", "listener": "L"}, known_trials=["t01"])
    with pytest.raises(AbTestError, match="bilinmeyen deneme"):
        Vote.from_payload({"trial": "t99", "preference": "a", "listener": "L"}, known_trials=["t01"])
    with pytest.raises(AbTestError, match="kimliği boş"):
        Vote.from_payload({"trial": "t01", "preference": "a"})
    with pytest.raises(AbTestError, match="güven"):
        Vote.from_payload(
            {"trial": "t01", "preference": "a", "listener": "L", "confidence": 9}
        )

    unsure = Vote.from_payload(
        {"trial": "t01", "preference": "tie", "listener": "L", "human_guess": "none"}
    )
    assert unsure.human_guess is None


def test_load_votes_skips_broken_lines(tmp_path: Path) -> None:
    path = tmp_path / "votes.jsonl"
    path.write_text(
        '{"trial": "t01", "preference": "a", "listener": "L"}\n'
        "bozuk satır\n"
        '{"trial": "t02", "preference": "tie", "listener": "L"}\n',
        encoding="utf-8",
    )
    votes, broken = load_votes([path])
    assert len(votes) == 2
    assert broken == 1


# --- rapor -----------------------------------------------------------------


def _vote(session, trial_id: str, *, preference: str, listener: str = "L-1", guess=None):
    return Vote.from_payload(
        {
            "trial": trial_id,
            "preference": preference,
            "listener": listener,
            "human_guess": guess,
        }
    )


def test_report_maps_sides_through_key(session) -> None:
    first, second = session.trials

    def opposite(side: str) -> str:
        return "b" if side == "a" else "a"

    votes = [
        # doğru tercih + doğru tanıma
        _vote(session, first.id, preference=first.real_side, guess=first.real_side),
        # yanlış tercih, ama insanı doğru tanıma
        _vote(session, second.id, preference=opposite(second.real_side), guess=second.real_side),
        # eşit: orana girmez; tanıma yanlış
        _vote(
            session,
            second.id,
            preference="tie",
            listener="L-2",
            guess=opposite(second.real_side),
        ),
    ]
    report = summarize(session, votes)

    assert report.votes == 3
    assert (report.real_choice, report.generated_choice, report.ties) == (1, 1, 1)
    assert report.decisive == 2
    assert report.rate_real == pytest.approx(0.5)
    low, high = report.ci
    assert low <= 0.5 <= high
    assert "ayırt edilemedi" in report.verdict

    assert report.identification_n == 3
    assert report.identification_correct == 2
    assert {item.listener for item in report.listeners} == {"L-1", "L-2"}


def test_report_flags_clear_preference(session) -> None:
    first, second = session.trials
    votes = []
    for index in range(30):
        for trial in (first, second):
            side = trial.real_side
            votes.append(_vote(session, trial.id, preference=side, listener=f"L-{index:02d}"))
    report = summarize(session, votes)
    assert report.rate_real == pytest.approx(1.0)
    assert report.ci[0] > 0.5
    assert "gerçek koral anlamlı biçimde tercih edildi" in report.verdict
    assert report.p_value < 0.001


def test_report_ignores_unknown_trials(session) -> None:
    vote = Vote(trial_id="t99", preference="a", listener="L")
    report = summarize(session, [vote])
    assert report.votes == 0
    assert report.verdict == "yeterli karar yok"


def test_report_counts_one_vote_per_listener_and_trial(session) -> None:
    """Aynı kişi aynı denemeyi tekrar oylarsa örneklem şişmemeli."""
    first, second = session.trials
    real_side = first.real_side
    votes = [
        _vote(session, first.id, preference=real_side, listener="L-1"),
        _vote(session, first.id, preference=real_side, listener="L-1"),  # tekrar: sayılmaz
        _vote(session, second.id, preference="tie", listener="L-1"),
        _vote(session, second.id, preference="a", listener="L-2"),
    ]
    report = summarize(session, votes)

    assert report.votes == 3
    assert report.duplicate_votes == 1
    assert len(report.listeners) == 2
    assert {item.listener: item.votes for item in report.listeners} == {"L-1": 2, "L-2": 1}


def test_page_supports_a_second_listener(session) -> None:
    """Aynı tarayıcıda ikinci dinleyici için kimlik yenileme yolu olmalı."""
    page = (session.directory / "index.html").read_text(encoding="utf-8")
    assert "?yeni=1" in page
    assert 'id="newListener"' in page
    assert "ophelia.listener" in page


# --- sunucu ----------------------------------------------------------------


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


def test_listener_urls_switch_between_loopback_and_lan(monkeypatch) -> None:
    from ophelia.eval import server as server_module

    assert server_module.listener_urls("127.0.0.1", 8800) == ["http://127.0.0.1:8800/"]
    monkeypatch.setattr(server_module, "lan_addresses", lambda: ["192.168.1.42"])
    assert server_module.listener_urls("0.0.0.0", 8800) == ["http://192.168.1.42:8800/"]


def test_server_can_open_to_the_local_network(session) -> None:
    httpd = make_server(session, host="0.0.0.0", port=0)
    try:
        assert httpd.server_address[0] == "0.0.0.0"
        assert httpd.server_address[1] > 0
    finally:
        httpd.server_close()


def test_server_serves_page_trials_and_audio(running_server) -> None:
    session, httpd = running_server
    base = httpd.url

    status, body = _get(base)
    assert status == 200
    assert "<html" in body.decode("utf-8")

    status, body = _get(base + "trials.json")
    trials = json.loads(body)
    assert len(trials) == 2
    assert set(trials[0]) == {"id", "a", "b"}

    status, body = _get(base + trials[0]["a"])
    assert status == 200
    assert body[:4] == b"RIFF"


def test_server_hides_key_and_unknown_paths(running_server) -> None:
    _, httpd = running_server
    for route in ("key.json", "audio/../key.json", "votes.jsonl", "bilinmeyen"):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _get(httpd.url + route)
        assert excinfo.value.code == 404


def test_server_records_vote(running_server) -> None:
    session, httpd = running_server
    trial = session.trials[0]
    payload = json.dumps(
        {"trial": trial.id, "preference": "a", "listener": "L-9", "human_guess": "b"}
    ).encode("utf-8")
    request = urllib.request.Request(
        httpd.url + "vote",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310 - yerel test
        assert response.status == 200
        assert json.loads(response.read())["ok"] is True

    lines = session.votes_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["listener"] == "L-9"


def test_server_rejects_bad_vote(running_server) -> None:
    session, httpd = running_server
    payload = json.dumps({"trial": "t99", "preference": "a", "listener": "L"}).encode("utf-8")
    request = urllib.request.Request(
        httpd.url + "vote", data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=5)  # noqa: S310 - yerel test
    assert excinfo.value.code == 400
    assert not session.votes_path.exists()
