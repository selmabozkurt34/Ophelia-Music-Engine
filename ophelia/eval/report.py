"""Sonuçların okunabilir hâli: tek dosya HTML rapor (tarayıcıda açılır).

Neden: ``ophelia abtest score`` terminale yazıyor; dinleme turu bittiğinde
sonucu görmek için kimsenin komut çalıştırması gerekmesin. Rapor ya dosya
olarak yazılır (``ophelia abtest rapor``) ya da sunucudan canlı olarak
verilir: test sürerken bile oylar biriktikçe kendini yeniler.

Dinleyiciler bu sayfayı **görmemeli**: canlı sonuç, sonraki oyları yanıltır.
Sunucu bu yüzden raporu yalnızca sunucunun çalıştığı bilgisayardan açar.
"""

from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from pathlib import Path

from ophelia.eval.abtest import (
    Report,
    Session,
    load_votes,
    required_votes,
    summarize,
)

REPORT_FILE = "rapor.html"
REPORT_ROUTE = "rapor"
REPORT_REFRESH_SECONDS = 15
TARGET_MARGIN = 0.10

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; padding: 32px 20px 64px; font: 16px/1.55 "Segoe UI", system-ui, sans-serif;
       background: #fbfaf8; color: #23201c; }
main { max-width: 880px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 1.05rem; margin: 36px 0 12px; text-transform: uppercase;
     letter-spacing: 0.08em; color: #7a7167; font-weight: 600; }
.sub { color: #7a7167; font-size: 0.92rem; }
.hero { margin: 24px 0 8px; padding: 22px 24px; border-radius: 14px;
        background: #fff; border: 1px solid #e8e2d9; }
.hero .big { font-size: 2.6rem; font-weight: 650; letter-spacing: -0.02em; line-height: 1.05; }
.hero .ci { color: #7a7167; font-size: 0.98rem; margin-top: 6px; }
.hero .verdict { margin-top: 14px; font-size: 1.02rem; font-weight: 600; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px; margin-top: 16px; }
.card { background: #fff; border: 1px solid #e8e2d9; border-radius: 12px; padding: 14px 16px; }
.card .k { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: #8d8478; }
.card .v { font-size: 1.45rem; font-weight: 600; margin-top: 2px; }
table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 12px;
        overflow: hidden; border: 1px solid #e8e2d9; font-variant-numeric: tabular-nums; }
th, td { padding: 9px 12px; text-align: right; border-bottom: 1px solid #f0ebe3; }
th { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em;
     color: #8d8478; font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
tbody tr:last-child td { border-bottom: none; }
.bar { height: 10px; border-radius: 999px; background: #ece6dc; overflow: hidden; margin-top: 10px; }
.bar > span { display: block; height: 100%; background: #6f8f5f; }
.note { margin-top: 10px; font-size: 0.9rem; color: #7a7167; }
.warn { margin-top: 24px; padding: 14px 16px; border-radius: 12px; background: #fff6e5;
        border: 1px solid #f0dcb4; font-size: 0.92rem; }
.empty { margin: 28px 0; padding: 22px; border-radius: 14px; background: #fff;
         border: 1px dashed #d9d2c6; color: #7a7167; }
footer { margin-top: 40px; font-size: 0.82rem; color: #9b938a; }
@media (prefers-color-scheme: dark) {
  body { background: #17150f; color: #ece7de; }
  .hero, .card, table { background: #1f1c16; border-color: #322d24; }
  th, td { border-bottom-color: #2a251d; }
  .hero .ci, .sub, .note, .card .k, h2 { color: #a49a8c; }
  .bar { background: #2a251d; }
  .warn { background: #2a2113; border-color: #4a3b1e; }
  .empty { background: #1f1c16; border-color: #322d24; color: #a49a8c; }
}
@media print {
  body { background: #fff; color: #000; padding: 0; }
  .hero, .card, table { border-color: #ccc; }
}
"""


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _ci(interval: tuple[float, float] | None) -> str:
    if interval is None:
        return "—"
    low, high = interval
    return f"%95 güven aralığı {low * 100:.1f}–{high * 100:.1f}%"


def _p(value: float | None) -> str:
    if value is None:
        return "—"
    return "< 0.0001" if value < 0.0001 else f"{value:.4f}"


def report_dict(report: Report) -> dict[str, object]:
    """Raporun JSON hâli (dosya, betik, arşiv için)."""
    session = report.session
    return {
        "session": session.session_id,
        "seconds": session.seconds,
        "trials": len(session.trials),
        "votes": report.votes,
        "decisive": report.decisive,
        "listeners": len(report.listeners),
        "real_choice": report.real_choice,
        "generated_choice": report.generated_choice,
        "ties": report.ties,
        "rate_real": report.rate_real,
        "ci95": report.ci,
        "p_value": report.p_value,
        "identification": {
            "n": report.identification_n,
            "correct": report.identification_correct,
            "rate": report.identification_rate,
            "ci95": report.identification_ci,
            "p_value": report.identification_p,
        },
        "mean_confidence": report.mean_confidence,
        "mean_seconds": report.mean_seconds,
        "duplicate_votes": report.duplicate_votes,
        "broken_lines": report.broken_lines,
        "verdict": report.verdict,
        "identification_verdict": report.identification_verdict(),
        "per_trial": [vars(item) for item in report.per_trial],
        "per_listener": [vars(item) for item in report.listeners],
    }


def _trial_rows(report: Report) -> str:
    rows = []
    for item in report.per_trial:
        rows.append(
            "<tr>"
            f"<td>{escape(item.trial_id)}</td>"
            f"<td>{item.votes}</td>"
            f"<td>{item.real_choice}</td>"
            f"<td>{item.generated_choice}</td>"
            f"<td>{item.ties}</td>"
            f"<td>{item.correct_guesses}/{item.guesses}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _listener_rows(report: Report) -> str:
    if not report.listeners:
        return '<tr><td colspan="4">—</td></tr>'
    rows = []
    for item in report.listeners:
        rows.append(
            "<tr>"
            f"<td>{escape(item.listener)}</td>"
            f"<td>{item.votes}</td>"
            f"<td>{_pct(item.rate)}</td>"
            f"<td>{item.correct_guesses}/{item.votes}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_report_html(
    report: Report,
    *,
    refresh_seconds: int | None = None,
    updated: datetime | None = None,
) -> str:
    """Raporu tek dosyalık HTML'e çevirir (gömülü CSS, harici istek yok)."""
    session = report.session
    stamp = (updated or datetime.now(UTC)).astimezone().strftime("%d.%m.%Y %H:%M")
    refresh = (
        f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">'
        if refresh_seconds
        else ""
    )
    needed = required_votes(TARGET_MARGIN)
    progress = min(1.0, report.decisive / needed) if needed else 0.0

    if report.decisive:
        hero = (
            '<div class="hero">'
            f'<div class="big">Gerçek koral: {_pct(report.rate_real)}</div>'
            f'<div class="ci">{_ci(report.ci)} · p = {_p(report.p_value)}'
            f" · {report.decisive} kararlı oy</div>"
            f'<div class="verdict">{escape(report.verdict)}</div>'
            "</div>"
        )
    else:
        hero = (
            '<div class="empty">'
            "<strong>Henüz kararlı oy yok.</strong> Oylar geldikçe bu sayfa kendini yeniler "
            f"(her {int(refresh_seconds)} saniyede bir)."
            "</div>"
            if refresh_seconds
            else '<div class="empty"><strong>Henüz kararlı oy yok.</strong></div>'
        )

    identification = (
        f"{report.identification_correct}/{report.identification_n} = "
        f"{_pct(report.identification_rate)}"
        if report.identification_n
        else "—"
    )
    confidence = (
        f"{report.mean_confidence:.2f} / 5" if report.mean_confidence is not None else "—"
    )
    seconds = f"{report.mean_seconds:.1f} sn" if report.mean_seconds is not None else "—"

    return f"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh}<title>Ophelia kör A/B sonuçları · {escape(session.session_id)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
  <h1>{escape(session.title)}</h1>
  <div class="sub">oturum {escape(session.session_id)} · {len(session.trials)} deneme ·
    {session.seconds:.0f} sn'lik kayıtlar · güncellendi {stamp}</div>

  {hero}

  <div class="grid">
    <div class="card"><div class="k">Oy</div><div class="v">{report.votes}</div></div>
    <div class="card"><div class="k">Dinleyici</div><div class="v">{len(report.listeners)}</div></div>
    <div class="card"><div class="k">Gerçek / üretim</div>
      <div class="v">{report.real_choice} / {report.generated_choice}</div></div>
    <div class="card"><div class="k">Eşit</div><div class="v">{report.ties}</div></div>
    <div class="card"><div class="k">Tanıma</div><div class="v">{identification}</div></div>
    <div class="card"><div class="k">Ortalama güven</div><div class="v">{confidence}</div></div>
    <div class="card"><div class="k">Karar süresi</div><div class="v">{seconds}</div></div>
    <div class="card"><div class="k">Yinelenen oy</div><div class="v">{report.duplicate_votes}</div></div>
  </div>

  <h2>Örneklem yeterli mi</h2>
  <div class="bar"><span style="width: {progress * 100:.1f}%"></span></div>
  <div class="note">{report.decisive} / ~{needed} kararlı oy · ±10 puan kesinlik için gereken
    (kaba) sayı. Az oyla çıkan fark istatistiksel olarak anlamlı görünse bile tekrar
    edilemeyebilir.</div>

  <h2>Deneme bazında</h2>
  <table>
    <thead><tr><th>Deneme</th><th>Oy</th><th>Gerçek</th><th>Üretim</th><th>Eşit</th>
      <th>Doğru tanıma</th></tr></thead>
    <tbody>
{_trial_rows(report)}
    </tbody>
  </table>

  <h2>Dinleyici bazında</h2>
  <table>
    <thead><tr><th>Dinleyici</th><th>Oy</th><th>Gerçek tercihi</th><th>Doğru tanıma</th></tr></thead>
    <tbody>
{_listener_rows(report)}
    </tbody>
  </table>

  <div class="warn">
    Bu sayfa <strong>dinleyicilere gösterilmemeli</strong>: canlı sonucu gören kişinin
    sonraki oyları yanıltılır. Ayrıca farklı ses sürümleriyle kurulmuş oturumların oyları
    birleştirilmemeli — her turun kendi raporu geçerli.
  </div>

  <footer>ophelia abtest rapor · etiket anahtarı (key.json) bu sayfada yok,
    kaynaklar yalnız toplu sayı olarak gösterilir.</footer>
</main>
</body>
</html>
"""


def build_report(directory: str | Path) -> Report:
    """Oturum klasöründen oyları okuyup raporu üretir (oy yoksa da çalışır)."""
    from ophelia.eval.abtest import load_session

    session = load_session(directory)
    if session.votes_path.is_file():
        votes, broken = load_votes([session.votes_path])
    else:
        votes, broken = [], 0
    return summarize(session, votes, broken_lines=broken)


def live_report_html(session: Session, *, refresh_seconds: int = REPORT_REFRESH_SECONDS) -> str:
    """Sunucunun her istekte taze oylardan ürettiği rapor (kendini yeniler)."""
    if session.votes_path.is_file():
        votes, broken = load_votes([session.votes_path])
    else:
        votes, broken = [], 0
    report = summarize(session, votes, broken_lines=broken)
    return render_report_html(report, refresh_seconds=refresh_seconds)


def write_report(directory: str | Path, path: str | Path | None = None) -> Path:
    """Raporu dosyaya yazar; varsayılan ``<oturum>/rapor.html``."""
    folder = Path(directory)
    target = Path(path) if path is not None else folder / REPORT_FILE
    report = build_report(folder)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report_html(report), encoding="utf-8")
    return target


def report_visible_to(client_ip: str, *, local_ips: list[str] | None = None) -> bool:
    """Raporu yalnızca sunucunun kendi bilgisayarından açtırır."""
    if client_ip in {"127.0.0.1", "::1", "localhost"}:
        return True
    if local_ips is None:
        from ophelia.eval.server import lan_addresses

        local_ips = lan_addresses()
    return client_ip in local_ips
