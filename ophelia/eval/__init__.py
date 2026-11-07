"""Değerlendirme (kör A/B dinleme testi).

Modelin müzikal kalitesi ilk kez burada **dinleyici tercihiyle** ölçülür:
gerçek koral ile üretim aynı sentezden geçer, sıra/taraf rastgele gizlenir,
sonuç tercih oranı + Wilson güven aralığı olarak raporlanır.
"""

from __future__ import annotations

from ophelia.eval.abtest import (
    AbTestError,
    ListenerReport,
    Report,
    Session,
    Trial,
    TrialReport,
    Vote,
    binomial_p_value,
    build_session,
    default_generated_pool,
    default_real_pool,
    load_session,
    load_votes,
    render_clip,
    required_votes,
    summarize,
    wilson_interval,
)
from ophelia.eval.report import (
    REPORT_FILE,
    REPORT_ROUTE,
    build_report,
    live_report_html,
    render_report_html,
    report_dict,
    report_visible_to,
    write_report,
)
from ophelia.eval.server import AbTestServer, make_server, serve

__all__ = [
    "REPORT_FILE",
    "REPORT_ROUTE",
    "AbTestError",
    "AbTestServer",
    "ListenerReport",
    "Report",
    "Session",
    "Trial",
    "TrialReport",
    "Vote",
    "binomial_p_value",
    "build_report",
    "build_session",
    "default_generated_pool",
    "default_real_pool",
    "load_session",
    "live_report_html",
    "load_votes",
    "make_server",
    "render_report_html",
    "report_dict",
    "report_visible_to",
    "render_clip",
    "required_votes",
    "serve",
    "summarize",
    "wilson_interval",
    "write_report",
]
