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
from ophelia.eval.server import AbTestServer, make_server, serve

__all__ = [
    "AbTestError",
    "AbTestServer",
    "ListenerReport",
    "Report",
    "Session",
    "Trial",
    "TrialReport",
    "Vote",
    "binomial_p_value",
    "build_session",
    "default_generated_pool",
    "default_real_pool",
    "load_session",
    "load_votes",
    "make_server",
    "render_clip",
    "required_votes",
    "serve",
    "summarize",
    "wilson_interval",
]
