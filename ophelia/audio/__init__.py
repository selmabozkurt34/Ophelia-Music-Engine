"""Ses üretimi (7-8. aşama): düzenleme → sentez → salon → yayın zinciri.

Akış: tek kanallı notalar → [ensemble](ensemble.py) (seslere ayırma, enstrüman
ataması, çok kanallı MIDI) → [timbre](timbre.py) (sürekli tını, ADSR) →
[render](render.py) (miks, pan) → [reverb](reverb.py) (salon) →
[mastering](mastering.py) (−14 LUFS, true peak).
"""

from __future__ import annotations

from ophelia.audio.ensemble import (
    CHORALE_ENSEMBLE,
    ROLE_ORDER,
    Part,
    arrange,
    arrange_by_role,
    chorale_ensemble,
    fit_to_range,
    split_into_voices,
    write_multitrack_midi,
)
from ophelia.audio.mastering import MasterStats, integrated_lufs, master, true_peak
from ophelia.audio.render import (
    SAMPLE_RATE,
    MixSettings,
    channel_correlation,
    loudness,
    peak,
    render_file,
    render_parts,
    render_with_stats,
    rms,
    stereo_width,
    synthesize_part,
    write_wav,
)
from ophelia.audio.reverb import apply_reverb, hall_ir

__all__ = [
    "CHORALE_ENSEMBLE",
    "MixSettings",
    "MasterStats",
    "Part",
    "ROLE_ORDER",
    "apply_reverb",
    "arrange",
    "arrange_by_role",
    "channel_correlation",
    "chorale_ensemble",
    "fit_to_range",
    "hall_ir",
    "integrated_lufs",
    "loudness",
    "master",
    "peak",
    "render_file",
    "render_parts",
    "render_with_stats",
    "rms",
    "SAMPLE_RATE",
    "split_into_voices",
    "stereo_width",
    "synthesize_part",
    "true_peak",
    "write_multitrack_midi",
    "write_wav",
]
