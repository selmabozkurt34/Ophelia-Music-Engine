"""Ses üretimi (7. aşama): düzenleme → sentez → miks.

Akış: tek kanallı notalar → [ensemble](ensemble.py) (seslere ayırma, enstrüman
ataması, çok kanallı MIDI) → [render](render.py) (harmonik sentez, pan, stereo WAV).
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
from ophelia.audio.render import (
    SAMPLE_RATE,
    MixSettings,
    peak,
    render_file,
    render_parts,
    rms,
    stereo_width,
    synthesize_part,
    write_wav,
)

__all__ = [
    "CHORALE_ENSEMBLE",
    "MixSettings",
    "Part",
    "ROLE_ORDER",
    "chorale_ensemble",
    "SAMPLE_RATE",
    "arrange",
    "arrange_by_role",
    "fit_to_range",
    "peak",
    "render_file",
    "render_parts",
    "rms",
    "split_into_voices",
    "stereo_width",
    "synthesize_part",
    "write_multitrack_midi",
    "write_wav",
]
