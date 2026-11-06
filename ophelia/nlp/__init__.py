"""Doğal dil komut katmanı (6. aşama).

Akış: ``komut`` → [parse](parse.py) → [CompositionSpec](spec.py) →
[seeding](seeding.py) → model → [arrange](arrange.py) → MIDI.
"""

from __future__ import annotations

from ophelia.nlp.arrange import (
    AdjustReport,
    clip_to_duration,
    estimated_key,
    project_to_scale,
    transpose_to_key,
)
from ophelia.nlp.compose import (
    ComposeResult,
    compose,
    generate_until,
    load_models,
    tokens_for,
)
from ophelia.nlp.parse import normalize, parse_command
from ophelia.nlp.seeding import prompt_ids, seed_notes, seed_tokens
from ophelia.nlp.spec import CompositionSpec

__all__ = [
    "AdjustReport",
    "ComposeResult",
    "CompositionSpec",
    "compose",
    "generate_until",
    "project_to_scale",
    "clip_to_duration",
    "estimated_key",
    "load_models",
    "normalize",
    "parse_command",
    "prompt_ids",
    "seed_notes",
    "seed_tokens",
    "tokens_for",
    "transpose_to_key",
]
