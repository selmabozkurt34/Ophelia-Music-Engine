"""Enstrüman kaydı, rolleri ve kanal ataması testleri."""

from __future__ import annotations

import pytest
from ophelia.theory.instruments import (
    DEFAULT_ENSEMBLE,
    DRUM_CHANNEL,
    INSTRUMENTS,
    ROLES,
    all_instruments,
    assign_channels,
    by_role,
    default_ensemble,
    get,
)


def test_registry_contents() -> None:
    assert len(INSTRUMENTS) >= 20
    assert get("piano").gm_program == 0
    assert get("strings").gm_program == 48
    assert get("synth_pad").role == "texture"
    assert get("drums").is_drum


def test_get_is_case_insensitive() -> None:
    assert get("Piano").slug == "piano"
    assert get("  SYNTH_LEAD  ").slug == "synth_lead"


def test_unknown_instrument_raises() -> None:
    with pytest.raises(ValueError):
        get("theremin")
    with pytest.raises(ValueError):
        by_role("olmayan_rol")


def test_roles() -> None:
    melody = by_role("melody")
    assert any(instrument.slug == "violin" for instrument in melody)
    assert any(instrument.slug == "synth_lead" for instrument in melody)
    assert all(instrument.role in ROLES for instrument in all_instruments())


def test_pitch_range_helpers() -> None:
    piano = get("piano")
    assert piano.midi_range == (21, 108)
    assert piano.in_range(60)
    assert not piano.in_range(10)
    assert piano.clamp(10) == 21
    assert piano.clamp(120) == 108

    bass = get("bass")
    assert bass.clamp(90) < 90


def test_assign_channels_puts_drums_on_nine() -> None:
    ensemble = [get("piano"), get("drums"), get("bass"), get("strings")]
    channels = assign_channels(ensemble)
    values = list(channels.values())

    assert DRUM_CHANNEL in values
    assert len(values) == len(set(values))
    assert channels["drums_1"] == DRUM_CHANNEL
    assert 9 not in [channel for key, channel in channels.items() if not key.startswith("drums")]


def test_assign_channels_rejects_overflow() -> None:
    non_drums = [instrument for instrument in all_instruments() if not instrument.is_drum][:17]
    with pytest.raises(ValueError):
        assign_channels(non_drums)


def test_default_ensemble() -> None:
    ensemble = default_ensemble()
    assert len(ensemble) == len(DEFAULT_ENSEMBLE)
    assert any(instrument.is_drum for instrument in ensemble)
    assert any(instrument.role == "bass" for instrument in ensemble)


def test_instrument_validation() -> None:
    from ophelia.theory.instruments import Instrument

    with pytest.raises(ValueError):
        Instrument("x", "X", 200, "melody", 60, 70)
    with pytest.raises(ValueError):
        Instrument("x", "X", 0, "olmayan", 60, 70)
    with pytest.raises(ValueError):
        Instrument("x", "X", 0, "melody", 80, 60)
