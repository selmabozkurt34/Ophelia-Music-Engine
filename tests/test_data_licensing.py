"""Lisans modeli ve kaynak kararlarının testleri."""

from __future__ import annotations

import pytest
from ophelia.data.licensing import LICENSES, get_license, is_commercially_usable
from ophelia.data.sources import (
    APPROVED,
    REJECTED,
    REVIEW,
    SOURCES,
    approved_sources,
    assert_usable,
    get_source,
    rejected_sources,
    review_sources,
)


def test_license_registry_consistency() -> None:
    for license_id, license in LICENSES.items():
        assert license.id == license_id
        assert license.name
        assert isinstance(license.commercial_use, bool)


def test_noncommercial_licenses_flagged() -> None:
    assert not is_commercially_usable("CC-BY-NC-SA-4.0")
    assert not is_commercially_usable("research-only")
    assert not is_commercially_usable("unknown")


def test_permissive_licenses_flagged() -> None:
    assert is_commercially_usable("CC0-1.0")
    assert is_commercially_usable("PD")
    assert is_commercially_usable("CC-BY-4.0")


def test_unknown_license_raises() -> None:
    with pytest.raises(ValueError):
        get_license("WTFPL")


def test_license_open_flag() -> None:
    assert get_license("CC0-1.0").is_open
    assert not get_license("CC-BY-SA-4.0").is_open
    assert not get_license("CC-BY-NC-SA-4.0").is_open


def test_attribution_text() -> None:
    assert "atıf gerekmiyor" in get_license("CC0-1.0").attribution_text("PDMX")
    assert "Creative Commons Attribution 4.0" in get_license("CC-BY-4.0").attribution_text(
        "GiantMIDI"
    )


def test_source_registry_verdicts() -> None:
    assert get_source("pdmx").verdict == APPROVED
    assert get_source("maestro").verdict == REJECTED
    assert get_source("giantmidi-piano").verdict == REVIEW
    assert get_source("PDMX").slug == "pdmx"  # büyük/küçük harf duyarsız


def test_approved_sources_are_all_commercial() -> None:
    for source in approved_sources():
        assert source.usable, source.slug
        assert source.license.commercial_use


def test_rejected_sources_reasons() -> None:
    rejected = {source.slug for source in rejected_sources()}
    assert {"maestro", "metamidi", "essen", "lakh"} <= rejected
    for source in rejected_sources():
        assert source.rationale


def test_no_source_is_both_approved_and_rejected() -> None:
    approved = {source.slug for source in approved_sources()}
    rejected = {source.slug for source in rejected_sources()}
    review = {source.slug for source in review_sources()}
    assert not approved & rejected
    assert not approved & review
    assert len(SOURCES) == len(approved) + len(rejected) + len(review)


def test_assert_usable_gate() -> None:
    assert assert_usable("music21-bach-chorales").slug == "music21-bach-chorales"

    with pytest.raises(PermissionError):
        assert_usable("maestro")
    with pytest.raises(PermissionError):
        assert_usable("giantmidi-piano")  # review -> onaylı değil
    with pytest.raises(ValueError):
        assert_usable("olmayan-kaynak")


def test_get_source_error_lists_options() -> None:
    with pytest.raises(ValueError, match="pdmx"):
        get_source("yok")
