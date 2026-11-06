"""Doğal dil → CompositionSpec çözümleme testleri."""

from __future__ import annotations

import pytest
from ophelia.nlp.parse import normalize, parse_command
from ophelia.nlp.spec import CompositionSpec


def test_normalize_strips_turkish_diacritics() -> None:
    assert normalize("Hüzünlü, La Minör!") == "huzunlu la minor"
    assert normalize("İSTANBUL şarkı") == "istanbul sarki"


@pytest.mark.parametrize(
    ("command", "tonic", "mode"),
    [
        ("la minör", 9, "natural_minor"),
        ("La minör bir parça", 9, "natural_minor"),
        ("A minor", 9, "natural_minor"),
        ("am", 9, "natural_minor"),
        ("do majör", 0, "major"),
        ("C major", 0, "major"),
        ("fa diyez minör", 6, "natural_minor"),
        ("F# minor", 6, "natural_minor"),
        ("si bemol majör", 10, "major"),
        ("Bb major", 10, "major"),
    ],
)
def test_key_parsing(command: str, tonic: int, mode: str) -> None:
    spec = parse_command(command)
    assert spec.key is not None
    assert spec.key.tonic_pc == tonic
    assert spec.key.mode == mode


def test_flat_key_is_spelled_with_flats() -> None:
    spec = parse_command("Bb major")
    assert spec.key is not None
    assert spec.key.name == "Bb major"


def test_key_not_named_and_no_mood_leaves_key_free() -> None:
    assert parse_command("bir piyano parçası").key is None


def test_mood_supplies_default_tonic_when_key_missing() -> None:
    """Ruh hâli tonik uydurmaz ama makul bir varsayılan seçer ve bunu söyler."""
    spec = parse_command("sakin bir piyano parçası")
    assert spec.key is not None
    assert spec.key.name == "C major"
    assert any("varsayılan" in note for note in spec.notes)


@pytest.mark.parametrize(
    ("command", "tempo"),
    [
        ("120 bpm", 120.0),
        ("90 tempo la minör", 90.0),
        ("yavaş bir eser", 66.0),
        ("hızlı ve neşeli", 132.0),
        ("andante", 76.0),
    ],
)
def test_tempo_parsing(command: str, tempo: float) -> None:
    assert parse_command(command).tempo_bpm == tempo


@pytest.mark.parametrize(
    ("command", "seconds"),
    [
        ("30 saniye", 30.0),
        ("45 sn", 45.0),
        ("1 dakika", 60.0),
        ("kısa bir parça", 25.0),
        ("uzun eser", 90.0),
    ],
)
def test_duration_parsing(command: str, seconds: float) -> None:
    assert parse_command(command).duration_seconds == seconds


def test_bars_need_tempo_to_become_seconds() -> None:
    spec = parse_command("8 ölçü 120 bpm")
    # 8 ölçü × 4 vuruş × 60/120 = 16 sn
    assert abs(spec.duration_seconds - 16.0) < 0.5


def test_bars_use_mood_tempo_when_tempo_missing() -> None:
    spec = parse_command("4 ölçü hüzünlü")
    # 4 ölçü × 4 vuruş × 60/68 ≈ 14,1 sn
    assert abs(spec.duration_seconds - 14.1) < 1.0


@pytest.mark.parametrize(
    ("command", "polyphony"),
    [("solo", 1), ("tek ses", 1), ("dört ses", 4), ("dolu doku", 6), ("oda müziği", 4)],
)
def test_polyphony_parsing(command: str, polyphony: int) -> None:
    assert parse_command(command).max_polyphony == polyphony


def test_mood_sets_mode_and_defaults() -> None:
    spec = parse_command("hüzünlü bir parça")
    assert spec.key is not None
    assert spec.key.mode == "natural_minor"
    assert spec.key.name == "A natural_minor"
    assert spec.tempo_bpm == 68.0

    cheerful = parse_command("neşeli bir eser")
    assert cheerful.key is not None
    assert cheerful.key.mode == "major"
    assert cheerful.tempo_bpm == 126.0


def test_explicit_values_beat_mood_defaults() -> None:
    spec = parse_command("hüzünlü ama neşeli C major 140 bpm 20 saniye")
    assert spec.key is not None
    assert spec.key.name == "C major"
    assert spec.tempo_bpm == 140.0
    assert spec.duration_seconds == 20.0


def test_unknown_command_falls_back_to_defaults() -> None:
    spec = parse_command("bilmem ne")
    assert spec.key is None
    assert spec.tempo_bpm == 96.0
    assert spec.duration_seconds == 45.0
    assert "varsayılan" in " ".join(spec.notes)


def test_notes_explain_the_decisions() -> None:
    spec = parse_command("la minör 100 bpm 40 saniye")
    joined = " ".join(spec.notes)
    assert "tonalite" in joined
    assert "tempo" in joined
    assert "süre" in joined


def test_seed_is_carried_into_spec() -> None:
    assert parse_command("la minör", seed=7).seed == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tempo_bpm": 10.0},
        {"tempo_bpm": 400.0},
        {"duration_seconds": 1.0},
        {"max_polyphony": 0},
        {"max_polyphony": 12},
    ],
)
def test_spec_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CompositionSpec(**kwargs)  # type: ignore[arg-type]


def test_spec_bars_and_beats() -> None:
    spec = CompositionSpec(tempo_bpm=120.0, duration_seconds=16.0)
    assert spec.beats == 32.0
    assert spec.bars == 8.0


def test_roles_are_parsed_and_ordered() -> None:
    spec = parse_command("bas ve melodi ile bir eser")
    assert spec.roles == ("melody", "bass")
    assert "bass" in spec.ensemble  # "bas" enstrüman olarak da anlaşılır


def test_turkish_role_words() -> None:
    spec = parse_command("ezgi, eşlik ve bas hattı")
    assert spec.roles == ("melody", "harmony", "bass")


def test_spec_normalizes_role_aliases() -> None:
    assert CompositionSpec(roles=("melodi", "bas")).roles == ("melody", "bass")


def test_bare_bas_is_instrument_not_role() -> None:
    """Tek başına "bas" düzenleme kuralını değiştirmemeli: enstrüman seçer."""
    spec = parse_command("kontrbas ve piyano ile sakin bir parça")
    assert spec.roles == ()
    assert set(spec.ensemble) == {"piano", "contrabass"}


def test_role_decision_is_explained() -> None:
    spec = parse_command("melodi ve armoni")
    assert any("roller" in note for note in spec.notes)


def test_spec_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="rol"):
        CompositionSpec(roles=("rastgele",))


def test_spec_dict_includes_roles() -> None:
    spec = CompositionSpec(roles=("melody", "bass"))
    assert spec.to_dict()["roles"] == ["melody", "bass"]
