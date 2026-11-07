"""Türkçe/İngilizce doğal dil komutunu :class:`CompositionSpec`'e çevirir.

Kural tabanlı ve **deterministik**: aynı cümle her zaman aynı isteği üretir,
ağ gerekmez, test edilebilir. LLM'e ancak bu katman yetmediğinde gidilir.

Anlaşılan kalıplar (örnekler ``tests/test_nlp_parse.py`` içinde):

* Tonalite: ``la minör``, ``A minor``, ``am``, ``fa# minör``, ``do diyez majör``,
  ``Bb major``, ``si bemol minör``
* Tempo: ``120 bpm``, ``90 tempo``, ``yavaş``, ``hızlı``, ``andante``
* Süre: ``30 saniye``, ``1 dakika``, ``8 ölçü``, ``kısa``, ``uzun``
* Doku: ``solo``, ``dört ses``, ``dolu`` / ``kalın``
* Enstrüman: ``keman``, ``kontrbas``, ``yaylılar``, ``piyano``, ``flüt``
* Rol: ``melodi`` / ``ezgi``, ``armoni`` / ``eşlik``, ``bas hattı``, ``ritim``, ``doku``
* Ruh hâli: ``hüzünlü``, ``neşeli``, ``sakin``, ``dramatik``

Ruh hâli bir **varsayılan** kaynağıdır: cümlede açıkça verilen tonalite, tempo
ya da süre her zaman kazanır.
"""

from __future__ import annotations

import re
import unicodedata

from ophelia.nlp.spec import CompositionSpec
from ophelia.theory.instruments import BASS, HARMONY, MELODY, RHYTHM, ROLES, TEXTURE
from ophelia.theory.scales import Key

# --- sözlükler --------------------------------------------------------------

_SOLFEGE = {"do": 0, "re": 2, "mi": 4, "fa": 5, "sol": 7, "la": 9, "si": 11}
_LETTERS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

_SHARP_WORDS = frozenset({"diyez", "sharp", "s"})
_FLAT_WORDS = frozenset({"bemol", "flat"})

_MAJOR_WORDS = frozenset({"major", "maj", "m", "majeur", "buyuk"})
_MINOR_WORDS = frozenset({"minor", "min", "moll", "kucuk"})

_TEMPO_WORDS = {
    "cok yavas": 50.0,
    "largo": 52.0,
    "yavas": 66.0,
    "andante": 76.0,
    "adagio": 70.0,
    "orta": 96.0,
    "moderato": 96.0,
    "hizli": 132.0,
    "allegro": 132.0,
    "cok hizli": 168.0,
    "presto": 168.0,
}

_DURATION_WORDS = {"kisa": 25.0, "uzun": 90.0}

_POLYPHONY_WORDS = {
    "solo": 1,
    "tek ses": 1,
    "monofonik": 1,
    "ikili": 2,
    "duo": 2,
    "oda": 4,
    "ince": 4,
    "dolu": 6,
    "kalin": 6,
    "akorlu": 6,
    "zengin": 6,
}

# Ruh hâli → (mod tercihi, tempo). Dokuya **karışmaz**: kullanıcı polifoni
# istemediyse modelin doğal dokusu korunur ("hüzünlü" diye notaları kesmek
# müziği delik deşik ediyordu).
_MOODS: dict[str, tuple[str, float]] = {
    "huzunlu": ("minor", 68.0),
    "uzgun": ("minor", 68.0),
    "hasret": ("minor", 64.0),
    "dramatik": ("minor", 84.0),
    "karanlik": ("minor", 60.0),
    "kara": ("minor", 60.0),
    "neseli": ("major", 126.0),
    "mutlu": ("major", 120.0),
    "enerjik": ("major", 138.0),
    "canli": ("major", 132.0),
    "parlak": ("major", 120.0),
    "sakin": ("major", 78.0),
    "dingin": ("major", 74.0),
    "huzurlu": ("major", 76.0),
}

_MOOD_ALIASES = {"huzun": "huzunlu", "sevincli": "neseli", "sakinlik": "sakin"}

# Komutta geçen enstrüman sözcükleri → enstrüman kimliği (theory/instruments).
_INSTRUMENT_WORDS: dict[str, str] = {
    "piyano": "piano",
    "piano": "piano",
    "gitar": "nylon_guitar",
    "klasik gitar": "nylon_guitar",
    "keman": "violin",
    "viyola": "viola",
    "cello": "cello",
    "celo": "cello",
    "kontrbas": "contrabass",
    "bas": "bass",
    "flut": "flute",
    "trompet": "trumpet",
    "korno": "french_horn",
    "koro": "choir",
    "yaylilar": "strings",
    "yayli": "strings",
    "strings": "strings",
    "org": "synth_pad",
    "pad": "synth_pad",
    "davul": "drums",
    "perkusyon": "drums",
}

# Rol sözcükleri: düzenlemeyi "perde bölgesi" yerine **müzikal işleve** çevirir.
# Çıplak "bas" burada yok: o sözcük enstrüman seçiyor (bas gitar). Rol olarak
# anlaşılması için cümlede başka bir rol geçmesi gerekir ("melodi, armoni, bas").
_ROLE_WORDS: dict[str, str] = {
    "melodi": MELODY,
    "melody": MELODY,
    "ezgi": MELODY,
    "solist": MELODY,
    "lead": MELODY,
    "armoni": HARMONY,
    "harmony": HARMONY,
    "eslik": HARMONY,
    "eslikci": HARMONY,
    "akor": HARMONY,
    "akompaniman": HARMONY,
    "doku": TEXTURE,
    "texture": TEXTURE,
    "pedal": TEXTURE,
    "ritim": RHYTHM,
    "ritmik": RHYTHM,
    "bas hatti": BASS,
    "bas cizgisi": BASS,
    "bas partisi": BASS,
    "bass line": BASS,
}

_NUMBER_WORDS = {
    "bir": 1,
    "iki": 2,
    "uc": 3,
    "dort": 4,
    "bes": 5,
    "alti": 6,
    "yedi": 7,
    "sekiz": 8,
    "dokuz": 9,
    "on": 10,
    "yirmi": 20,
    "otuz": 30,
    "kirk": 40,
    "elli": 50,
    "altmis": 60,
    "yetmis": 70,
    "seksen": 80,
}


def normalize(text: str) -> str:
    """Küçük harfe indirger, Türkçe karakterleri sadeleştirir, noktalama atar."""
    lowered = text.replace("İ", "i").replace("I", "i").lower()
    decomposed = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    stripped = (
        stripped.replace("ı", "i")
        .replace("ş", "s")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ö", "o")
        .replace("ç", "c")
    )
    return re.sub(r"[^a-z0-9#]+", " ", stripped).strip()


def _number(token: str) -> float | None:
    if token.isdigit():
        return float(token)
    return float(_NUMBER_WORDS[token]) if token in _NUMBER_WORDS else None


def _parse_key(words: list[str]) -> tuple[Key | None, str | None]:
    """Cümledeki tonaliteyi bulur; (tonalite, gerekçe) döner."""
    index = 0
    while index < len(words):
        word = words[index]
        tonic: int | None = None
        used = 0
        flat = False

        # 1) Solfej: do / re / mi ... (+ diyez|bemol)
        if word in _SOLFEGE:
            tonic, used = _SOLFEGE[word], 1
            if index + 1 < len(words) and words[index + 1] in _SHARP_WORDS | _FLAT_WORDS:
                flat = words[index + 1] in _FLAT_WORDS
                tonic += -1 if flat else 1
                used = 2
        # 2) Harf + minör kısaltması: am / cm / f#m
        elif re.fullmatch(r"([a-g])(#|b)?m", word):
            tonic = _LETTERS[word[0]]
            if word.endswith("#m"):
                tonic += 1
            elif word.endswith("bm") and len(word) == 3:
                tonic -= 1
                flat = True
            used = 1
            key = Key(tonic % 12, "natural_minor", flats=flat)
            return key, f"tonalite: '{word.upper()}' → {key.name}"
        # 3) Harf: c / f# / bb
        elif re.fullmatch(r"[a-g](#|b)?", word):
            tonic = _LETTERS[word[0]]
            if word.endswith("#"):
                tonic += 1
            elif len(word) == 2 and word.endswith("b"):
                tonic -= 1
                flat = True
            used = 1

        if tonic is None:
            index += 1
            continue

        mode: str | None = None
        # "la minör", "a minor", "am"
        if index + used < len(words):
            candidate = words[index + used]
            if candidate in _MINOR_WORDS:
                mode = "natural_minor"
                used += 1
            elif candidate in _MAJOR_WORDS:
                mode = "major"
                used += 1
        if mode is None and word in _SOLFEGE and used == 1:
            # "la" tek başına tonalite değil (artikel/kısaltma olabilir)
            index += 1
            continue

        key = Key(tonic % 12, mode or "major", flats=flat)
        spelled = " ".join(words[index : index + used]).upper()
        return key, f"tonalite: '{spelled}' → {key.name}"

    return None, None


def _parse_tempo(words: list[str]) -> tuple[float | None, str | None]:
    for index, word in enumerate(words):
        if word in {"bpm", "tempo"} and index > 0:
            number = _number(words[index - 1])
            if number:
                return number, f"tempo: '{words[index - 1]} {word}' → {number:.0f} BPM"
        number = _number(word)
        if number and index + 1 < len(words) and words[index + 1] in {"bpm", "tempo"}:
            return number, f"tempo: '{word} {words[index + 1]}' → {number:.0f} BPM"

    for phrase, value in _TEMPO_WORDS.items():
        if phrase in " ".join(words):
            return value, f"tempo: '{phrase}' → {value:.0f} BPM"
    return None, None


def _parse_duration(words: list[str]) -> tuple[float | None, str | None, float | None]:
    """(saniye, gerekçe, ölçü sayısı) döner."""
    for index, word in enumerate(words):
        number = _number(word)
        if not number or index + 1 >= len(words):
            continue
        unit = words[index + 1]
        if unit in {"saniye", "sn", "s", "sec", "second", "seconds"}:
            return number, f"süre: '{word} {unit}' → {number:.0f} sn", None
        if unit in {"dakika", "dk", "min", "minute", "minutes"}:
            return (
                number * 60,
                f"süre: '{word} {unit}' → {number * 60:.0f} sn",
                None,
            )
        if unit in {"olcu", "bar", "bars", "measure", "measures"}:
            return None, f"süre: '{word} {unit}' → {number:.0f} ölçü", number

    for phrase, value in _DURATION_WORDS.items():
        if phrase in words:
            return value, f"süre: '{phrase}' → {value:.0f} sn", None
    return None, None, None


def _parse_polyphony(words: list[str]) -> tuple[int | None, str | None]:
    text = " ".join(words)
    for phrase, value in _POLYPHONY_WORDS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return value, f"doku: '{phrase}' → en fazla {value} ses"

    for index, word in enumerate(words):
        number = _number(word)
        if number and index + 1 < len(words) and words[index + 1] == "ses":
            value = max(1, min(8, int(number)))
            return value, f"doku: '{word} ses' → en fazla {value} ses"
    return None, None


#: Topluluk adları tek bir pad sesi değil **bölüm** demektir: "yaylılar"
#: dendiğinde dört yaylı sesi ayrı ayrı çalar. Tek bir pad'e düşen notalar
#: "oyun müziği" gibi duyuluyordu (tek tını, bölüm ayrımı yok).
_SECTIONS: dict[str, tuple[str, ...]] = {
    "strings": ("violin", "viola", "cello", "contrabass"),
}


def _parse_ensemble(words: list[str]) -> tuple[tuple[str, ...], str | None]:
    """Cümledeki enstrümanları sırayla toplar (tizden pese kullanılacak)."""
    found: list[str] = []
    text = " ".join(words)
    for phrase in sorted(_INSTRUMENT_WORDS, key=len, reverse=True):
        if not re.search(rf"\b{re.escape(phrase)}\b", text):
            continue
        slug = _INSTRUMENT_WORDS[phrase]
        for member in _SECTIONS.get(slug, (slug,)):
            if member not in found:
                found.append(member)

    if not found:
        return (), None
    return tuple(found), f"enstrüman: {', '.join(found)}"


def _parse_roles(words: list[str]) -> tuple[tuple[str, ...], str | None]:
    """Cümledeki roller (= sesler) sırayla; çıplak "bas" yalnızca eşlik eder."""
    text = " ".join(words)
    found: list[str] = []
    for phrase in sorted(_ROLE_WORDS, key=len, reverse=True):
        if not re.search(rf"\b{re.escape(phrase)}\b", text):
            continue
        role = _ROLE_WORDS[phrase]
        if role not in found:
            found.append(role)

    # "melodi, armoni ve bas" → üçüncü rol de bas.
    if found and "bas" in words and BASS not in found:
        found.append(BASS)

    if not found:
        return (), None
    ordered = tuple(sorted(found, key=ROLES.index))
    return ordered, f"roller: {', '.join(ordered)} → {len(ordered)} ses"


def _parse_mood(words: list[str]) -> tuple[str, float] | None:
    for word in words:
        mood = _MOOD_ALIASES.get(word, word)
        if mood in _MOODS:
            return _MOODS[mood]
    return None


def parse_command(command: str, *, seed: int | None = None) -> CompositionSpec:
    """Komutu çözer; hiçbir şey bulunamazsa varsayılan istek üretir."""
    words = normalize(command).split()
    notes: list[str] = []

    key, key_note = _parse_key(words)
    tempo, tempo_note = _parse_tempo(words)
    duration, duration_note, bars = _parse_duration(words)
    polyphony, polyphony_note = _parse_polyphony(words)
    ensemble, ensemble_note = _parse_ensemble(words)
    roles, roles_note = _parse_roles(words)
    mood = _parse_mood(words)

    if bars is not None:
        effective_tempo = tempo or (mood[1] if mood else 96.0)
        duration = bars * 4 * 60 / effective_tempo
        duration_note = f"{duration_note} @ {effective_tempo:.0f} BPM → {duration:.0f} sn"

    if mood is not None and key is None:
        mode = mood[0]
        key = Key(0 if mode == "major" else 9, mode)
        notes.append(f"ruh hâli: {mode} → varsayılan {key.name}")

    if key_note:
        notes.append(key_note)
    if tempo_note:
        notes.append(tempo_note)
    if duration_note:
        notes.append(duration_note)
    if polyphony_note:
        notes.append(polyphony_note)
    if ensemble_note:
        notes.append(ensemble_note)
    if roles_note:
        notes.append(roles_note)

    spec = CompositionSpec(
        command=command,
        key=key,
        tempo_bpm=tempo if tempo is not None else (mood[1] if mood else 96.0),
        duration_seconds=duration if duration is not None else 45.0,
        max_polyphony=polyphony if polyphony is not None else 6,
        seed=seed,
        ensemble=ensemble,
        roles=roles,
        notes=tuple(notes),
    )

    if not notes:
        return spec.with_notes("tanınan parametre yok; varsayılan istek kullanıldı")
    return spec
