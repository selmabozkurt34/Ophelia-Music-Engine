"""Nota olaylarını MusicXML'e (partwise 4.0) aktarır.

Neden gerekli: modelin çıktısı MIDI olarak duyuluyor ama **nota olarak
düzenlenemiyor**. MusicXML, MuseScore / Dorico / Sibelius / Finale gibi
araçların ortak dili; piyasa analizinde notasyon pazarına girişin ön koşulu
olarak işaretlenmişti (P0).

Tasarım kararları:

* Yazan taraf ``xml.etree`` — elle dizgi birleştirme yok, kaçış karakterleri
  kütüphanenin işi.
* Ölçüler **hesaplanır**, MIDI'nin ham tick'i olduğu gibi yazılmaz: ölçü çizgisini
  aşan notalar bağ (``tie``) ile bölünür, boşluklar sus ile doldurulur. Aksi
  hâlde notasyon programı "ölçü eksik" uyarısı verir ya da notaları kaydırır.
* Aynı başlangıç ve süredeki notalar ``<chord/>`` ile tek akora toplanır.
  Üst üste binen notalar (model tek kanalda çoksesli yazıyor) **ayrı seslere**
  dağıtılır ve sesler arasına ``<backup>`` konur; tek sese sığdırmak süreleri
  birbirine yapıştırıyordu.
* Çıktı **deterministik**: aynı notalar aynı baytları üretir (tarih, rastgele
  kimlik yazılmaz).

Sınır: MusicXML nota başına velocity taşımaz; ortalama velocity bir kez
``<sound dynamics>`` olarak yazılır. Perde ve süre bilgisi kayıpsızdır.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from ophelia.data.midi_io import NoteEvent, iter_notes, load_midi
from ophelia.theory.instruments import INSTRUMENTS
from ophelia.theory.scales import estimate_key

if TYPE_CHECKING:  # pragma: no cover - yalnızca tip denetimi
    from collections.abc import Sequence

    from ophelia.audio.ensemble import Part

MUSICXML_VERSION = "4.0"
SOFTWARE_NAME = "Ophelia Music Engine"
DEFAULT_TITLE = "Ophelia eseri"
DEFAULT_TEMPO_BPM = 100.0
DEFAULT_TIME_SIGNATURE = (4, 4)
TICKS_PER_BEAT = 480

DOCTYPE = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">\n'
)

# Perde sınıfı → (step, alter). Anahtar iminde diyez varsa diyez, bemol varsa
# bemol yazılır: Si bemol majörde "A#" değil "Bb" görünmeli.
_SHARP_SPELLING = (
    ("C", 0),
    ("C", 1),
    ("D", 0),
    ("D", 1),
    ("E", 0),
    ("F", 0),
    ("F", 1),
    ("G", 0),
    ("G", 1),
    ("A", 0),
    ("A", 1),
    ("B", 0),
)
_FLAT_SPELLING = (
    ("C", 0),
    ("D", -1),
    ("D", 0),
    ("E", -1),
    ("E", 0),
    ("F", 0),
    ("G", -1),
    ("G", 0),
    ("A", -1),
    ("A", 0),
    ("B", -1),
    ("B", 0),
)

# Anahtar iminin sırasıyla diyezlediği/bemolleştirdiği perde sınıfları.
_SHARP_ORDER = (5, 0, 7, 2, 9, 4, 11)  # F C G D A E B
_FLAT_ORDER = (11, 4, 9, 2, 7, 0, 5)  # B E A D G C F

# Nota değerleri, çeyrek vuruş cinsinden (divisions ile çarpılır).
_VALUE_NAMES = (
    ("whole", 4.0),
    ("half", 2.0),
    ("quarter", 1.0),
    ("eighth", 0.5),
    ("16th", 0.25),
    ("32nd", 0.125),
    ("64th", 0.0625),
)

# Ortalama velocity → dinamik işareti (üst sınırlar).
_DYNAMICS = ((40, "p"), (55, "mp"), (70, "mf"), (85, "f"))


@dataclass(frozen=True)
class XmlPart:
    """MusicXML'e yazılacak tek parti (bir enstrümanın notaları)."""

    name: str
    notes: list[NoteEvent]
    program: int = 0
    channel: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.program <= 127:
            raise ValueError(f"GM program 0-127 olmalı: {self.program}")
        if not 0 <= self.channel <= 15:
            raise ValueError(f"MIDI kanalı 0-15 olmalı: {self.channel}")


@dataclass(frozen=True)
class _Segment:
    """Bir notanın tek ölçüye düşen parçası."""

    measure: int
    offset: int
    duration: int
    pitch: int
    velocity: int
    voice: int = 1
    tie_start: bool = False
    tie_stop: bool = False


def midi_name(pitch: int, fifths: int = 0) -> tuple[str, int, int]:
    """MIDI numarasını (step, alter, oktav) üçlüsüne çevirir.

    MIDI oktavı -1'de başlar (60 = C4); MusicXML de aynı numarayı kullanır.
    """
    spelling = _FLAT_SPELLING if fifths < 0 else _SHARP_SPELLING
    step, alter = spelling[pitch % 12]
    return step, alter, pitch // 12 - 1


def fifths_for_key(tonic_pc: int, mode: str = "major") -> int:
    """Tonaliteyi MusicXML ``<fifths>`` değerine çevirir (minör → ilgili majör)."""
    if "minor" in mode:
        tonic_pc = (tonic_pc + 3) % 12
    fifths = (tonic_pc * 7) % 12
    return fifths if fifths <= 6 else fifths - 12


def detect_fifths(parts: Sequence[XmlPart]) -> int:
    """Notaların perde histogramından anahtar imini tahmin eder."""
    histogram: dict[int, float] = {}
    for part in parts:
        for note in part.notes:
            histogram[note.pitch % 12] = (
                histogram.get(note.pitch % 12, 0.0) + note.duration / TICKS_PER_BEAT
            )
    if not histogram:
        return 0
    key = estimate_key(histogram)
    return fifths_for_key(key.tonic_pc, key.mode)


def _implied_alter(pitch_pc: int, fifths: int) -> int:
    """Anahtar iminin bu perde sınıfı için zaten söylediği değiştirici."""
    if fifths > 0:
        return 1 if pitch_pc in _SHARP_ORDER[:fifths] else 0
    if fifths < 0:
        return -1 if pitch_pc in _FLAT_ORDER[:-fifths] else 0
    return 0


def _type_for(ticks: int, divisions: int) -> tuple[str | None, int]:
    """Süreyi notasyon değerine çevirir: (ad, nokta sayısı); uymuyorsa (None, 0)."""
    for name, quarters in _VALUE_NAMES:
        base = int(round(quarters * divisions))
        if base <= 0:
            continue
        if ticks == base:
            return name, 0
        if ticks == base * 3 // 2 and base % 2 == 0:
            return name, 1
    return None, 0


def _rest_splits(ticks: int, divisions: int) -> list[tuple[int, str | None, int]]:
    """Bir sus süresini yazılabilir değerlere böler (büyükten küçüğe)."""
    candidates: list[tuple[int, str, int]] = []
    for name, quarters in _VALUE_NAMES[:6]:
        base = int(round(quarters * divisions))
        if base <= 0:
            continue
        if base % 2 == 0:
            candidates.append((base * 3 // 2, name, 1))
        candidates.append((base, name, 0))
    candidates.sort(key=lambda item: -item[0])

    pieces: list[tuple[int, str | None, int]] = []
    remaining = ticks
    while remaining > 0:
        for value, name, dots in candidates:
            if 0 < value <= remaining:
                pieces.append((value, name, dots))
                remaining -= value
                break
        else:  # 64'lükten küçük artık: değer yazılmaz, süre korunur
            pieces.append((remaining, None, 0))
            break
    return pieces


def _assign_voices(notes: Sequence[NoteEvent]) -> list[tuple[NoteEvent, int]]:
    """Notaları üst üste binmeyen seslere dağıtır (aralık boyama, aç gözlü).

    Model tek kanalda çoksesli yazıyor: aynı anda başlayan notaların süreleri
    farklı olabiliyor, notasyonda bir seste üst üste binen iki nota olamaz.

    Atama **akor birimi** düzeyinde yapılır: aynı başlangıç ve süreye sahip
    notalar (akor) tek birimdir ve aynı sese gider. Birimler boşta olan ilk sese
    verilir. Ölçü çizgisini aşan nota sonradan ikiye bölünür; parçaların aynı
    seste kalması için atama bölmeden **önce** yapılır (aksi hâlde bağ iki ses
    arasında kalır).
    """
    units: dict[tuple[int, int], list[NoteEvent]] = {}
    for note in notes:
        units.setdefault((note.start, note.end), []).append(note)

    lanes: list[int] = []
    voices: dict[tuple[int, int], int] = {}
    for (start, end), _ in sorted(units.items(), key=lambda item: (item[0][0], -item[0][1])):
        for index, lane_end in enumerate(lanes):
            if lane_end <= start:
                lanes[index] = end
                voices[(start, end)] = index + 1
                break
        else:
            lanes.append(end)
            voices[(start, end)] = len(lanes)

    return [(note, voices[(note.start, note.end)]) for note in notes]


def _segments(
    notes: Sequence[tuple[NoteEvent, int]], measure_ticks: int
) -> tuple[list[_Segment], int]:
    """Notaları ölçü sınırlarından böler; (parçalar, son ölçü indeksi)."""
    segments: list[_Segment] = []
    last_measure = 0
    for note, voice in notes:
        start = note.start
        end = note.end
        while start < end:
            index = start // measure_ticks
            piece_end = min(end, (index + 1) * measure_ticks)
            segments.append(
                _Segment(
                    measure=index,
                    offset=start - index * measure_ticks,
                    duration=piece_end - start,
                    pitch=note.pitch,
                    velocity=note.velocity,
                    voice=voice,
                    tie_start=piece_end < end,
                    tie_stop=start > note.start,
                )
            )
            last_measure = max(last_measure, index)
            start = piece_end
    return segments, last_measure


def _append_rest(measure: ET.Element, ticks: int, divisions: int, *, voice: int = 1) -> None:
    """Sus yazar; uzun suslar standart değerlere bölünür."""
    for value, name, dots in _rest_splits(ticks, divisions):
        note = ET.SubElement(measure, "note")
        ET.SubElement(note, "rest")
        ET.SubElement(note, "duration").text = str(value)
        ET.SubElement(note, "voice").text = str(voice)
        if name:
            ET.SubElement(note, "type").text = name
            for _ in range(dots):
                ET.SubElement(note, "dot")


def _append_note(
    measure: ET.Element,
    segment: _Segment,
    *,
    divisions: int,
    fifths: int,
    chord: bool = False,
    voice: int = 1,
) -> None:
    note = ET.SubElement(measure, "note")
    if chord:
        ET.SubElement(note, "chord")

    step, alter, octave = midi_name(segment.pitch, fifths)
    pitch = ET.SubElement(note, "pitch")
    ET.SubElement(pitch, "step").text = step
    if alter:
        ET.SubElement(pitch, "alter").text = str(alter)
    ET.SubElement(pitch, "octave").text = str(octave)

    ET.SubElement(note, "duration").text = str(segment.duration)
    if segment.tie_stop:
        ET.SubElement(note, "tie", {"type": "stop"})
    if segment.tie_start:
        ET.SubElement(note, "tie", {"type": "start"})
    ET.SubElement(note, "voice").text = str(voice)

    name, dots = _type_for(segment.duration, divisions)
    if name:
        ET.SubElement(note, "type").text = name
        for _ in range(dots):
            ET.SubElement(note, "dot")

    implied = _implied_alter(segment.pitch % 12, fifths)
    if alter != implied:
        accidental = "sharp" if alter > 0 else "flat" if alter < 0 else "natural"
        ET.SubElement(note, "accidental").text = accidental

    if segment.tie_stop or segment.tie_start:
        notations = ET.SubElement(note, "notations")
        if segment.tie_stop:
            ET.SubElement(notations, "tied", {"type": "stop"})
        if segment.tie_start:
            ET.SubElement(notations, "tied", {"type": "start"})


def _append_attributes(
    measure: ET.Element,
    *,
    divisions: int,
    fifths: int,
    time_signature: tuple[int, int],
    clef: tuple[str, str],
) -> None:
    attributes = ET.SubElement(measure, "attributes")
    ET.SubElement(attributes, "divisions").text = str(divisions)
    key = ET.SubElement(attributes, "key")
    ET.SubElement(key, "fifths").text = str(fifths)
    time = ET.SubElement(attributes, "time")
    ET.SubElement(time, "beats").text = str(time_signature[0])
    ET.SubElement(time, "beat-type").text = str(time_signature[1])
    clef_element = ET.SubElement(attributes, "clef")
    ET.SubElement(clef_element, "sign").text = clef[0]
    ET.SubElement(clef_element, "line").text = clef[1]


def _dynamics_percent(velocity: int) -> int:
    """MusicXML ``dynamics`` yüzdesi: standart MIDI velocity 90 kabul edilir."""
    return max(0, min(400, int(round(velocity / 90 * 100))))


def _append_tempo(measure: ET.Element, tempo_bpm: float, mean_velocity: int) -> None:
    """Tempo + genel dinamik: tek yönlendirme, nota başına velocity taşınamaz."""
    direction = ET.SubElement(measure, "direction", {"placement": "above"})
    direction_type = ET.SubElement(direction, "direction-type")
    metronome = ET.SubElement(direction_type, "metronome")
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    ET.SubElement(metronome, "per-minute").text = str(int(round(tempo_bpm)))

    label = "ff"
    for limit, name in _DYNAMICS:
        if mean_velocity < limit:
            label = name
            break
    dynamics = ET.SubElement(direction_type, "dynamics")
    ET.SubElement(dynamics, label)
    ET.SubElement(
        direction,
        "sound",
        {"tempo": f"{tempo_bpm:g}", "dynamics": str(_dynamics_percent(mean_velocity))},
    )


def _clef_for(notes: Sequence[NoteEvent]) -> tuple[str, str]:
    """Sesin ortalamasına göre sol/sa anahtarı seçer."""
    if not notes:
        return "G", "2"
    pitches = sorted(note.pitch for note in notes)
    median = pitches[len(pitches) // 2]
    return ("F", "4") if median < 60 else ("G", "2")


@dataclass(frozen=True)
class _Unit:
    """Aynı anda başlayan ve aynı süren notalar: notasyonda tek akor."""

    offset: int
    duration: int
    segments: tuple[_Segment, ...]


def _units(segments: Sequence[_Segment]) -> list[_Unit]:
    """Aynı (başlangıç, süre) çiftini tek birime toplar; sıra deterministiktir."""
    grouped: dict[tuple[int, int], list[_Segment]] = {}
    for segment in segments:
        grouped.setdefault((segment.offset, segment.duration), []).append(segment)
    return [
        _Unit(offset=offset, duration=duration, segments=tuple(items))
        for (offset, duration), items in sorted(grouped.items())
    ]


def _append_voice(
    measure: ET.Element,
    units: Sequence[_Unit],
    *,
    voice: int,
    divisions: int,
    fifths: int,
    measure_ticks: int,
) -> None:
    """Tek sesi baştan sona yazar: boşluklar sus, eşzamanlı notalar akor."""
    position = 0
    for unit in units:
        if unit.offset > position:
            _append_rest(measure, unit.offset - position, divisions, voice=voice)
        for index, segment in enumerate(sorted(unit.segments, key=lambda item: -item.pitch)):
            _append_note(
                measure,
                segment,
                divisions=divisions,
                fifths=fifths,
                chord=index > 0,
                voice=voice,
            )
        position = unit.offset + unit.duration
    if position < measure_ticks:
        _append_rest(measure, measure_ticks - position, divisions, voice=voice)


def _part_element(
    part: XmlPart,
    *,
    part_id: str,
    divisions: int,
    measure_ticks: int,
    fifths: int,
    time_signature: tuple[int, int],
    tempo_bpm: float | None,
) -> ET.Element:
    """Tek partiyi (ölçüleriyle) kurar."""
    element = ET.Element("part", {"id": part_id})
    segments, last_measure = _segments(_assign_voices(part.notes), measure_ticks)

    clef = _clef_for(part.notes)
    mean_velocity = (
        int(round(sum(note.velocity for note in part.notes) / len(part.notes)))
        if part.notes
        else 90
    )

    for index in range(last_measure + 1):
        measure = ET.SubElement(element, "measure", {"number": str(index + 1)})
        if index == 0:
            _append_attributes(
                measure,
                divisions=divisions,
                fifths=fifths,
                time_signature=time_signature,
                clef=clef,
            )
            if tempo_bpm is not None:
                _append_tempo(measure, tempo_bpm, mean_velocity)

        in_measure = [segment for segment in segments if segment.measure == index]
        voices = sorted({segment.voice for segment in in_measure}) or [1]
        for order, voice in enumerate(voices, start=1):
            # Her ses tek başına ölçü uzunluğunda yazılır; aralarında geri sarma
            # (``backup``) olmadan ikinci ses ilk sesin üstüne binmiş görünürdü.
            if order > 1:
                backup = ET.SubElement(measure, "backup")
                ET.SubElement(backup, "duration").text = str(measure_ticks)
            _append_voice(
                measure,
                _units([segment for segment in in_measure if segment.voice == voice]),
                voice=voice,
                divisions=divisions,
                fifths=fifths,
                measure_ticks=measure_ticks,
            )

    return element


def notes_to_musicxml(
    parts: Sequence[XmlPart],
    *,
    title: str = DEFAULT_TITLE,
    composer: str = SOFTWARE_NAME,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
    time_signature: tuple[int, int] = DEFAULT_TIME_SIGNATURE,
    divisions: int = TICKS_PER_BEAT,
    fifths: int | None = None,
) -> str:
    """Partileri MusicXML metnine çevirir (dosyaya yazmaz)."""
    if not parts:
        raise ValueError("en az bir parti gerekli")
    if divisions <= 0:
        raise ValueError("divisions pozitif olmalı")
    beats, beat_type = time_signature
    if beats <= 0 or beat_type <= 0:
        raise ValueError("geçersiz ölçü sayısı")
    if fifths is not None and not -7 <= fifths <= 7:
        raise ValueError("fifths -7…7 aralığında olmalı")

    measure_ticks = int(round(divisions * 4 * beats / beat_type))
    signature = fifths if fifths is not None else detect_fifths(parts)

    root = ET.Element("score-partwise", {"version": MUSICXML_VERSION})
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = title

    identification = ET.SubElement(root, "identification")
    ET.SubElement(identification, "creator", {"type": "composer"}).text = composer
    encoding = ET.SubElement(identification, "encoding")
    ET.SubElement(encoding, "software").text = SOFTWARE_NAME

    part_list = ET.SubElement(root, "part-list")
    for index, part in enumerate(parts, start=1):
        score_part = ET.SubElement(part_list, "score-part", {"id": f"P{index}"})
        ET.SubElement(score_part, "part-name").text = part.name
        score_instrument = ET.SubElement(score_part, "score-instrument", {"id": f"P{index}-I1"})
        ET.SubElement(score_instrument, "instrument-name").text = part.name
        midi_instrument = ET.SubElement(score_part, "midi-instrument", {"id": f"P{index}-I1"})
        # MusicXML kanal/program numaraları 1 tabanlıdır.
        ET.SubElement(midi_instrument, "midi-channel").text = str(part.channel + 1)
        ET.SubElement(midi_instrument, "midi-program").text = str(part.program + 1)

    for index, part in enumerate(parts, start=1):
        root.append(
            _part_element(
                part,
                part_id=f"P{index}",
                divisions=divisions,
                measure_ticks=measure_ticks,
                fifths=signature,
                time_signature=time_signature,
                tempo_bpm=tempo_bpm if index == 1 else None,
            )
        )

    ET.indent(root)
    return DOCTYPE + ET.tostring(root, encoding="unicode") + "\n"


def write_musicxml(parts: Sequence[XmlPart], path: str | Path, **options: object) -> str:
    """MusicXML dosyasını yazar; yolunu döndürür."""
    text = notes_to_musicxml(parts, **options)  # type: ignore[arg-type]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return str(target)


def parts_from_arrangement(parts: Sequence[Part]) -> list[XmlPart]:
    """``audio.Part`` listesini yazılabilir partilere çevirir."""
    return [
        XmlPart(
            name=part.instrument.name,
            notes=list(part.notes),
            program=part.instrument.gm_program,
            channel=part.channel,
        )
        for part in parts
    ]


def parts_from_midi(path: str | Path) -> list[XmlPart]:
    """Çok kanallı MIDI'yi parti listesine çevirir (iz adı → enstrüman adı)."""
    midi = load_midi(path)
    notes = iter_notes(midi)

    grouped: dict[tuple[int, int], list[NoteEvent]] = {}
    for note in notes:
        grouped.setdefault((note.track, note.channel), []).append(note)

    parts: list[XmlPart] = []
    for (track_index, channel), group in sorted(grouped.items()):
        track_name = midi.tracks[track_index].name.strip() if track_index < len(midi.tracks) else ""
        instrument = INSTRUMENTS.get(track_name)
        parts.append(
            XmlPart(
                name=instrument.name if instrument else (track_name or f"Kanal {channel + 1}"),
                notes=group,
                program=group[0].program,
                channel=channel,
            )
        )
    if not parts:
        raise ValueError(f"MIDI'de nota yok: {path}")
    return parts


def musicxml_from_midi(
    midi_path: str | Path,
    out_path: str | Path | None = None,
    **options: object,
) -> str:
    """MIDI dosyasını MusicXML'e çevirir (varsayılan hedef: aynı ad, .musicxml)."""
    target = Path(out_path) if out_path is not None else Path(midi_path).with_suffix(".musicxml")
    return write_musicxml(parts_from_midi(midi_path), target, **options)
