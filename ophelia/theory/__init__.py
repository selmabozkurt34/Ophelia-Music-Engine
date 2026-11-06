"""2. aşama — Müzik teorisi altyapısı.

Modelin öğrenmesi ve kullanması gereken yapıların tek kaynağı:

* :mod:`ophelia.theory.pitch` — nota, oktav, MIDI, frekans, enharmonik
* :mod:`ophelia.theory.chords` — akor yapıları, semboller, akor tanıma
* :mod:`ophelia.theory.scales` — gamlar, modlar, tonalite, diyatonik akorlar
* :mod:`ophelia.theory.harmony` — akor geçişleri, kadanslar, ses yürütme
* :mod:`ophelia.theory.rhythm` — tempo, ölçü, alt bölünme, senkop
* :mod:`ophelia.theory.form` — motif dönüşümleri, cümle, form şablonları
* :mod:`ophelia.theory.instruments` — enstrüman rolleri ve MIDI programları
"""

from ophelia.theory.chords import (
    CHORD_INTERVALS,
    Chord,
    chord_from_notes,
    identify_chord,
)
from ophelia.theory.form import (
    FORM_TEMPLATES,
    Form,
    Motif,
    Section,
    build_form,
    transform,
)
from ophelia.theory.harmony import (
    PROGRESSIONS,
    detect_cadence,
    function_of_degree,
    progression,
    progression_romans,
    voice_lead,
)
from ophelia.theory.instruments import (
    INSTRUMENTS,
    Instrument,
    assign_channels,
    by_role,
    get,
)
from ophelia.theory.pitch import (
    Pitch,
    enharmonic,
    midi_to_freq,
    midi_to_note,
    note_to_midi,
    note_to_pitch_class,
    transpose,
)
from ophelia.theory.rhythm import (
    TimedNote,
    TimeSignature,
    is_syncopated,
    quantize,
    swing,
)
from ophelia.theory.scales import (
    MODES,
    Key,
    Scale,
    estimate_key,
    roman_numeral,
    scale_pitch_classes,
)

__all__ = [
    # pitch
    "Pitch",
    "enharmonic",
    "midi_to_freq",
    "midi_to_note",
    "note_to_midi",
    "note_to_pitch_class",
    "transpose",
    # chords
    "CHORD_INTERVALS",
    "Chord",
    "chord_from_notes",
    "identify_chord",
    # scales
    "MODES",
    "Key",
    "Scale",
    "estimate_key",
    "roman_numeral",
    "scale_pitch_classes",
    # harmony
    "PROGRESSIONS",
    "detect_cadence",
    "function_of_degree",
    "progression",
    "progression_romans",
    "voice_lead",
    # rhythm
    "TimeSignature",
    "TimedNote",
    "is_syncopated",
    "quantize",
    "swing",
    # form
    "FORM_TEMPLATES",
    "Form",
    "Motif",
    "Section",
    "build_form",
    "transform",
    # instruments
    "INSTRUMENTS",
    "Instrument",
    "assign_channels",
    "by_role",
    "get",
]
