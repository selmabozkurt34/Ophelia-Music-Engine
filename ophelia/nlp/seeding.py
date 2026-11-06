"""İstenen tonalitede bir **tohum** motif üretir.

Model koşulsuz eğitildi: "la minör yaz" diyemez. Ama bir eserin ilk notalarını
vermek ona hem tonaliteyi hem de tempoyu sezdirir — dil modellerindeki
"prompt" mantığının müzik karşılığı. Tohum, istenen toniğin triadıdır
(kök–üçlü–beşli), 4/4'te dörtlük notalar hâlinde.

Tohum yalnızca bir **eğilim** yaratır; garanti değildir. Kesin garanti, üretim
sonrası tonalite düzeltmesinde (``arrange.transpose_to_key``) verilir.
"""

from __future__ import annotations

from ophelia.data.midi_io import NoteEvent
from ophelia.data.tokens import BOS, EOS, MidiTokenizer
from ophelia.nlp.spec import CompositionSpec
from ophelia.theory.scales import Key

# Tohum perdesinin yerleşeceği rahat aralık (korallerin medyanı ~C4).
_SEED_LOW = 55
_SEED_HIGH = 66


def _root_pitch(key: Key) -> int:
    for octave in range(2, 7):
        pitch = 12 * (octave + 1) + key.tonic_pc
        if _SEED_LOW <= pitch <= _SEED_HIGH:
            return pitch
    return 60  # pragma: no cover - aralık her zaman bulunur


def seed_notes(spec: CompositionSpec, *, grid: int = 120) -> list[NoteEvent]:
    """İstenen tonalitenin tonik triadını dörtlük notalar olarak döndürür."""
    key = spec.key or Key(0, "major")
    root = _root_pitch(key)
    third = 3 if key.mode in {"natural_minor", "harmonic_minor", "melodic_minor"} else 4
    pitches = (root, root + third, root + 7, root)

    return [
        NoteEvent(
            start=index * grid,
            duration=grid,
            pitch=pitch,
            velocity=96,
            channel=0,
            program=0,
            track=0,
        )
        for index, pitch in enumerate(pitches)
    ]


def seed_tokens(tokenizer: MidiTokenizer, spec: CompositionSpec) -> list[str]:
    """Tohum motifin token karşılığı (``<bos>``/``<eos>`` olmadan)."""
    tokens = tokenizer.encode_notes(seed_notes(spec, grid=tokenizer.config.grid_ticks))
    return [token for token in tokens if token not in (BOS, EOS)]


def prompt_ids(tokenizer: MidiTokenizer, spec: CompositionSpec) -> list[int]:
    """Üretime verilecek başlangıç kimlikleri: ``<bos>`` + tohum."""
    vocabulary = tokenizer.vocabulary
    return [vocabulary.id_of(BOS)] + vocabulary.to_ids(seed_tokens(tokenizer, spec))
