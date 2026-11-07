"""Komut → istek → tohum → üretim → düzenleme → MIDI.

Bu modül 6. aşamanın omurgası: doğal dil ile model arasındaki tüm adımları tek
bir yerde toplar. Her adım kendi başına test edilebilir; burada yalnızca sıra
ve aktarım var.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from ophelia.audio.ensemble import (
    Part,
    arrange,
    arrange_by_role,
    chorale_ensemble,
    write_multitrack_midi,
)
from ophelia.audio.render import render_file
from ophelia.data.midi_io import MidiStats, NoteEvent, analyze, write_midi
from ophelia.data.tokens import BOS, EOS, MidiTokenizer
from ophelia.model.config import SampleConfig
from ophelia.model.generate import generate_tokens, limit_polyphony
from ophelia.model.transformer import OpheliaGPT
from ophelia.nlp.arrange import (
    AdjustReport,
    clip_to_duration,
    estimated_key,
    project_to_scale,
    transpose_to_key,
)
from ophelia.nlp.metrify import metrify
from ophelia.nlp.parse import parse_command
from ophelia.nlp.seeding import seed_notes
from ophelia.nlp.spec import CompositionSpec
from ophelia.notation.musicxml import (
    DEFAULT_TITLE,
    XmlPart,
    parts_from_arrangement,
    write_musicxml,
)

# Veri setinden ölçülen token yoğunluğu: 393.839 token / 16.416 sn ≈ 24 token/sn.
TOKENS_PER_SECOND = 24.0
# Koral yoğunluğu ≈ 6 nota/sn; asgari uzunluk için ölçülü bir alt sınır.
NOTES_PER_SECOND = 3.0


@dataclass
class ComposeResult:
    """Üretim + düzenleme sonucu; rapor için her ara değer saklanır."""

    spec: CompositionSpec
    notes: list[NoteEvent] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    stopped_at_eos: bool = False
    adjust: AdjustReport = field(default_factory=AdjustReport)
    midi_path: Path | None = None
    wav_path: Path | None = None
    xml_path: Path | None = None
    stats: MidiStats | None = None
    final_key: object | None = None
    parts: list[Part] = field(default_factory=list)

    @property
    def n_notes(self) -> int:
        return len(self.notes)

    @property
    def duration_seconds(self) -> float:
        return self.stats.duration_seconds if self.stats else 0.0


def tokens_for(seconds: float) -> int:
    """İstenen süreye yetecek token bütçesi (EOS erken bitirebilir)."""
    return int(min(4096, max(64, seconds * TOKENS_PER_SECOND + 200)))


def min_notes_for(seconds: float) -> int:
    """Bu sürede beklenen asgari nota sayısı (EOS için alt sınır)."""
    return int(max(8, seconds * NOTES_PER_SECOND))


def _stream_ids(tokenizer: MidiTokenizer, tokens: list[str]) -> list[int]:
    """Tohum + üretilmiş token'ları yeniden istem olarak kullanılacak kimliklere çevirir."""
    return tokenizer.vocabulary.to_ids([token for token in tokens if token != EOS])


def generate_until(
    model: OpheliaGPT,
    tokenizer: MidiTokenizer,
    spec: CompositionSpec,
    *,
    sample: SampleConfig,
    device: str = "cpu",
    max_passes: int = 6,
) -> tuple[list[str], bool, int]:
    """İstenen süre dolana kadar üretir; EOS bir **cümle sonu**, eser sonu değil.

    Model 5-15 saniyede "bitebilirim" diyor; süre istendiyse sonu ``EOS``'suz
    bırakıp akışın devamını üretiyoruz. Dönen: (token'lar, EOS'a ulaşıldı mı,
    geçiş sayısı).
    """
    target_ticks = spec.duration_seconds * spec.tempo_bpm / 60.0 * 480
    tokens = tokenizer.encode_notes(seed_notes(spec, grid=tokenizer.config.grid_ticks))
    tokens = [token for token in tokens if token not in (BOS, EOS)]
    stopped = False
    passes = 0

    for _ in range(max_passes):
        passes += 1
        prompt = [tokenizer.vocabulary.id_of(BOS)] + _stream_ids(tokenizer, tokens)
        produced, stopped = generate_tokens(
            model, tokenizer, sample, device=device, prompt_tokens=prompt
        )
        fresh = produced[1:] if produced and produced[0] == BOS else produced
        if fresh and fresh[-1] == EOS:
            fresh = fresh[:-1]
        if not fresh:
            break
        tokens = tokens + fresh

        notes = tokenizer.decode_notes(tokens, strict=False)
        end = max((note.end for note in notes), default=0)
        if end >= target_ticks:
            break
        # ``stopped`` yanlışsa model token bütçesini doldurmuş ama eser hâlâ
        # kısa: döngü devam eder (en fazla ``max_passes``).

    return tokens, stopped, passes


def compose(
    command: str,
    model: OpheliaGPT,
    tokenizer: MidiTokenizer,
    *,
    out_path: str | Path | None = None,
    wav_path: str | Path | None = None,
    xml_path: str | Path | None = None,
    seed: int | None = None,
    temperature: float = 0.95,
    top_k: int = 40,
    spec: CompositionSpec | None = None,
    device: str = "cpu",
    min_duration_steps: int = 2,
    time_step_steps: int = 2,
) -> ComposeResult:
    """Komutu çözer, eseri üretir, isteğe oturtur ve MIDI yazar.

    ``min_duration_steps`` / ``time_step_steps`` ızgara birimidir (1 = 16'lık):
    varsayılan 2, yani en kısa nota 8'lik ve vuruşlar 8'lik ızgarada. Gerçek
    korallerde notaların %3'ü 16'lıktır; kısıtsız model %30 üretiyor ve bu,
    "oyun müziği" etkisinin başlıca sebebiydi.
    """
    request = spec or parse_command(command, seed=seed)

    sample = SampleConfig(
        max_new_tokens=tokens_for(request.duration_seconds),
        temperature=temperature,
        top_k=top_k,
        seed=request.seed,
        max_polyphony=request.max_polyphony,
        min_notes=min_notes_for(request.duration_seconds),
        constrained=True,
        min_duration_steps=min_duration_steps,
        time_step_steps=time_step_steps,
    )

    raw_tokens, stopped, passes = generate_until(
        model, tokenizer, request, sample=sample, device=device
    )
    notes = tokenizer.decode_notes(raw_tokens, strict=False)

    adjusted, report = transpose_to_key(notes, request.key)
    projected, off_before, off_after = project_to_scale(adjusted, request.key)
    clipped, _ = clip_to_duration(
        projected, request.duration_seconds, tempo_bpm=request.tempo_bpm
    )
    final = limit_polyphony(clipped, request.max_polyphony)
    if min_duration_steps or time_step_steps:
        grid = tokenizer.config.grid_ticks * max(1, time_step_steps)
        final = metrify(
            final,
            grid=grid,
            min_duration=tokenizer.config.grid_ticks * max(1, min_duration_steps),
        )
    report = replace(
        report,
        notes_before=report.notes_before or len(notes),
        notes_after=len(final),
        clipped_at_seconds=request.duration_seconds,
        off_scale_before=off_before,
        off_scale_after=off_after,
    )

    result = ComposeResult(
        spec=request,
        notes=final,
        tokens=raw_tokens,
        stopped_at_eos=stopped,
        adjust=report,
        final_key=estimated_key(final),
    )

    # WAV isteniyorsa enstrüman verilmese bile düzenleme yapılır ve **MIDI de
    # çok kanallı** yazılır: aksi hâlde .mid tek kanallı, .wav dört enstrümanlı
    # olurdu — iki dosya aynı eseri anlatmazdı.
    if request.roles:
        # Roller açıkça istendiyse düzenleme perde bölgesine değil işleve göre.
        result.parts = arrange_by_role(final, request.roles, request.ensemble)
    else:
        ensemble = request.ensemble or (chorale_ensemble() if wav_path else ())
        if ensemble:
            result.parts = arrange(final, ensemble)

    if out_path is not None:
        target = Path(out_path)
        if result.parts:
            write_multitrack_midi(
                result.parts, str(target), tempo_bpm=request.tempo_bpm
            )
        else:
            write_midi(final, target, tempo_bpm=request.tempo_bpm)
        result.midi_path = target
        result.stats = analyze(target)

    if wav_path is not None and result.parts:
        result.wav_path = Path(render_file(result.parts, str(wav_path), tempo_bpm=request.tempo_bpm))

    # Nota düzenlemesi: düzenlenmiş partiler varsa seslendirilen hâli yazılır,
    # yoksa tek partili (model çıktısı) nota kâğıdı.
    if xml_path is not None:
        xml_parts = parts_from_arrangement(result.parts) or [XmlPart("Ophelia", list(final))]
        result.xml_path = Path(
            write_musicxml(
                xml_parts,
                xml_path,
                title=command.strip()[:80] or DEFAULT_TITLE,
                tempo_bpm=request.tempo_bpm,
            )
        )

    return result


def load_models(checkpoint: str | Path) -> tuple[OpheliaGPT, MidiTokenizer]:
    """Checkpoint + sözlük yükler (CLI ve betikler için tek giriş noktası)."""
    from ophelia.model.checkpoint import load_checkpoint, load_tokenizer

    model, info = load_checkpoint(checkpoint)
    tokenizer = load_tokenizer(checkpoint)
    if tokenizer.vocabulary.size != info.model_config.vocab_size:
        raise ValueError(
            f"sözlük boyutu uyuşmuyor: {tokenizer.vocabulary.size} ≠ "
            f"{info.model_config.vocab_size}"
        )
    model.eval()
    return model, tokenizer


def device_for() -> str:
    """CUDA varsa ona, yoksa CPU'ya (CPU-only makinede davranış değişmez)."""
    return "cuda" if torch.cuda.is_available() else "cpu"
