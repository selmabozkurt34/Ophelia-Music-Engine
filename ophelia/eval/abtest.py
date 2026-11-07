"""Kör A/B dinleme testi: gerçek koral ↔ model üretimi.

Neden gerekli: piyasa raporunda "hiç ölçmediğimiz tek kritik boyut" diye
işaretlenen şey buydu. Modelin müzikal kalitesine dair elimizde yalnızca
doğrulama kaybı ve birkaç izlenim vardı; dinleyici tercihi ölçülmemişti.

Protokol (kör, iki alternatifli zorunlu seçim — 2AFC):

* Her denemede iki kısa kayıt var: biri **gerçek Bach korali**, biri **model
  üretimi**. İkisi de aynı sentezleyiciden, aynı topluluktan ve aynı tepe
  normalizasyonundan geçer; böylece dinleyici timbre'ı değil müziği puanlar.
* Kayıtlar aynı uzunluğa getirilir (kırpılır/doldurulur): süre farkı bir ipucu
  olmasın.
* Sıra ve A/B tarafı dinleyici başına rastgele; dosya adları opak
  (``t03_a.wav``). Etiketler yalnızca ``key.json`` içinde durur ve tarayıcıya
  **gitmez**.
* Gerçek örnekler **test bölütünden** seçilir: model bu koralleri eğitimde
  görmedi; aksi hâlde ezber ölçülürdü.
* Dinleyici önce zorunlu tercihini (A / B / eşit), sonra "hangisi insan
  tarafından yazılmış?" tahminini ve 1-5 güvenini verir.

Ölçüt: gerçek lehine tercih oranı ve **Wilson %95 güven aralığı**. Aralık
%50'yi içeriyorsa sonuç "ayırt edilemedi" demektir — bu bir başarısızlık değil,
dürüst bir ölçümdür.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ophelia.audio.ensemble import chorale_ensemble
from ophelia.audio.render import SAMPLE_RATE, MixSettings, render_parts
from ophelia.audio.render import write_wav as write_stereo_wav
from ophelia.data.midi_io import iter_notes, load_midi
from ophelia.nlp.arrange import clip_to_duration

if TYPE_CHECKING:  # pragma: no cover - yalnızca tip denetimi
    from collections.abc import Iterable, Sequence

KEY_FILE = "key.json"
TRIALS_FILE = "trials.json"
VOTES_FILE = "votes.jsonl"
PAGE_FILE = "index.html"
AUDIO_DIR = "audio"

DEFAULT_SECONDS = 20.0
DEFAULT_TRIALS = 10
DEFAULT_SEED = 0
FADE_SECONDS = 0.15
SIDES = ("a", "b")
PREFERENCES = ("a", "b", "tie")


class AbTestError(ValueError):
    """Test kurulumu ya da oy geçersiz."""


# --- Yardımcılar -----------------------------------------------------------


def tempo_of(path: str | Path, default: float = 100.0) -> float:
    """MIDI'deki ilk tempo işaretini BPM'e çevirir."""
    import mido

    for track in mido.MidiFile(str(path)).tracks:
        for message in track:
            if message.type == "set_tempo":
                return 60_000_000 / message.tempo
    return default


def _mix_settings() -> MixSettings:
    return MixSettings()


def render_clip(
    midi_path: str | Path,
    *,
    seconds: float = DEFAULT_SECONDS,
    ensemble: Sequence[str] | None = None,
) -> np.ndarray:
    """MIDI'yi sabit uzunlukta, tepe normalizasyonlu stereo diziye çevirir.

    Uzunluk **her iki kaynak için de** aynıdır (kırp ya da sessizlikle doldur):
    süre farkı dinleyiciye ipucu vermemeli. Uçta kısa bir sönüm uygulanır ki
    kırpma noktası cızırtı üretmesin.
    """
    if seconds <= 0:
        raise AbTestError("süre pozitif olmalı")
    notes = list(iter_notes(load_midi(midi_path)))
    if not notes:
        raise AbTestError(f"nota yok: {midi_path}")

    tempo = tempo_of(midi_path)
    clipped, _ = clip_to_duration(notes, seconds, tempo_bpm=tempo)
    parts = _arrange(clipped, ensemble)
    samples = render_parts(parts, tempo_bpm=tempo, settings=_mix_settings())

    target = int(round(seconds * SAMPLE_RATE))
    if samples.shape[1] > target:
        samples = samples[:, :target]
    elif samples.shape[1] < target:
        padded = np.zeros((2, target), dtype=np.float32)
        padded[:, : samples.shape[1]] = samples
        samples = padded

    fade = min(int(FADE_SECONDS * SAMPLE_RATE), samples.shape[1] // 4)
    if fade > 1:
        samples[:, -fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    peak = float(np.max(np.abs(samples))) or 1.0
    settings = _mix_settings()
    return (samples / peak * settings.peak * settings.master_gain).astype(np.float32)


def _arrange(notes: list, ensemble: Sequence[str] | None) -> list:
    from ophelia.audio.ensemble import arrange

    return arrange(notes, tuple(ensemble or chorale_ensemble()))


# --- Oturum ----------------------------------------------------------------


@dataclass(frozen=True)
class Trial:
    """Tek bir kör karşılaştırma."""

    id: str
    a: str  # ses klasöründeki dosya adı
    b: str
    real_side: str  # "a" | "b" — YALNIZ sunucu tarafında kalır
    real_source: str
    generated_source: str

    def __post_init__(self) -> None:
        if self.real_side not in SIDES:
            raise AbTestError(f"geçersiz taraf: {self.real_side!r}")


@dataclass(frozen=True)
class Session:
    """Bir test oturumu: denemeler + etiket anahtarı + klasör."""

    session_id: str
    directory: Path
    trials: tuple[Trial, ...]
    seconds: float = DEFAULT_SECONDS
    seed: int = DEFAULT_SEED
    title: str = "Ophelia kör A/B dinleme testi"

    @property
    def audio_dir(self) -> Path:
        return self.directory / AUDIO_DIR

    @property
    def votes_path(self) -> Path:
        return self.directory / VOTES_FILE

    def trial(self, trial_id: str) -> Trial:
        for item in self.trials:
            if item.id == trial_id:
                return item
        raise AbTestError(f"bilinmeyen deneme: {trial_id!r}")

    def public_trials(self) -> list[dict[str, str]]:
        """Tarayıcıya giden liste — kaynak etiketi içermez."""
        return [
            {"id": trial.id, "a": f"{AUDIO_DIR}/{trial.a}", "b": f"{AUDIO_DIR}/{trial.b}"}
            for trial in self.trials
        ]

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.session_id,
            "seconds": self.seconds,
            "seed": self.seed,
            "title": self.title,
            "created": _now(),
            "trials": [
                {
                    "id": trial.id,
                    "a": trial.a,
                    "b": trial.b,
                    "real_side": trial.real_side,
                    "real_source": trial.real_source,
                    "generated_source": trial.generated_source,
                }
                for trial in self.trials
            ],
        }
        (self.directory / KEY_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (self.directory / TRIALS_FILE).write_text(
            json.dumps(self.public_trials(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        page = Path(__file__).with_name("listener.html")
        shutil.copyfile(page, self.directory / PAGE_FILE)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def load_session(directory: str | Path) -> Session:
    """Kaydedilmiş oturumu (etiket anahtarıyla) okur."""
    folder = Path(directory)
    key_path = folder / KEY_FILE
    if not key_path.is_file():
        raise AbTestError(f"oturum bulunamadı (key.json yok): {folder}")
    payload = json.loads(key_path.read_text(encoding="utf-8"))
    try:
        trials = tuple(
            Trial(
                id=item["id"],
                a=item["a"],
                b=item["b"],
                real_side=item["real_side"],
                real_source=item["real_source"],
                generated_source=item["generated_source"],
            )
            for item in payload["trials"]
        )
    except (KeyError, TypeError) as exc:
        raise AbTestError(f"bozuk key.json: {exc}") from exc
    return Session(
        session_id=str(payload.get("session_id", folder.name)),
        directory=folder,
        trials=trials,
        seconds=float(payload.get("seconds", DEFAULT_SECONDS)),
        seed=int(payload.get("seed", DEFAULT_SEED)),
        title=str(payload.get("title", "Ophelia kör A/B dinleme testi")),
    )


def build_session(
    real: Iterable[str | Path],
    generated: Iterable[str | Path],
    directory: str | Path,
    *,
    trials: int = DEFAULT_TRIALS,
    seconds: float = DEFAULT_SECONDS,
    seed: int = DEFAULT_SEED,
    title: str = "Ophelia kör A/B dinleme testi",
    ensemble: Sequence[str] | None = None,
) -> Session:
    """Gerçek ve üretilmiş MIDI'lerden kör bir test oturumu kurar.

    Aynı eser iki kez kullanılmaz; havuzlar ``trials`` kadar dosya içermelidir.
    """
    if trials < 1:
        raise AbTestError("en az bir deneme gerekli")
    real_paths = _existing(real)
    generated_paths = _existing(generated)
    if len(real_paths) < trials:
        raise AbTestError(
            f"gerçek havuz yetersiz: {len(real_paths)} dosya var, {trials} gerekli"
        )
    if len(generated_paths) < trials:
        raise AbTestError(
            f"üretim havuzu yetersiz: {len(generated_paths)} dosya var, {trials} gerekli "
            "(önce: ophelia compose/ model generate)"
        )

    rng = random.Random(seed)
    picked_real = rng.sample(real_paths, trials)
    picked_generated = rng.sample(generated_paths, trials)
    rng.shuffle(picked_real)
    rng.shuffle(picked_generated)

    folder = Path(directory)
    audio = folder / AUDIO_DIR
    if audio.exists():
        shutil.rmtree(audio)
    audio.mkdir(parents=True, exist_ok=True)

    entries: list[Trial] = []
    for index, (real_midi, generated_midi) in enumerate(
        zip(picked_real, picked_generated, strict=True), start=1
    ):
        trial_id = f"t{index:02d}"
        real_is_a = rng.random() < 0.5
        real_name = f"{trial_id}_{'a' if real_is_a else 'b'}.wav"
        generated_name = f"{trial_id}_{'b' if real_is_a else 'a'}.wav"

        write_stereo_wav(
            render_clip(real_midi, seconds=seconds, ensemble=ensemble),
            str(audio / real_name),
            sample_rate=SAMPLE_RATE,
        )
        write_stereo_wav(
            render_clip(generated_midi, seconds=seconds, ensemble=ensemble),
            str(audio / generated_name),
            sample_rate=SAMPLE_RATE,
        )
        entries.append(
            Trial(
                id=trial_id,
                a=real_name if real_is_a else generated_name,
                b=generated_name if real_is_a else real_name,
                real_side="a" if real_is_a else "b",
                real_source=Path(real_midi).name,
                generated_source=Path(generated_midi).name,
            )
        )

    session = Session(
        session_id=f"abtest-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        directory=folder,
        trials=tuple(entries),
        seconds=seconds,
        seed=seed,
        title=title,
    )
    session.save()
    return session


def _existing(paths: Iterable[str | Path]) -> list[Path]:
    seen: set[Path] = set()
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path not in seen:
            seen.add(path)
            found.append(path)
    return found


def default_real_pool(project_root: str | Path) -> list[Path]:
    """Gerçek havuz: **test bölütü** (model bu dosyaları eğitimde görmedi)."""
    root = Path(project_root)
    processed = root / "data" / "processed" / "music21-bach-chorales"
    splits = root / "data" / "splits" / "splits.json"
    if splits.is_file():
        payload = json.loads(splits.read_text(encoding="utf-8"))
        stems = payload.get("splits", {}).get("test", [])
        paths = [processed / f"{stem}.mid" for stem in stems]
        found = [path for path in paths if path.is_file()]
        if found:
            return found
    return sorted(processed.glob("*.mid"))


def default_generated_pool(project_root: str | Path) -> list[Path]:
    """Üretim havuzu: ``outputs/midi`` altındaki MIDI'ler."""
    return sorted((Path(project_root) / "outputs" / "midi").glob("*.mid"))


# --- Oylar -----------------------------------------------------------------


@dataclass(frozen=True)
class Vote:
    """Tek dinleyicinin tek denemeye verdiği yanıt."""

    trial_id: str
    preference: str  # "a" | "b" | "tie"
    listener: str
    human_guess: str | None = None  # "a" | "b" | None (bilmiyorum)
    confidence: int | None = None
    elapsed_ms: int | None = None
    received: str | None = None

    @classmethod
    def from_payload(cls, payload: dict, *, known_trials: Iterable[str] = ()) -> Vote:
        """Tarayıcıdan gelen JSON'u doğrular; geçersizse :class:`AbTestError`."""
        trial_id = str(payload.get("trial", "")).strip()
        known = set(known_trials)
        if known and trial_id not in known:
            raise AbTestError(f"bilinmeyen deneme: {trial_id!r}")

        preference = str(payload.get("preference", "")).strip().lower()
        if preference not in PREFERENCES:
            raise AbTestError("tercih 'a', 'b' ya da 'tie' olmalı")

        listener = str(payload.get("listener", "")).strip()
        if not listener:
            raise AbTestError("dinleyici kimliği boş")
        if len(listener) > 64:
            raise AbTestError("dinleyici kimliği çok uzun")

        guess = payload.get("human_guess")
        guess = str(guess).strip().lower() if guess not in (None, "") else None
        if guess in {"none", "bilmiyorum", "unknown"}:
            guess = None
        if guess is not None and guess not in SIDES:
            raise AbTestError("tahmin 'a', 'b' ya da boş olmalı")

        confidence = payload.get("confidence")
        if confidence in (None, ""):
            confidence = None
        else:
            try:
                confidence = int(confidence)
            except (TypeError, ValueError) as exc:
                raise AbTestError("güven 1-5 arası tam sayı olmalı") from exc
            if not 1 <= confidence <= 5:
                raise AbTestError("güven 1-5 arası olmalı")

        elapsed = payload.get("elapsed_ms")
        if elapsed not in (None, ""):
            try:
                elapsed = max(0, int(elapsed))
            except (TypeError, ValueError):
                elapsed = None
        else:
            elapsed = None

        return cls(
            trial_id=trial_id,
            preference=preference,
            listener=listener,
            human_guess=guess,
            confidence=confidence,
            elapsed_ms=elapsed,
            received=_now(),
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "trial": self.trial_id,
                "preference": self.preference,
                "listener": self.listener,
                "human_guess": self.human_guess,
                "confidence": self.confidence,
                "elapsed_ms": self.elapsed_ms,
                "received": self.received,
            },
            ensure_ascii=False,
        )


def load_votes(paths: Iterable[str | Path]) -> tuple[list[Vote], int]:
    """JSONL oy dosyalarını okur; (oylar, bozuk satır sayısı) döner."""
    votes: list[Vote] = []
    broken = 0
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            nested, inner = load_votes(sorted(path.glob("*.jsonl")))
            votes.extend(nested)
            broken += inner
            continue
        if not path.is_file():
            raise AbTestError(f"oy dosyası yok: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                votes.append(Vote.from_payload(payload))
            except (json.JSONDecodeError, AbTestError):
                broken += 1
    return votes, broken


# --- İstatistik ------------------------------------------------------------


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Oran için Wilson skor güven aralığı (küçük örneklerde normal yaklaşımdan iyi)."""
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def binomial_p_value(successes: int, total: int) -> float:
    """İki yönlü kesin binom testi (H0: p = 0.5)."""
    if total <= 0:
        return 1.0
    from scipy.stats import binomtest

    return float(binomtest(successes, total, 0.5).pvalue)


def required_votes(margin: float = 0.1, z: float = 1.96, p: float = 0.5) -> int:
    """±``margin`` kesinlik için gereken oy sayısı (en kötü durum p = 0.5)."""
    if not 0 < margin < 1:
        raise AbTestError("marj 0 ile 1 arasında olmalı")
    return math.ceil(z * z * p * (1 - p) / (margin * margin))


# --- Rapor -----------------------------------------------------------------


@dataclass
class ListenerReport:
    listener: str
    votes: int = 0
    real_choice: int = 0
    generated_choice: int = 0
    ties: int = 0
    correct_guesses: int = 0

    @property
    def rate(self) -> float | None:
        decisive = self.real_choice + self.generated_choice
        return self.real_choice / decisive if decisive else None


@dataclass
class TrialReport:
    trial_id: str
    votes: int = 0
    real_choice: int = 0
    generated_choice: int = 0
    ties: int = 0
    correct_guesses: int = 0
    guesses: int = 0


@dataclass
class Report:
    """Test sonucunun tamamı: oranlar, aralıklar, kırılımlar."""

    session: Session
    votes: int
    real_choice: int
    generated_choice: int
    ties: int
    rate_real: float | None
    ci: tuple[float, float] | None
    p_value: float | None
    identification_n: int
    identification_correct: int
    identification_rate: float | None
    identification_ci: tuple[float, float] | None
    identification_p: float | None
    mean_confidence: float | None
    mean_seconds: float | None
    listeners: list[ListenerReport] = field(default_factory=list)
    per_trial: list[TrialReport] = field(default_factory=list)
    broken_lines: int = 0
    duplicate_votes: int = 0

    @property
    def decisive(self) -> int:
        return self.real_choice + self.generated_choice

    @property
    def verdict(self) -> str:
        """Tek cümlelik yorum: %95 aralığı %50'yi içeriyorsa ayırt edilemedi."""
        if not self.ci or self.decisive == 0:
            return "yeterli karar yok"
        low, high = self.ci
        if low <= 0.5 <= high:
            return "gerçek ile üretim ayırt edilemedi"
        if low > 0.5:
            return "gerçek koral anlamlı biçimde tercih edildi"
        return "model üretimi anlamlı biçimde tercih edildi"

    def identification_verdict(self) -> str:
        if not self.identification_ci or self.identification_n == 0:
            return "yeterli tahmin yok"
        low, high = self.identification_ci
        if low <= 0.5 <= high:
            return "insan/üretim ayrımı şans düzeyinde"
        if low > 0.5:
            return "dinleyiciler gerçek korali şanstan iyi tanıdı"
        return "dinleyiciler sistematik olarak ters seçti"


def summarize(session: Session, votes: Sequence[Vote], *, broken_lines: int = 0) -> Report:
    """Oyları oturumun etiket anahtarıyla birleştirip raporu üretir."""
    known = {trial.id for trial in session.trials}
    real_choice = generated_choice = ties = 0
    identification_n = identification_correct = 0
    confidences: list[float] = []
    times: list[float] = []
    listeners: dict[str, ListenerReport] = {}
    per_trial: dict[str, TrialReport] = {
        trial.id: TrialReport(trial_id=trial.id) for trial in session.trials
    }
    # Bir dinleyicinin aynı denemeye verdiği ikinci oy sayılmaz: aksi hâlde
    # aynı kişi testi tekrar ederek örneklemi kendi lehine şişirir (pseudo-replication).
    seen: set[tuple[str, str]] = set()
    duplicates = 0

    for vote in votes:
        if vote.trial_id not in known:
            continue
        key = (vote.listener, vote.trial_id)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        trial = session.trial(vote.trial_id)
        who = listeners.setdefault(vote.listener, ListenerReport(listener=vote.listener))
        detail = per_trial[vote.trial_id]
        who.votes += 1
        detail.votes += 1

        if vote.preference == "tie":
            ties += 1
            who.ties += 1
            detail.ties += 1
        elif vote.preference == trial.real_side:
            real_choice += 1
            who.real_choice += 1
            detail.real_choice += 1
        else:
            generated_choice += 1
            who.generated_choice += 1
            detail.generated_choice += 1

        if vote.human_guess is not None and vote.human_guess in SIDES:
            identification_n += 1
            detail.guesses += 1
            if vote.human_guess == trial.real_side:
                identification_correct += 1
                who.correct_guesses += 1
                detail.correct_guesses += 1

        if vote.confidence is not None:
            confidences.append(float(vote.confidence))
        if vote.elapsed_ms is not None:
            times.append(vote.elapsed_ms / 1000.0)

    decisive = real_choice + generated_choice
    return Report(
        session=session,
        votes=real_choice + generated_choice + ties,
        real_choice=real_choice,
        generated_choice=generated_choice,
        ties=ties,
        rate_real=(real_choice / decisive) if decisive else None,
        ci=wilson_interval(real_choice, decisive) if decisive else None,
        p_value=binomial_p_value(real_choice, decisive) if decisive else None,
        identification_n=identification_n,
        identification_correct=identification_correct,
        identification_rate=(identification_correct / identification_n) if identification_n else None,
        identification_ci=(
            wilson_interval(identification_correct, identification_n) if identification_n else None
        ),
        identification_p=(
            binomial_p_value(identification_correct, identification_n) if identification_n else None
        ),
        mean_confidence=(sum(confidences) / len(confidences)) if confidences else None,
        mean_seconds=(sum(times) / len(times)) if times else None,
        listeners=sorted(listeners.values(), key=lambda item: item.listener),
        per_trial=[per_trial[trial.id] for trial in session.trials],
        broken_lines=broken_lines,
        duplicate_votes=duplicates,
    )
