"""Model ve eğitim yapılandırması.

Boyutlar bu makinenin kısıtlarına göre seçildi: Intel i3-1315U (4C/6T),
7,6 GB RAM, CUDA yok. Hedef **≤ 5 M parametre** ve 256–512 token bağlam;
böylece eğitim dakikalar mertebesinde kalır ve bellek taşmaz.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Decoder-only transformer boyutları."""

    vocab_size: int
    context: int = 256
    d_model: int = 192
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.1
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size en az 2 olmalı")
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) n_head ({self.n_head}) ile tam bölünmeli"
            )
        if self.context < 2:
            raise ValueError("context en az 2 olmalı")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 0-1 arasında olmalı")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def parameter_estimate(self) -> int:
        """Tam parametre sayısı (model kurmadan bütçe hesabı).

        ``OpheliaGPT`` ile birebir aynı olmalı; eşitlik testle korunuyor.
        """
        embedding = self.vocab_size * self.d_model + self.context * self.d_model
        per_layer = (
            4 * self.d_model * self.d_model  # attention projeksiyonları (bias yok)
            + 2 * self.d_model * 4 * self.d_model  # MLP ağırlıkları
            + 5 * self.d_model  # MLP bias'ları (fc 4d + proj d)
            + 4 * self.d_model  # iki LayerNorm (gamma + beta)
        )
        head = 0 if self.tie_weights else self.d_model * self.vocab_size
        return embedding + self.n_layer * per_layer + head + 2 * self.d_model

    def to_dict(self) -> dict[str, object]:
        return {
            "vocab_size": self.vocab_size,
            "context": self.context,
            "d_model": self.d_model,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "dropout": self.dropout,
            "tie_weights": self.tie_weights,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ModelConfig:
        return cls(
            vocab_size=int(payload["vocab_size"]),
            context=int(payload.get("context", 256)),
            d_model=int(payload.get("d_model", 192)),
            n_layer=int(payload.get("n_layer", 4)),
            n_head=int(payload.get("n_head", 4)),
            dropout=float(payload.get("dropout", 0.1)),
            tie_weights=bool(payload.get("tie_weights", True)),
        )


@dataclass(frozen=True)
class TrainConfig:
    """Eğitim döngüsü parametreleri."""

    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 80
    max_steps: int = 900
    eval_interval: int = 150
    eval_batches: int = 10
    grad_clip: float = 1.0
    seed: int = 42
    threads: int = 6
    log_every: int = 50
    checkpoint_every: int = 150

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_size", self.batch_size),
            ("grad_accum", self.grad_accum),
            ("max_steps", self.max_steps),
            ("eval_interval", self.eval_interval),
        ):
            if value <= 0:
                raise ValueError(f"{name} pozitif olmalı: {value}")
        if self.lr <= 0:
            raise ValueError("lr pozitif olmalı")

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "lr": self.lr,
            "min_lr": self.min_lr,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "eval_interval": self.eval_interval,
            "eval_batches": self.eval_batches,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
            "threads": self.threads,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TrainConfig:
        defaults = cls()
        return cls(
            batch_size=int(payload.get("batch_size", defaults.batch_size)),
            grad_accum=int(payload.get("grad_accum", defaults.grad_accum)),
            lr=float(payload.get("lr", defaults.lr)),
            min_lr=float(payload.get("min_lr", defaults.min_lr)),
            weight_decay=float(payload.get("weight_decay", defaults.weight_decay)),
            warmup_steps=int(payload.get("warmup_steps", defaults.warmup_steps)),
            max_steps=int(payload.get("max_steps", defaults.max_steps)),
            eval_interval=int(payload.get("eval_interval", defaults.eval_interval)),
            eval_batches=int(payload.get("eval_batches", defaults.eval_batches)),
            grad_clip=float(payload.get("grad_clip", defaults.grad_clip)),
            seed=int(payload.get("seed", defaults.seed)),
            threads=int(payload.get("threads", defaults.threads)),
            log_every=int(payload.get("log_every", defaults.log_every)),
            checkpoint_every=int(
                payload.get("checkpoint_every", defaults.checkpoint_every)
            ),
        )


@dataclass(frozen=True)
class SampleConfig:
    """Üretim (örnekleme) parametreleri."""

    max_new_tokens: int = 512
    temperature: float = 0.95
    top_k: int = 40
    seed: int | None = None
    max_polyphony: int = 6
    # Token dilbilgisi kısıtı: model dilbilgisi dışı dizi üretemez (5. aşama).
    constrained: bool = True
    # EOS bu kadar nota üretilmeden yasak: birkaç notalık kırıntılar engellenir.
    # (Gerçek koraller 200-430 nota; 120 ≈ 30 sn'lik asgari bir eser.)
    min_notes: int = 120
    # Ritim politikası (ızgarada 1 = 16'lık). 0 = kısıtsız.
    # Gerçek korallerde notaların yalnızca %3'ü 16'lık ve başlangıçların %78'i
    # güçlü vuruşta; model 16'lık üretmeye bırakıldığında %30'a çıkıyor ve
    # sonuç "oyun müziği" gibi duyuluyor (bkz. docs/08_mixing.md).
    min_duration_steps: int = 0
    time_step_steps: int = 0

    def __post_init__(self) -> None:
        if self.min_notes < 1:
            raise ValueError("min_notes en az 1 olmalı")
        if self.min_duration_steps < 0:
            raise ValueError("min_duration_steps negatif olamaz")
        if self.time_step_steps < 0:
            raise ValueError("time_step_steps negatif olamaz")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens pozitif olmalı")
        if self.temperature <= 0:
            raise ValueError("temperature pozitif olmalı")
        if self.top_k <= 0:
            raise ValueError("top_k pozitif olmalı")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "seed": self.seed,
            "max_polyphony": self.max_polyphony,
            "constrained": self.constrained,
            "min_notes": self.min_notes,
            "min_duration_steps": self.min_duration_steps,
            "time_step_steps": self.time_step_steps,
        }
