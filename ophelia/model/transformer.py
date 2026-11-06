"""Decoder-only transformer (GPT benzeri), CPU dostu minimal uygulama.

``torch.nn`` içindeki hazır bloklar yerine açık yazıldı: causal maskenin
doğruluğu testlerle kanıtlanabilsin ve katman sayısı/boyutları kolay deneyle
ayarlanabilsin.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from ophelia.model.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Maskelenmiş çok başlı öz-dikkat."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context, config.context, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        query = self.query(x).view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch, length, self.n_head, self.head_dim).transpose(1, 2)

        weights = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:length, :length]
        weights = weights.masked_fill(~causal, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        out = (weights @ value).transpose(1, 2).contiguous().view(batch, length, dim)
        return self.proj(out)


class MLP(nn.Module):
    """Pozisyon bazlı ileri besleme katmanı."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.proj = nn.Linear(4 * config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer bloğu."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class OpheliaGPT(nn.Module):
    """Küçük decoder-only dil modeli (müzik token'ları için)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_params(self) -> int:
        # Tied ağırlık iki kez sayılmasın
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        pad_id: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``idx``: (B, T) int64 token kimlikleri. Dönen: logits, kayıp."""
        batch, length = idx.shape
        if length > self.config.context:
            raise ValueError(
                f"dizi bağlamdan uzun: {length} > {self.config.context}"
            )

        positions = torch.arange(length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=pad_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_id: int | None = None,
        generator: torch.Generator | None = None,
        allowed: Callable[[list[int]], torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Otoregresif örnekleme; ``stop_id`` üretilince durur.

        ``allowed`` verilirse her adımda üretilmiş kimlikler listesiyle çağrılır
        ve dönen ``bool`` maske dışındaki token'lar yasaklanır (dilbilgisi
        kısıtlı örnekleme).
        """
        self.eval()
        prompt_length = idx.size(1)
        for _ in range(max_new_tokens):
            window = idx[:, -self.config.context :]
            logits, _ = self(window)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if allowed is not None:
                generated = [int(value) for value in idx[0, prompt_length:].tolist()]
                mask = allowed(generated)
                if mask is not None:
                    blocked = ~mask.to(logits.device)
                    logits = logits.masked_fill(blocked.unsqueeze(0), float("-inf"))

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            idx = torch.cat([idx, next_token], dim=1)

            if stop_id is not None and int(next_token.item()) == stop_id:
                break
        return idx
