"""Model mimarisi testleri: şekiller, causal maske, kayıp, üretim."""

from __future__ import annotations

import pytest
import torch
from ophelia.model.config import ModelConfig, SampleConfig, TrainConfig
from ophelia.model.transformer import OpheliaGPT


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(
        vocab_size=50, context=16, d_model=32, n_layer=2, n_head=4, dropout=0.0
    )


@pytest.fixture
def model(config: ModelConfig) -> OpheliaGPT:
    torch.manual_seed(0)
    return OpheliaGPT(config)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=10, d_model=30, n_head=4)
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=10, context=1)
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=1)
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=10, dropout=1.0)
    with pytest.raises(ValueError):
        SampleConfig(max_new_tokens=0)
    with pytest.raises(ValueError):
        TrainConfig(batch_size=0)


def test_head_dim(config: ModelConfig) -> None:
    assert config.head_dim == 8


def test_parameter_estimate_is_exact(model: OpheliaGPT, config: ModelConfig) -> None:
    """Bütçe hesabı modelin gerçek parametre sayısına eşit olmalı."""
    assert config.parameter_estimate == model.num_params


def test_parameter_estimate_counts_untied_head() -> None:
    tied = ModelConfig(vocab_size=100, d_model=32, n_layer=2, n_head=2)
    untied = ModelConfig(
        vocab_size=100, d_model=32, n_layer=2, n_head=2, tie_weights=False
    )
    assert untied.parameter_estimate - tied.parameter_estimate == 100 * 32


def test_forward_shapes(model: OpheliaGPT, config: ModelConfig) -> None:
    idx = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(idx)

    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is None

    targets = torch.randint(0, config.vocab_size, (2, 8))
    _, loss = model(idx, targets)
    assert loss is not None
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0


def test_loss_ignores_padding(model: OpheliaGPT, config: ModelConfig) -> None:
    idx = torch.randint(0, config.vocab_size, (1, 8))
    targets = torch.randint(0, config.vocab_size, (1, 8))
    masked = targets.clone()
    masked[:, 4:] = -100

    _, full_loss = model(idx, targets)
    _, masked_loss = model(idx, masked)

    assert float(masked_loss.detach()) != float(full_loss.detach())
    assert torch.isfinite(masked_loss)


def test_causal_mask_blocks_future_tokens(model: OpheliaGPT, config: ModelConfig) -> None:
    torch.manual_seed(1)
    idx = torch.randint(0, config.vocab_size, (1, 10))
    altered = idx.clone()
    altered[0, -1] = (int(idx[0, -1]) + 1) % config.vocab_size

    logits_a, _ = model(idx)
    logits_b, _ = model(altered)

    # Son token dışındaki konumlar değişmemeli
    assert torch.allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-5)


def test_tied_weights(model: OpheliaGPT) -> None:
    assert model.lm_head.weight is model.token_embedding.weight


def test_context_overflow_raises(model: OpheliaGPT, config: ModelConfig) -> None:
    idx = torch.zeros((1, config.context + 1), dtype=torch.long)
    with pytest.raises(ValueError):
        model(idx)


def test_generate_respects_positional_limit(model: OpheliaGPT) -> None:
    torch.manual_seed(2)
    prompt = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(prompt, 5, temperature=1.0, top_k=10)
    assert out.shape == (1, 6)
    assert out.dtype == torch.long


def test_generate_is_deterministic_with_seed(model: OpheliaGPT) -> None:
    prompt = torch.zeros((1, 1), dtype=torch.long)

    generator_a = torch.Generator().manual_seed(123)
    first = model.generate(prompt, 6, temperature=1.0, top_k=20, generator=generator_a)

    generator_b = torch.Generator().manual_seed(123)
    second = model.generate(prompt, 6, temperature=1.0, top_k=20, generator=generator_b)

    assert torch.equal(first, second)


def test_generate_stops_at_stop_token(config: ModelConfig) -> None:
    class _Stub(OpheliaGPT):
        def forward(self, idx, targets=None, pad_id=-100):  # type: ignore[override]
            batch, length = idx.shape
            logits = torch.full((batch, length, self.config.vocab_size), -20.0)
            logits[..., 1] = 20.0
            return logits, None

    stub = _Stub(config)
    out = stub.generate(torch.zeros((1, 1), dtype=torch.long), 10, stop_id=1)
    assert out.shape == (1, 2)
    assert int(out[0, -1]) == 1


def test_generate_within_context_window(config: ModelConfig) -> None:
    model = OpheliaGPT(config)
    prompt = torch.zeros((1, config.context), dtype=torch.long)
    out = model.generate(prompt, 3, temperature=1.0)
    assert out.shape[1] == config.context + 3
