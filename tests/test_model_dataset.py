"""Pencere veri yükleyici testleri."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ophelia.data.token_dataset import SplitArrays
from ophelia.model.dataset import TokenWindows, make_loader, pad_token_id


def _arrays(sequences: list[list[int]]) -> SplitArrays:
    tokens: list[int] = []
    offsets = [0]
    for sequence in sequences:
        tokens.extend(sequence)
        offsets.append(len(tokens))
    return SplitArrays(
        tokens=np.asarray(tokens, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
        files=tuple(f"f{index}" for index in range(len(sequences))),
    )


def test_pad_token_id() -> None:
    assert pad_token_id(["<pad>", "<bos>", "<eos>"]) == 0
    with pytest.raises(ValueError):
        pad_token_id(["<bos>"])


def test_short_sequence_is_padded() -> None:
    arrays = _arrays([list(range(1, 6))])  # 5 token
    dataset = TokenWindows(arrays, context=8, pad_id=0)

    assert len(dataset) == 1
    inputs, targets = dataset[0]
    assert inputs.shape == (8,)
    assert targets.shape == (8,)

    assert inputs.tolist()[:5] == [1, 2, 3, 4, 5]
    assert inputs.tolist()[5:] == [0, 0, 0]
    # Son gerçek token'ın hedefi yok; kalanı dolgu kimliği (0)
    assert targets.tolist()[4] == 0
    assert torch.all(targets[4:] == 0)


def test_targets_are_shifted_by_one() -> None:
    arrays = _arrays([[10, 11, 12, 13]])
    dataset = TokenWindows(arrays, context=6, pad_id=0)
    inputs, targets = dataset[0]

    assert inputs.tolist()[:4] == [10, 11, 12, 13]
    assert targets.tolist()[0] == 11
    assert targets.tolist()[1] == 12
    assert targets.tolist()[2] == 13


def test_long_sequence_is_windowed_with_stride() -> None:
    arrays = _arrays([list(range(1, 101))])  # 100 token
    dataset = TokenWindows(arrays, context=32, pad_id=0, stride=16)

    # range(0, 99, 16) -> 0,16,...,96 artı dosya sonu (EOS) penceresi 67
    assert len(dataset) == 8
    assert dataset.windows[0].start == 0
    assert dataset.windows[1].start == 16


def test_tail_window_covers_end_of_file() -> None:
    """Son pencere tam bağlamlı olmalı: EOS öğrenilebilir bir sinyal olsun."""
    arrays = _arrays([list(range(1, 101))])  # 100 token
    dataset = TokenWindows(arrays, context=32, pad_id=0, stride=16)

    tail = next(window for window in dataset.windows if window.start == 100 - 1 - 32)
    assert tail.length == 33

    sequence = dataset.sample_sequence(dataset.windows.index(tail))
    assert sequence[-1] == 100  # dosyanın son token'ı (EOS) pencerede


def test_tail_window_can_be_disabled() -> None:
    arrays = _arrays([list(range(1, 101))])
    dataset = TokenWindows(arrays, context=32, pad_id=0, stride=16, tail=False)
    assert len(dataset) == 7


def test_default_stride_is_half_context() -> None:
    arrays = _arrays([list(range(1, 65))])
    dataset = TokenWindows(arrays, context=16, pad_id=0)
    assert dataset.stride == 8


def test_overlapping_windows_share_tokens() -> None:
    arrays = _arrays([list(range(1, 41))])
    dataset = TokenWindows(arrays, context=16, pad_id=0, stride=8)
    first = dataset.sample_sequence(0)
    second = dataset.sample_sequence(1)
    # Stride kadar örtüşen bölge birebir aynı olmalı
    assert first[8:16].tolist() == second[:8].tolist()


def test_multiple_files_are_covered() -> None:
    arrays = _arrays([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    dataset = TokenWindows(arrays, context=8, pad_id=0)
    assert len(dataset) == 3
    assert dataset.files_covered == 3


def test_single_token_file_is_skipped() -> None:
    arrays = _arrays([[1], [1, 2, 3]])
    dataset = TokenWindows(arrays, context=8, pad_id=0)
    assert len(dataset) == 1
    assert dataset.windows[0].file_index == 1


def test_max_windows_per_file() -> None:
    arrays = _arrays([list(range(1, 801))])
    dataset = TokenWindows(arrays, context=16, pad_id=0, max_windows_per_file=10)
    assert len(dataset) == 10


def test_invalid_context() -> None:
    with pytest.raises(ValueError):
        TokenWindows(_arrays([[1, 2]]), context=1, pad_id=0)


def test_loader_batching() -> None:
    arrays = _arrays([[index] * 33 for index in range(1, 11)])
    dataset = TokenWindows(arrays, context=16, pad_id=0)
    loader = make_loader(dataset, batch_size=4, shuffle=False)

    batch_inputs, batch_targets = next(iter(loader))
    assert batch_inputs.shape == (4, 16)
    assert batch_targets.shape == (4, 16)
    assert batch_inputs.dtype == torch.long
