"""Pencere tabanlı veri yükleyici.

Token dizileri değişken uzunlukta (312–4491), model ise sabit bağlam ister.
Her dosya ``context + 1`` token'lık pencerelere bölünür; kısa dosyalar
``PAD`` ile doldurulur ve dolgu konumları kayıpta **yok sayılır**.

Pencere örtüşmesi (stride) varsayılan olarak ``context // 2``'dir; böylece
dosya sınırlarında bağlam kaybı azalır ve örnek sayısı iki katına çıkar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ophelia.data.token_dataset import SplitArrays
from ophelia.data.tokens import PAD


@dataclass(frozen=True)
class WindowSpec:
    """Bir pencerenin kaynağı.

    ``start`` **mutlak** token indeksidir ve ``length`` dosya sınırını aşmaz;
    böylece bir dosyanın sonundaki pencere komşu dosyanın token'larına taşmaz.
    """

    file_index: int
    start: int
    length: int


class TokenWindows(Dataset):
    """``SplitArrays`` üzerinden (girdi, hedef) pencereleri üretir."""

    def __init__(
        self,
        arrays: SplitArrays,
        context: int,
        *,
        pad_id: int,
        stride: int | None = None,
        max_windows_per_file: int | None = None,
        tail: bool = True,
    ) -> None:
        if context < 2:
            raise ValueError("context en az 2 olmalı")

        self.context = context
        self.pad_id = pad_id
        self.stride = stride if stride is not None else max(1, context // 2)
        self.tail = tail
        self.windows: list[WindowSpec] = []

        for file_index in range(len(arrays)):
            file_start = int(arrays.offsets[file_index])
            file_end = int(arrays.offsets[file_index + 1])
            length = file_end - file_start
            if length < 2:
                continue

            if length <= context + 1:
                self.windows.append(WindowSpec(file_index, file_start, length))
                continue

            starts = list(range(file_start, file_end - 1, self.stride))
            if self.tail:
                # Dosya sonunda tam bağlamlı bir pencere: ``EOS`` yalnızca burada
                # güçlü bir öğrenme sinyali olur (aksi hâlde model eseri
                # bitirmeyi hiç öğrenemez, üretim bütçeyle kesilir).
                tail_start = file_end - 1 - context
                if tail_start > file_start:
                    starts.append(tail_start)
                starts = sorted(set(starts))

            if max_windows_per_file is not None and len(starts) > max_windows_per_file:
                step = len(starts) / max_windows_per_file
                starts = [starts[int(index * step)] for index in range(max_windows_per_file)]

            self.windows.extend(
                WindowSpec(file_index, start, min(context + 1, file_end - start))
                for start in starts
            )

        self._tokens = arrays.tokens

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def files_covered(self) -> int:
        return len({window.file_index for window in self.windows})

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self.windows[index]
        length = min(spec.length, self.context + 1)

        chunk = np.full(self.context + 1, self.pad_id, dtype=np.int64)
        chunk[:length] = self._tokens[spec.start : spec.start + length]

        inputs = torch.from_numpy(chunk[:-1])
        targets = torch.from_numpy(chunk[1:].copy())
        # Dolgu ve son gerçek token'ın hedefi yok: ``pad_id`` ile işaretlenir ve
        # modelin kaybı bu kimliği yok sayar (tek kural, iki farklı sabit yok).
        targets[length - 1 :] = self.pad_id
        return inputs, targets

    def sample_sequence(self, index: int) -> np.ndarray:
        """Pencerenin dolgusuz token dizisi (inceleme/test için)."""
        spec = self.windows[index]
        return self._tokens[spec.start : spec.start + spec.length]


def make_loader(
    dataset: TokenWindows,
    *,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Tek iş parçacıklı (Windows'ta ucuz) veri yükleyici."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def pad_token_id(vocab_tokens: tuple[str, ...] | list[str]) -> int:
    """Sözlükteki ``<pad>`` kimliği."""
    try:
        return list(vocab_tokens).index(PAD)
    except ValueError as exc:  # pragma: no cover - sözlük her zaman pad içerir
        raise ValueError("sözlükte <pad> yok") from exc
