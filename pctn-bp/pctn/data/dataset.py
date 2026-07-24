from __future__ import annotations

from os.path import isfile, join
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _read_ppg_segment(ppg: np.ndarray, segment_length: int | None) -> np.ndarray:
    """Return PPG as stored in h5; optionally truncate if longer than segment_length."""
    if segment_length is None:
        return ppg
    if ppg.shape[-1] > segment_length:
        return ppg[..., :segment_length]
    return ppg


class MIMICPPGDataset(Dataset):
    """Read PPG/BP samples from the prepared MIMIC h5 file via precomputed indices."""

    def __init__(self,h5_path: str,indices: np.ndarray | str,segment_length: int | None = None,) -> None:
        self.h5_path = h5_path
        self.segment_length = segment_length
        if isinstance(indices, str):
            self.indices = np.load(indices)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)
        self._h5: h5py.File | None = None

        with h5py.File(h5_path, "r") as f:
            self.native_segment_length = int(f["ppg"].shape[1])

    def __len__(self) -> int:
        return self.indices.shape[0]

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r") #返回一个 h5py.File 文件句柄（类似数据库连接）
        return self._h5

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_idx = int(self.indices[idx])
        h5 = self._get_h5()

        ppg = np.array(h5["ppg"][sample_idx], dtype=np.float32)
        label = np.array(h5["label"][sample_idx], dtype=np.float32)
        ppg = _read_ppg_segment(ppg, self.segment_length)

        x = torch.from_numpy(ppg).unsqueeze(0) #加一个通道数的维度
        y = torch.from_numpy(label)
        return x, y


def _resolve_indices_dir(split_dir: str) -> str:
    if isfile(join(split_dir, "train_indices.npy")):
        return split_dir
    nested = join(split_dir, "splits")
    if isfile(join(nested, "train_indices.npy")):
        return nested
    raise FileNotFoundError(
        f"Could not find train_indices.npy in {split_dir} or {nested}. "
        "Run data_preprocessing/h5_split_for_pytorch.py first."
    )


def build_dataloaders(data_cfg: dict[str, Any]) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from pctn config `data` section."""
    h5_path = data_cfg["h5_path"]
    indices_dir = _resolve_indices_dir(data_cfg["split_dir"])
    segment_length = data_cfg.get("segment_length")
    batch_size = data_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 4)

    loaders = {}
    for split in ("train", "val", "test"):
        dataset = MIMICPPGDataset(
            h5_path=h5_path,
            indices=join(indices_dir, f"{split}_indices.npy"),
            segment_length=segment_length,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=split == "train",
        )
    return loaders
