"""Split MIMIC-III PPG/BP h5 dataset for PyTorch training.

Mirrors the subject-based split logic in h5_to_tfrecord.py, but writes index
files and CSV summaries instead of TFRecord shards. The resulting splits are
consumed by pctn-bp via pctn.data.dataset.MIMICPPGDataset.

File: h5_split_for_pytorch.py
"""

from __future__ import annotations

import json
from datetime import datetime
from os import makedirs
from os.path import isdir, join
from pctn.utils.config import load_data_config

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _split_subjects(subject_idx: np.ndarray, train_ratio: float,val_ratio: float, test_ratio: float, seed: int,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    subjects = np.unique(subject_idx)
    holdout_ratio = val_ratio + test_ratio

    subjects_train, subjects_holdout = train_test_split(
        subjects,
        test_size=holdout_ratio,
        random_state=seed,
    )
    test_share_of_holdout = test_ratio / holdout_ratio
    subjects_val, subjects_test = train_test_split(
        subjects_holdout,
        test_size=test_share_of_holdout,
        random_state=seed,
    )
    return subjects_train, subjects_val, subjects_test


def _indices_for_subjects(subject_idx: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(subject_idx, subjects))


def _maybe_subsample(indices: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=max_samples, replace=False)


def _save_bp_csv(labels: np.ndarray, indices: np.ndarray, csv_path: str) -> None:
    bp = labels[indices]
    pd.DataFrame({"SBP": bp[:, 0], "DBP": bp[:, 1]}).to_csv(csv_path, index=False)


def split_h5_for_pytorch(
    source_file: str,
    output_dir: str,
    train_ratio: float = 0.65,
    val_ratio: float = 0.10,
    test_ratio: float = 0.25,
    divide_by_subject: bool = True,
    seed: int = 42,
    max_train: int | None = None,
    max_val: int | None = None,
    max_test: int | None = None,
) -> dict:
    """Create train/val/test index files from a prepared MIMIC h5 dataset."""
    if not isdir(output_dir):
        makedirs(output_dir)

    split_dir = join(output_dir, "splits")
    if not isdir(split_dir):
        makedirs(split_dir)

    with h5py.File(source_file, "r") as f:
        labels = np.array(f["label"])
        subject_idx = np.squeeze(np.array(f["subject_idx"]))

    n_samples = labels.shape[0]
    subject_idx = subject_idx[:n_samples]
    all_indices = np.arange(n_samples, dtype=np.int64)

    if divide_by_subject:
        subjects_train, subjects_val, subjects_test = _split_subjects(
            subject_idx, train_ratio, val_ratio, test_ratio, seed
        )
        train_pool = _indices_for_subjects(subject_idx, subjects_train)
        val_pool = _indices_for_subjects(subject_idx, subjects_val)
        test_pool = _indices_for_subjects(subject_idx, subjects_test)
    else:
        subjects_train = subjects_val = subjects_test = np.array([], dtype=subject_idx.dtype)
        train_pool, temp_pool = train_test_split(
            all_indices,
            train_size=train_ratio,
            random_state=seed,
        )
        relative_test = test_ratio / (val_ratio + test_ratio)
        val_pool, test_pool = train_test_split(
            temp_pool,
            test_size=relative_test,
            random_state=seed,
        )

    idx_train = _maybe_subsample(train_pool, max_train, seed)
    idx_val = _maybe_subsample(val_pool, max_val, seed + 1)
    idx_test = _maybe_subsample(test_pool, max_test, seed + 2)

    rng = np.random.default_rng(seed)
    rng.shuffle(idx_train)

    np.save(join(split_dir, "train_indices.npy"), idx_train)
    np.save(join(split_dir, "val_indices.npy"), idx_val)
    np.save(join(split_dir, "test_indices.npy"), idx_test)

    _save_bp_csv(labels, idx_train, join(output_dir, "MIMIC-III_BP_trainset.csv"))
    _save_bp_csv(labels, idx_val, join(output_dir, "MIMIC-III_BP_valset.csv"))
    _save_bp_csv(labels, idx_test, join(output_dir, "MIMIC-III_BP_testset.csv"))

    meta = {
        "source_file": source_file,
        "divide_by_subject": divide_by_subject,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "max_train": max_train,
        "max_val": max_val,
        "max_test": max_test,
        "n_samples_total": int(n_samples),
        "n_subjects_total": int(np.unique(subject_idx).size),
        "n_subjects_train": int(subjects_train.size),
        "n_subjects_val": int(subjects_val.size),
        "n_subjects_test": int(subjects_test.size),
        "n_samples_train": int(idx_train.size),
        "n_samples_val": int(idx_val.size),
        "n_samples_test": int(idx_test.size),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(join(split_dir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Split complete: train={meta['n_samples_train']}, "
        f"val={meta['n_samples_val']}, test={meta['n_samples_test']}"
    )
    print(f"Indices saved to: {split_dir}")
    return meta


if __name__ == "__main__":
    cfg = load_data_config()
    SourceFile = cfg["processed_h5"]
    OutputDir  = cfg["split_dir"]
    # SourceFile = "/home/zhanglanli1/code/learning/projects/data_preprocessing/ppg_rawdata/MIMIC-III_ppg_dataset.h5"
    # OutputDir = "/home/zhanglanli1/code/learning/projects/data_preprocessing"
    train_ratio = 0.65
    val_ratio = 0.10
    test_ratio = 0.25
    divide_by_subject = True  # True=按患者划分, False=按样本随机划分
    seed = 42
    max_train = None  # 限制训练集最大样本数，None 表示不限制
    max_val = None
    max_test = None

    np.random.seed(seed=seed)

    split_h5_for_pytorch(
        source_file=SourceFile,
        output_dir=OutputDir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        divide_by_subject=divide_by_subject,
        seed=seed,
        max_train=max_train,
        max_val=max_val,
        max_test=max_test,
    )
