"""PPG preprocessing for inference (mirrors prepare_MIMIC_dataset.py logic)."""

from __future__ import annotations

from math import gcd
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly


def bandpass_filter(
    ppg: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 8.0,
    order: int = 4,
) -> np.ndarray:
    """Apply the same Butterworth bandpass filter used during dataset preparation."""
    b, a = butter(order, [low, high], btype="bandpass", fs=fs)
    return filtfilt(b, a, np.asarray(ppg, dtype=np.float64))


def normalize_window(ppg: np.ndarray) -> np.ndarray:
    """Per-window demean and z-score normalization (same as training pipeline)."""
    ppg = np.asarray(ppg, dtype=np.float64)
    ppg = ppg - np.mean(ppg)
    std = np.std(ppg)
    if std > 0:
        ppg = ppg / std
    return ppg


def resample_to_target_fs(ppg: np.ndarray, fs: float, target_fs: float) -> np.ndarray:
    """Resample PPG to the target sampling rate using polyphase resampling."""
    ppg = np.asarray(ppg, dtype=np.float64)
    if np.isclose(fs, target_fs):
        return ppg

    fs_scaled = int(round(fs * 1000))
    target_scaled = int(round(target_fs * 1000))
    common = gcd(fs_scaled, target_scaled)
    up = target_scaled // common
    down = fs_scaled // common
    return resample_poly(ppg, up, down)


def create_window_starts(signal_length: int, segment_length: int, overlap: float) -> np.ndarray:
    """Return start indices for sliding windows."""
    if signal_length < segment_length:
        raise ValueError(
            f"Signal too short: got {signal_length} samples, need at least {segment_length}."
        )

    step = max(1, int(round(segment_length * (1.0 - overlap))))
    return np.arange(0, signal_length - segment_length + 1, step, dtype=int)


def preprocess_window(ppg_win: np.ndarray, fs: float, cfg: dict[str, Any]) -> np.ndarray:
    """Filter and normalize a single window that is already at target_fs and correct length."""
    target_fs = float(cfg.get("target_fs", 125))
    band = cfg.get("filter_band", [0.5, 8.0])
    order = int(cfg.get("filter_order", 4))

    filtered = bandpass_filter(ppg_win, fs=target_fs, low=band[0], high=band[1], order=order)
    normalized = normalize_window(filtered)
    return normalized.astype(np.float32)


def iter_preprocessed_windows(
    ppg: np.ndarray,
    fs: float,
    cfg: dict[str, Any],
) -> list[np.ndarray]:
    """
    Resample raw PPG, slide over windows, and apply per-window preprocessing.

    Returns a list of windows shaped (segment_length,) ready for model input.
    """
    target_fs = float(cfg.get("target_fs", 125))
    segment_length = int(cfg.get("segment_length", 875))
    overlap = float(cfg.get("win_overlap", 0.5))

    resampled = resample_to_target_fs(ppg, fs=fs, target_fs=target_fs)
    starts = create_window_starts(len(resampled), segment_length, overlap)

    windows: list[np.ndarray] = []
    for start in starts:
        stop = start + segment_length
        ppg_win = resampled[start:stop]
        windows.append(preprocess_window(ppg_win, fs=target_fs, cfg=cfg))
    return windows
