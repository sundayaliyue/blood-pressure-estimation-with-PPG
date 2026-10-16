"""Production-facing inference service API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from pctn.inference.predictor import BPPredictor, INFERENCE_PATH_KEYS
from pctn.utils import load_config


@dataclass(frozen=True)
class BPPrediction:
    sbp: float
    dbp: float
    n_windows: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class BPInferenceService:
    """
    Stable inference interface for application integration.

    Load the model once at startup, then call predict() from App / API / device bridge code.
    """

    def __init__(self, predictor: BPPredictor, aggregate: str = "median") -> None:
        self._predictor = predictor
        self._aggregate = aggregate

    @classmethod
    def from_config(cls, config_path: str | Path) -> BPInferenceService:
        cfg = load_config(config_path, path_keys=INFERENCE_PATH_KEYS)
        predictor = BPPredictor.from_checkpoint(cfg)
        aggregate = cfg.get("inference", {}).get("aggregate", "median")
        return cls(predictor=predictor, aggregate=aggregate)

    @property
    def device(self):
        return self._predictor.device

    def predict(self, ppg: Sequence[float] | np.ndarray, fs: float) -> BPPrediction:
        """Predict BP from a raw 1D PPG waveform and its sampling rate."""
        ppg_array = np.asarray(ppg, dtype=np.float64).squeeze()
        if ppg_array.ndim != 1:
            raise ValueError(f"Expected a 1D PPG array, got shape {ppg_array.shape}")
        if fs <= 0:
            raise ValueError(f"Sampling rate must be positive, got {fs}")

        result = self._predictor.predict_raw(ppg_array, fs=fs, aggregate=self._aggregate)
        return BPPrediction(
            sbp=result["sbp"],
            dbp=result["dbp"],
            n_windows=result["n_windows"],
        )
