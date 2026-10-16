"""Load trained PCTN checkpoints and run blood-pressure inference."""

from __future__ import annotations

from os.path import isfile
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pctn.data import build_dataloaders
from pctn.models import build_pctn
from pctn.preprocessing import iter_preprocessed_windows
from pctn.training.metrics import check_aami, compute_bp_metrics


INFERENCE_PATH_KEYS = (
    ("inference", "checkpoint"),
    ("data", "h5_path"),
    ("data", "split_dir"),
)


def pick_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
    return torch.device("cpu")


class BPPredictor:
    """Wrapper around a trained PCTN model for BP inference."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        preprocess_cfg: dict[str, Any] | None = None,
        eval_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.preprocess_cfg = preprocess_cfg or {}
        self.eval_cfg = eval_cfg or {}
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        cfg: dict[str, Any],
        checkpoint_path: str | None = None,
        device: torch.device | None = None,
    ) -> BPPredictor:
        inference_cfg = cfg["inference"]
        ckpt_path = checkpoint_path or inference_cfg["checkpoint"]
        if not isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        resolved_device = device or pick_device(inference_cfg.get("device", "cuda"))
        model = build_pctn(cfg["model"])
        checkpoint = torch.load(ckpt_path, map_location=resolved_device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(resolved_device)

        return cls(
            model=model,
            device=resolved_device,
            preprocess_cfg=cfg.get("preprocessing", {}),
            eval_cfg=cfg.get("evaluation", {}),
        )

    def _predict_batch(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            preds = self.model(x.to(self.device, non_blocking=True))
        return preds.detach().cpu().numpy()

    def predict_preprocessed_window(self, ppg_win: np.ndarray) -> tuple[float, float]:
        """
        Predict BP from a single preprocessed window.

        The input should already match the h5 training format:
        filtered, normalized, length = segment_length.
        """
        x = torch.from_numpy(np.asarray(ppg_win, dtype=np.float32))
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 2:
            x = x.unsqueeze(0)

        pred = self._predict_batch(x)[0]
        return float(pred[0]), float(pred[1])

    def predict_raw(
        self,
        ppg: np.ndarray,
        fs: float,
        *,
        aggregate: str = "median",
    ) -> dict[str, Any]:
        """
        Predict BP from raw sensor PPG.

        Applies resampling, sliding windows, and the same per-window preprocessing
        used when building the MIMIC dataset.
        """
        windows = iter_preprocessed_windows(ppg, fs=fs, cfg=self.preprocess_cfg)
        if not windows:
            raise ValueError("No valid windows could be extracted from the input signal.")

        per_window = np.array([self.predict_preprocessed_window(win) for win in windows], dtype=np.float64)
        if aggregate == "median":
            sbp = float(np.median(per_window[:, 0]))
            dbp = float(np.median(per_window[:, 1]))
        elif aggregate == "mean":
            sbp = float(np.mean(per_window[:, 0]))
            dbp = float(np.mean(per_window[:, 1]))
        else:
            raise ValueError(f"Unsupported aggregate: {aggregate}")

        return {
            "sbp": sbp,
            "dbp": dbp,
            "n_windows": len(windows),
            "per_window": per_window,
        }

    def evaluate_loader(self, loader: DataLoader, split: str = "test") -> dict[str, Any]:
        """Run inference on a DataLoader of preprocessed h5 samples."""
        preds_list: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []

        for x, y in tqdm(loader, desc=f"infer/{split}", leave=False):
            preds_list.append(self._predict_batch(x))
            targets_list.append(y.numpy())

        preds = np.concatenate(preds_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        metrics = compute_bp_metrics(preds, targets)
        metrics["predictions"] = preds
        metrics["targets"] = targets
        return metrics

    def evaluate_test_set(self, data_cfg: dict[str, Any]) -> dict[str, Any]:
        """Evaluate on the prepared h5 test split (no extra preprocessing)."""
        loaders = build_dataloaders(data_cfg)
        metrics = self.evaluate_loader(loaders["test"], split="test")

        me_th = self.eval_cfg.get("aami_me_threshold", 5.0)
        std_th = self.eval_cfg.get("aami_std_threshold", 8.0)
        metrics["aami"] = check_aami(metrics, me_th, std_th)
        return metrics

    @staticmethod
    def format_metrics(metrics: dict[str, Any]) -> str:
        lines = [
            f"loss={metrics.get('loss', float('nan')):.4f} | "
            f"SBP MAE={metrics['sbp_mae']:.2f} ME={metrics['sbp_me']:.2f} STD={metrics['sbp_std']:.2f} | "
            f"DBP MAE={metrics['dbp_mae']:.2f} ME={metrics['dbp_me']:.2f} STD={metrics['dbp_std']:.2f}"
        ]
        if "aami" in metrics:
            aami = metrics["aami"]
            lines.append(
                f"AAMI check: SBP={'PASS' if aami['sbp_pass'] else 'FAIL'}, "
                f"DBP={'PASS' if aami['dbp_pass'] else 'FAIL'}"
            )
        return "\n".join(lines)
