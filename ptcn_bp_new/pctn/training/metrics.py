from __future__ import annotations

import numpy as np


def compute_bp_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute MAE, ME, STD for SBP and DBP."""
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    errors = preds - targets

    sbp_err = errors[:, 0]
    dbp_err = errors[:, 1]

    metrics = {
        "sbp_mae": float(np.mean(np.abs(sbp_err))),
        "dbp_mae": float(np.mean(np.abs(dbp_err))),
        "sbp_me": float(np.mean(sbp_err)),
        "dbp_me": float(np.mean(dbp_err)),
        "sbp_std": float(np.std(sbp_err)),
        "dbp_std": float(np.std(dbp_err)),
    }
    metrics["mae_mean"] = (metrics["sbp_mae"] + metrics["dbp_mae"]) / 2
    return metrics


def check_aami(metrics: dict[str, float], me_threshold: float, std_threshold: float) -> dict[str, bool]:
    return {
        "sbp_pass": abs(metrics["sbp_me"]) <= me_threshold and metrics["sbp_std"] <= std_threshold,
        "dbp_pass": abs(metrics["dbp_me"]) <= me_threshold and metrics["dbp_std"] <= std_threshold,
    }
