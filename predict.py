"""Offline validation script — mirrors train.py, not the production entry point."""

from __future__ import annotations

from os.path import abspath, dirname, isfile, join

from pctn.inference import BPPredictor, INFERENCE_PATH_KEYS
from pctn.utils import load_config


def main(config_path: str) -> None:
    cfg = load_config(config_path, path_keys=INFERENCE_PATH_KEYS)
    data_cfg = cfg["data"]

    if not isfile(data_cfg["h5_path"]):
        raise FileNotFoundError(f"H5 dataset not found: {data_cfg['h5_path']}")

    predictor = BPPredictor.from_checkpoint(cfg)
    print(f"Loaded checkpoint: {cfg['inference']['checkpoint']}")
    print(f"Using device: {predictor.device}")

    metrics = predictor.evaluate_test_set(data_cfg)
    print("[test] " + BPPredictor.format_metrics(metrics))


if __name__ == "__main__":
    ConfigPath = join(dirname(abspath(__file__)), "configs", "inference.yaml")
    main(ConfigPath)
