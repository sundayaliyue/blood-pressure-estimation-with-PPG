"""Train PCTN on the prepared MIMIC-III dataset."""

from __future__ import annotations

import random
from os.path import abspath, dirname, isfile, join
from pathlib import Path

import numpy as np
import torch
import yaml

from pctn.data import build_dataloaders
from pctn.models import build_pctn
from pctn.training import Trainer


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return str(candidate.resolve())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    data_cfg["h5_path"] = resolve_path(data_cfg["h5_path"])
    data_cfg["split_dir"] = resolve_path(data_cfg["split_dir"])
    cfg["training"]["checkpoint_dir"] = resolve_path(cfg["training"]["checkpoint_dir"])
    return cfg


def pick_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
    return torch.device("cpu")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    set_seed(train_cfg["seed"])

    if not isfile(data_cfg["h5_path"]):
        raise FileNotFoundError(f"H5 dataset not found: {data_cfg['h5_path']}")

    batch_size = train_cfg.get("batch_size", data_cfg.get("batch_size", 64))
    data_cfg["batch_size"] = batch_size

    device = pick_device(train_cfg.get("device", "cuda"))
    print(f"Using device: {device}")

    loaders = build_dataloaders(data_cfg)
    print(
        f"Dataset sizes: train={len(loaders['train'].dataset)}, "
        f"val={len(loaders['val'].dataset)}, test={len(loaders['test'].dataset)}"
    )

    model = build_pctn(cfg["model"]).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg["lr"],
        momentum=0.9,
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        optimizer=optimizer,
        device=device,
        checkpoint_dir=train_cfg["checkpoint_dir"],
        log_interval=train_cfg.get("log_interval", 50),
        eval_cfg=cfg.get("evaluation", {}),
    )

    trainer.train(epochs=train_cfg["epochs"])


if __name__ == "__main__":
    ConfigPath = join(dirname(abspath(__file__)), "configs", "default.yaml")

    main(ConfigPath)
