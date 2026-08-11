"""Train PCTN on the prepared MIMIC-III dataset."""

from __future__ import annotations

from os.path import abspath, dirname, isfile, join
from pathlib import Path

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from pctn.data import build_dataloaders
from pctn.models import build_pctn
from pctn.training import Trainer
from pctn.utils import load_config, set_seed
from pctn.utils.config import TRAIN_PATH_KEYS

PROJECT_ROOT = Path(__file__).resolve().parent

def pick_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
    return torch.device("cpu")


def main(config_path: str) -> None:
    cfg = load_config(config_path, path_keys=TRAIN_PATH_KEYS)
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
