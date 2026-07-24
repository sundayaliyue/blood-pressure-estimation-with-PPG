from __future__ import annotations

from dataclasses import dataclass
from os import makedirs
from os.path import join
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import check_aami, compute_bp_metrics


@dataclass
class TrainState:
    epoch: int = 0
    best_val_loss: float = float("inf") # 先赋值正无穷大


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str,
        log_interval: int = 50,
        eval_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.eval_cfg = eval_cfg or {}
        self.criterion = nn.MSELoss()
        self.state = TrainState()
        makedirs(checkpoint_dir, exist_ok=True)

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, dict[str, float]]:
        self.model.train(train)
        losses: list[float] = []
        preds_list: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []

        pbar = tqdm(loader, leave=False, desc="train" if train else "eval")
        for step, (x, y) in enumerate(pbar, start=1):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    pred = self.model(x)
                    loss = self.criterion(pred, y)

            loss_value = float(loss.item())
            losses.append(loss_value)
            preds_list.append(pred.detach().cpu().numpy())
            targets_list.append(y.detach().cpu().numpy())

            if train and step % self.log_interval == 0:
                pbar.set_postfix(loss=f"{loss_value:.4f}")

        metrics = compute_bp_metrics(np.concatenate(preds_list), np.concatenate(targets_list))
        metrics["loss"] = float(np.mean(losses))
        return metrics["loss"], metrics

    def train(self, epochs: int) -> None:
        for epoch in range(1, epochs + 1):
            self.state.epoch = epoch
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss, train_metrics = self._run_epoch(self.train_loader, train=True)
            val_loss, val_metrics = self._run_epoch(self.val_loader, train=False)

            self._print_metrics("train", train_loss, train_metrics)
            self._print_metrics("val", val_loss, val_metrics)

            if val_loss < self.state.best_val_loss:
                self.state.best_val_loss = val_loss
                self.save_checkpoint("best.pt")
                print(f"Saved best checkpoint (val_loss={val_loss:.4f})")

            self.save_checkpoint("last.pt")

        if self.test_loader is not None:
            self.load_checkpoint("best.pt")
            test_loss, test_metrics = self._run_epoch(self.test_loader, train=False)
            self._print_metrics("test", test_loss, test_metrics, final=True)

    def _print_metrics(self, split: str, loss: float, metrics: dict[str, float],final: bool = False,) -> None:
        me_th = self.eval_cfg.get("aami_me_threshold", 5.0)
        std_th = self.eval_cfg.get("aami_std_threshold", 8.0)
        aami = check_aami(metrics, me_th, std_th)

        print(
            f"[{split}] loss={loss:.4f} | "
            f"SBP MAE={metrics['sbp_mae']:.2f} ME={metrics['sbp_me']:.2f} STD={metrics['sbp_std']:.2f} | "
            f"DBP MAE={metrics['dbp_mae']:.2f} ME={metrics['dbp_me']:.2f} STD={metrics['dbp_std']:.2f}"
        )
        if final:
            print(
                f"AAMI check: SBP={'PASS' if aami['sbp_pass'] else 'FAIL'}, "
                f"DBP={'PASS' if aami['dbp_pass'] else 'FAIL'}"
            )

    def save_checkpoint(self, filename: str) -> None:
        path = join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": self.state.epoch,
                "best_val_loss": self.state.best_val_loss,
                "model_state_dict": self.model.state_dict(), # 模型的所有参数权重
                "optimizer_state_dict": self.optimizer.state_dict(), #优化器的状态
            },
            path,
        )

    def load_checkpoint(self, filename: str) -> None:
        path = join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state.epoch = checkpoint.get("epoch", 0)
        self.state.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
