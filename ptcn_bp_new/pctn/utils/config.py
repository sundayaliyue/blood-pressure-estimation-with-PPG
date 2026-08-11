from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH_KEYS = (
    ("raw_dir",),
    ("processed_h5",),
    ("split_dir",),
    ("records_file",),
)

TRAIN_PATH_KEYS = (
    ("data", "h5_path"),
    ("data", "split_dir"),
    ("training", "checkpoint_dir"),
)


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    base = root or PROJECT_ROOT
    return (base / candidate).resolve()


def load_config(
    path: str | Path,
    *,
    root: Path | None = None,
    path_keys: tuple[tuple[str, ...], ...] | None = None,
) -> dict[str, Any]:
    config_path = resolve_path(path, root=root)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if path_keys:
        base = root or PROJECT_ROOT
        for keys in path_keys:
            _resolve_nested_path(cfg, keys, base)

    return cfg


def load_data_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = path or (PROJECT_ROOT / "configs" / "data.yaml")
    return load_config(config_path, path_keys=DATA_PATH_KEYS)


def _resolve_nested_path(cfg: dict[str, Any], keys: tuple[str, ...], root: Path) -> None:
    node: Any = cfg
    for key in keys[:-1]:
        node = node[key] # 进入下一层字典
    leaf = keys[-1]
    node[leaf] = str(resolve_path(node[leaf], root=root))
