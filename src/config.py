from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} 不是合法的 yaml 配置")
    return cfg


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """支持 --set train.epochs=5 model.name=mobilenet_v3_small 形式的临时覆盖。"""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"覆盖项需要 key=value 形式，收到 {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node:
                raise KeyError(f"配置里没有 {part!r}（来自 {key}）")
            node = node[part]
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_with_overrides(path: str | Path, overrides: list[str] | None = None) -> dict:
    return apply_overrides(load_config(path), overrides)
