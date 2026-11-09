"""Dataset / DataLoader。全组统一从这里取数据，不要各自 ImageFolder。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .splits import resolve_color_root
from .transforms import build_corrupted_eval_transform, build_eval_transform, build_train_transform


class LeafDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        root: str | Path,
        classes: list[str],
        transform=None,
        subset_ratio: float = 1.0,
        seed: int = 42,
    ):
        df = pd.read_csv(split_csv)
        if subset_ratio < 1.0:
            df = stratified_subsample(df, subset_ratio, seed)
        self.df = df.reset_index(drop=True)
        self.root = resolve_color_root(root)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

        unknown = set(self.df["class_name"]) - set(self.classes)
        if unknown:
            raise ValueError(f"splits 中出现未在 classes 里声明的类别: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        img = Image.open(self.root / row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_to_idx[row["class_name"]]

    @property
    def labels(self) -> np.ndarray:
        return self.df["class_name"].map(self.class_to_idx).to_numpy()


def stratified_subsample(df: pd.DataFrame, ratio: float, seed: int = 42) -> pd.DataFrame:
    """E1b 用：每个类别按比例抽取，保证小数据子集仍然分层。"""
    rng = np.random.default_rng(seed)
    parts = []
    for _, sub in df.groupby("class_name"):
        k = max(1, int(round(len(sub) * ratio)))
        idx = np.sort(rng.choice(len(sub), size=min(k, len(sub)), replace=False))
        parts.append(sub.iloc[idx])
    return pd.concat(parts).sort_index()


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)


def build_datasets(cfg: dict, corruption: str | None = None, severity: int = 0) -> dict:
    dcfg = cfg["data"]
    common = dict(
        image_size=dcfg["image_size"],
        resize_size=dcfg["resize_size"],
        mean=dcfg.get("mean"),
        std=dcfg.get("std"),
    )
    if corruption:
        eval_tf = build_corrupted_eval_transform(corruption, severity, **common)
    else:
        eval_tf = build_eval_transform(**common)
    train_tf = build_train_transform(cfg["train"].get("aug", "basic"), **common)

    split_dir = Path(dcfg["split_dir"])
    root = dcfg["root"]
    classes = dcfg["classes"]
    return {
        "train": LeafDataset(
            split_dir / "train.csv",
            root,
            classes,
            transform=train_tf,
            subset_ratio=cfg["train"].get("subset_ratio", 1.0),
            seed=cfg.get("seed", 42),
        ),
        "val": LeafDataset(split_dir / "val.csv", root, classes, transform=eval_tf),
        "test": LeafDataset(split_dir / "test.csv", root, classes, transform=eval_tf),
    }


def build_loaders(
    cfg: dict,
    corruption: str | None = None,
    severity: int = 0,
    splits: tuple[str, ...] = ("val", "test"),
) -> dict[str, DataLoader]:
    datasets = build_datasets(cfg, corruption=corruption, severity=severity)
    dcfg = cfg["data"]
    loaders = {}
    for split in splits:
        is_train = split == "train"
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=cfg["train"]["batch_size"],
            shuffle=is_train,
            num_workers=dcfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=False,
            worker_init_fn=_worker_init,
            persistent_workers=dcfg.get("num_workers", 4) > 0,
        )
    return loaders
