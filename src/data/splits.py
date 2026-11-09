"""数据扫描、近重复分组、分层分组划分。

产出的 splits 是三份只读的 csv，全组共用，任何人不得自行改动：
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv
列: path, class_name, group_id     （path 相对于 color/ 目录）
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .dedupe import assign_group_ids, phash

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_data_root(root: str | Path, variant: str = "color") -> Path:
    """兼容三种传法：

    - spMohanty/PlantVillage-Dataset 仓库根目录（含 raw/color/）
    - raw/ 目录（含 color/）
    - color/ 目录本身
    """
    p = Path(root)
    for rel in (variant, f"raw/{variant}"):
        if (p / rel).is_dir():
            return p / rel
    return p


def resolve_color_root(root: str | Path) -> Path:
    return resolve_data_root(root, "color")


def scan_dataset(
    root: str | Path, classes: list[str], variant: str = "color"
) -> tuple[pd.DataFrame, Path]:
    color_root = resolve_data_root(root, variant)
    rows, missing = [], []
    for cls in classes:
        d = color_root / cls
        if not d.is_dir():
            missing.append(cls)
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                rows.append(
                    {
                        "path": f.relative_to(color_root).as_posix(),
                        "class_name": cls,
                        "abs_path": str(f),
                    }
                )
    if missing:
        available = sorted(d.name for d in color_root.iterdir() if d.is_dir())
        raise FileNotFoundError(
            "以下类别目录不存在：\n  "
            + "\n  ".join(missing)
            + f"\n\n当前 {color_root} 下可用的类别（{len(available)} 个）：\n  "
            + "\n  ".join(available)
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"{color_root} 下没有找到任何图片")
    return df, color_root


def add_hash_groups(
    df: pd.DataFrame,
    max_dist: int = 8,
    dedupe: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    if not dedupe:
        df["group_id"] = [f"{c}_single_{i}" for i, c in enumerate(df["class_name"])]
        return df, {"dedupe": False, "n_images": int(len(df)), "n_groups": int(len(df))}

    hashes = []
    for i, p in enumerate(df["abs_path"]):
        with Image.open(p) as im:
            hashes.append(phash(im))
        if verbose and (i + 1) % 2000 == 0:
            print(f"  pHash {i + 1}/{len(df)}")
    df["phash"] = hashes

    mapping = assign_group_ids(
        df[["path", "class_name", "phash"]].to_dict("records"), max_dist=max_dist
    )
    df["group_id"] = df["path"].map(mapping)
    stats = {
        "dedupe": True,
        "max_dist": max_dist,
        "n_images": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
        "n_merged": int(len(df) - df["group_id"].nunique()),
    }
    return df, stats


def split_by_group(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """每个类别内部按 group 整体划分，先切 test 再切 val，剩余全部为 train。

    以 group 为单位贪心累加，能保证同一 group 不跨集合（避免同叶多拍泄漏）。
    """
    rng = np.random.default_rng(seed)
    split_of_group: dict[str, str] = {}

    for _, sub in df.groupby("class_name", sort=True):
        sizes = sub.groupby("group_id").size()
        names = sizes.index.to_numpy()
        counts = sizes.to_numpy()
        n = int(counts.sum())
        target_test = int(round(n * test_ratio))
        target_val = int(round(n * val_ratio))
        seen_test = seen_val = 0

        for k in rng.permutation(len(names)):
            gid, cnt = names[k], int(counts[k])
            if seen_test < target_test:
                split_of_group[gid] = "test"
                seen_test += cnt
            elif seen_val < target_val:
                split_of_group[gid] = "val"
                seen_val += cnt
            else:
                split_of_group[gid] = "train"

    df = df.copy()
    df["split"] = df["group_id"].map(split_of_group)
    return df


def validate_splits(df: pd.DataFrame) -> dict:
    """返回检查结果；group 跨集合会被判定为失败。"""
    leaks = df.groupby("group_id")["split"].nunique()
    n_leak = int((leaks > 1).sum())
    counts = df.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    empty_split = [s for s in ("train", "val", "test") if (df["split"] == s).sum() == 0]
    return {
        "n_images": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
        "leaked_groups": n_leak,
        "empty_splits": empty_split,
        "counts": counts,
        "ok": n_leak == 0 and not empty_split,
    }


def write_splits(df: pd.DataFrame, out_dir: str | Path, meta: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["path", "class_name", "group_id"]
    for split in ("train", "val", "test"):
        sub = df.loc[df["split"] == split, cols].sort_values("path")
        sub.to_csv(out / f"{split}.csv", index=False, encoding="utf-8")

    info = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "classes": sorted(df["class_name"].unique().tolist()),
        "n_images": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
        "split_sizes": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
        "per_class": df.groupby(["class_name"]).size().to_dict(),
    }
    if meta:
        info.update(meta)
    info["sha256"] = {
        s: hashlib.sha256((out / f"{s}.csv").read_bytes()).hexdigest()
        for s in ("train", "val", "test")
    }
    (out / "meta.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


def plot_distribution(df: pd.DataFrame, out_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = df.groupby(["class_name", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["train", "val", "test"], fill_value=0)
    ax = counts.plot(kind="bar", stacked=True, figsize=(9, 4.5), width=0.7)
    ax.set_ylabel("images")
    ax.set_title("Split distribution (grouped split)")
    plt.xticks(rotation=25, ha="right", fontsize=8)
    ax.legend(title="split", fontsize=8)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
