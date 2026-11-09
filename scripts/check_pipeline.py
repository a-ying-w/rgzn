"""A 岗验收脚本：一条命令检查数据管线是否可用。

    python scripts/check_pipeline.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.corruptions import CORRUPTIONS, SEVERITY_MAP, apply_corruption  # noqa: E402
from src.data.splits import resolve_color_root  # noqa: E402
from src.utils.metrics import summarize  # noqa: E402

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((ok, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="检查数据管线")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--splits", default=None, help="覆盖 data.split_dir，例如 data/splits_dummy")
    parser.add_argument("--root", default=None, help="覆盖 data.root")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    if args.splits:
        dcfg["split_dir"] = args.splits
    if args.root:
        dcfg["root"] = args.root
    split_dir = Path(dcfg["split_dir"])
    root = resolve_color_root(dcfg["root"])

    # 1. 划分文件
    frames = {}
    for split in ("train", "val", "test"):
        p = split_dir / f"{split}.csv"
        if not p.exists():
            check(False, "划分文件存在", f"缺少 {p}，先跑 scripts/prepare_data.py")
            return 1
        frames[split] = pd.read_csv(p)
    df = pd.concat([f.assign(split=s) for s, f in frames.items()], ignore_index=True)
    check(True, "划分文件存在", f"train/val/test = {len(frames['train'])}/{len(frames['val'])}/{len(frames['test'])}")

    # 2. 字段与类别
    need = {"path", "class_name", "group_id"}
    check(need <= set(df.columns), "划分字段完整", str(sorted(set(df.columns))))
    classes = set(dcfg["classes"])
    unknown = set(df["class_name"]) - classes
    check(not unknown, "类别与配置一致", f"多余的类: {sorted(unknown)}" if unknown else f"{len(classes)} 类")

    # 3. 近重复泄漏
    leaks = int((df.groupby("group_id")["split"].nunique() > 1).sum())
    check(leaks == 0, "无近重复跨集合泄漏", f"{leaks} 个 group 跨集合" if leaks else f"{df['group_id'].nunique()} 个 group")

    # 4. 图片可读
    missing = [p for p in df["path"] if not (root / p).exists()]
    check(not missing, "图片文件存在", f"缺失 {len(missing)} 张，例如 {missing[:2]}" if missing else f"{len(df)} 张")

    sample = df.sample(min(30, len(df)), random_state=0)["path"].tolist()
    bad = []
    for p in sample:
        try:
            with Image.open(root / p) as im:
                if im.size[0] < 32:
                    bad.append(p)
        except Exception as error:  # noqa: BLE001
            bad.append(f"{p}: {error}")
    check(not bad, "抽样图片可解码", str(bad[:2]) if bad else f"{len(sample)} 张")

    # 5. 扰动模块
    img = Image.open(root / sample[0]).convert("RGB")
    base = np.asarray(img, dtype=np.int16)
    detail = []
    all_ok = True
    for name in CORRUPTIONS:
        changed = 0
        for sev in SEVERITY_MAP[name]:
            out = np.asarray(apply_corruption(img, name, sev), dtype=np.int16)
            if out.shape == base.shape and not np.array_equal(out, base):
                changed += 1
        all_ok &= changed == 5
        detail.append(f"{name}={changed}/5")
    check(all_ok, "扰动 severity 1-5 全部生效", ", ".join(detail))

    # 6. 指标
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 2, 2, 2]
    m = summarize(y_true, y_pred, 3, ["a", "b", "c"])
    check(abs(m["acc"] - 5 / 6) < 1e-6, "metrics 正确", f"acc={m['acc']:.3f} f1={m['macro_f1']:.3f}")

    # 7. torch 相关（本地没装 torch 时跳过）
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[SKIP] 未安装 torch：Dataset / DataLoader / 增强未验证（在 Kaggle 上跑这一步）")
    else:
        from src.data.dataset import build_loaders

        local_cfg = copy.deepcopy(cfg)
        local_cfg["data"]["num_workers"] = 0
        try:
            loaders = build_loaders(local_cfg, splits=("val",))
            x, y = next(iter(loaders["val"]))
            check(
                tuple(x.shape[1:]) == (3, dcfg["image_size"], dcfg["image_size"]),
                "Dataset / 增强 / DataLoader",
                f"batch={tuple(x.shape)} labels={y[:4].tolist()}",
            )
        except Exception as error:  # noqa: BLE001
            check(False, "Dataset / 增强 / DataLoader", str(error))

    failed = [r for r in results if not r[0]]
    print(f"\n{len(results) - len(failed)}/{len(results)} 项通过")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
