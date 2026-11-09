"""生成一批假叶片图，让 B/C 在真实数据下载完之前就能开工。

    python scripts/make_dummy_data.py            # 默认每类 20 张
    python scripts/make_dummy_data.py --per-class 60

输出结构：data/raw/dummy/color/<类别名>/*.jpg
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


def draw_leaf(rng: np.random.Generator, size: int = 256) -> Image.Image:
    bg = rng.integers(60, 90)
    canvas = np.full((size, size, 3), bg, dtype=np.uint8)
    canvas += rng.integers(-8, 8, canvas.shape, dtype=np.int16).astype(np.uint8)
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    cx, cy = size / 2, size / 2
    rx = rng.uniform(60, 95)
    ry = rng.uniform(85, 115)
    angle = rng.uniform(0, 180)
    color = tuple(int(v) for v in rng.integers(40, 180, 3))
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
    img = img.rotate(angle, resample=Image.BILINEAR)
    if rng.random() < 0.7:  # 随机病斑
        draw = ImageDraw.Draw(img)
        for _ in range(int(rng.integers(3, 12))):
            x, y = rng.uniform(60, size - 60, 2)
            r = rng.uniform(2, 9)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(90, 70, 40))
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="生成假数据")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--out", default=str(ROOT / "data" / "raw" / "dummy"))
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes = cfg["data"]["classes"]
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    out_root = Path(args.out) / "color"
    for cls in classes:
        d = out_root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            draw_leaf(rng).save(d / f"dummy_{i:04d}.jpg", quality=92)
        print(f"  {cls}: {args.per_class}")
    print(f"\n假数据已写入 {out_root}")
    print("下一步: python scripts/prepare_data.py --root data/raw/dummy --out data/splits_dummy")


if __name__ == "__main__":
    main()
