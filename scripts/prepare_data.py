"""生成 data/splits/{train,val,test}.csv —— 全组唯一的划分文件。

用法（在仓库根目录）：
    python scripts/prepare_data.py
    python scripts/prepare_data.py --dedupe off --out data/splits_leaky   # 给报告的"泄漏对比"用
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.splits import (  # noqa: E402
    add_hash_groups,
    plot_distribution,
    scan_dataset,
    split_by_group,
    validate_splits,
    write_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成训练/验证/测试划分")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--root", default=None, help="PlantVillage 根目录或 color 目录")
    parser.add_argument("--out", default=None, help="划分文件输出目录")
    parser.add_argument("--dedupe", choices=["on", "off"], default="on", help="是否做近重复分组")
    parser.add_argument("--hash-dist", type=int, default=8, help="pHash hamming 阈值（越小越严格）")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes = cfg["data"]["classes"]
    root = args.root or cfg["data"]["root"]
    out = Path(args.out or cfg["data"]["split_dir"])
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)

    print(f"[1/4] 扫描数据: {root}")
    df, color_root = scan_dataset(root, classes)
    print(f"      共 {len(df)} 张图，{df['class_name'].nunique()} 类")

    print(f"[2/4] 近重复分组 (dedupe={args.dedupe}, max_dist={args.hash_dist})")
    df, dedupe_stats = add_hash_groups(
        df, max_dist=args.hash_dist, dedupe=(args.dedupe == "on")
    )
    if dedupe_stats.get("dedupe"):
        print(f"      {dedupe_stats['n_images']} 张图 -> {dedupe_stats['n_groups']} 组")
        if dedupe_stats["n_groups"] == dedupe_stats["n_images"]:
            print("      提示: 没有任何图被合并，阈值可能偏严，可试 --hash-dist 10~12")
    else:
        print("      已跳过，每张图自成一組（此划分会存在同叶泄漏，仅用于对比实验）")

    print(f"[3/4] 分层分组划分 (seed={seed}, val={args.val_ratio}, test={args.test_ratio})")
    df = split_by_group(df, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=seed)
    report = validate_splits(df)
    print(report["counts"].to_string())
    if not report["ok"]:
        raise SystemExit(f"划分校验失败: 泄漏组 {report['leaked_groups']}，空集合 {report['empty_splits']}")

    print(f"[4/4] 写出到 {out}")
    resolved = Path(color_root).resolve()
    try:
        root_str = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        root_str = str(resolved)  # 数据集在仓库外时只能记绝对路径
    write_splits(
        df,
        out,
        meta={
            "root": root_str,
            "seed": seed,
            "dedupe": dedupe_stats,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
    )
    plot_distribution(df, out / "class_distribution.png")
    for name in ("train", "val", "test"):
        n = int((df["split"] == name).sum())
        print(f"      {name}.csv: {n}")
    print(f"      meta.json / class_distribution.png 已生成")
    print("\n完成。请把 data/splits/*.csv 和 meta.json 提交到仓库并宣布冻结。")


if __name__ == "__main__":
    main()
