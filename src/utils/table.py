"""results.csv 的读写与汇总。

长表格式，一行 = 一次评测（模型 × seed × 数据比例 × 测试条件）。
这样 E3 的鲁棒性曲线可以直接从这张表画出来，不需要再加列。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = [
    "exp_id",
    "model",
    "pretrained",
    "aug",
    "seed",
    "data_ratio",
    "split",
    "corruption",
    "severity",
    "acc",
    "macro_f1",
    "params_m",
    "notes",
]

GROUP_KEYS = ["model", "pretrained", "aug", "data_ratio", "split", "corruption", "severity"]


def append_row(path: str | Path, row: dict) -> None:
    path = Path(path)
    missing = set(COLUMNS) - set(row)
    if missing:
        raise ValueError(f"结果行缺少字段: {sorted(missing)}")
    df = pd.DataFrame([{k: row.get(k) for k in COLUMNS}])
    header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")


def load_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def summarize(path: str | Path, by: list[str] | None = None) -> pd.DataFrame:
    """按实验设置聚合多个 seed，输出 mean ± std。"""
    df = load_results(path)
    if df.empty:
        return df
    by = by or GROUP_KEYS
    agg = (
        df.groupby(by, dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            f1_mean=("macro_f1", "mean"),
            f1_std=("macro_f1", "std"),
        )
        .reset_index()
        .sort_values(by)
    )
    return agg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="汇总 results.csv")
    parser.add_argument("--path", default="results/results.csv")
    parser.add_argument("--out", default="results/summary.csv")
    args = parser.parse_args()

    table = summarize(args.path)
    if table.empty:
        print("results.csv 还没有数据")
    else:
        table.to_csv(args.out, index=False, encoding="utf-8")
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(table.to_string(index=False))
        print(f"\n已写出 {args.out}")
