"""D 的出图脚本：从 results/results.csv 生成报告用图到 results/figures/。

用法：python scripts/make_figures.py
数据未跑齐时也能画：缺失的组用灰色占位条标出，方便先定版式。

E4 跨域对比图取数约定（三组的 test macro-F1）：
  ① color→color      : exp_id=e1b, split=test, corruption=clean
  ② color→segmented  : exp_id=e4a, split=test, corruption=clean
  ③ segmented→segmented: exp_id=e4b, split=test, corruption=clean
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "results.csv"
FIG_DIR = ROOT / "results" / "figures"

# Windows 中文字体，避免负号乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def get_macro_f1(df: pd.DataFrame, exp_id: str) -> float | None:
    """取某实验 test 集 clean 行的 macro-F1；多 seed 时取均值；没有则返回 None。"""
    hit = df[
        (df["exp_id"] == exp_id)
        & (df["split"] == "test")
        & (df["corruption"] == "clean")
    ]
    if hit.empty:
        return None
    return float(hit["macro_f1"].mean())


def fig_e4_cross_domain(df: pd.DataFrame) -> None:
    """E4：模型是否依赖背景。①③是同域上下界，②相对①的落差即背景依赖度。"""
    groups = [
        ("color→color\n(E1b 同域)", get_macro_f1(df, "e1b")),
        ("color→segmented\n(E4a 跨域)", get_macro_f1(df, "e4a")),
        ("segmented→segmented\n(E4b 上界)", get_macro_f1(df, "e4b")),
    ]
    labels = [g[0] for g in groups]
    values = [g[1] for g in groups]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    colors = ["#4C78A8" if v is not None else "#BBBBBB" for _, v in groups]
    bars = ax.bar(labels, [v if v is not None else 0 for v in values], color=colors, width=0.55)

    known = [v for v in values if v is not None]
    ymin = max(0.0, min(known) - 0.05) if known else 0.0
    ax.set_ylim(ymin, 1.02)

    for bar, v in zip(bars, values):
        if v is None:
            ax.text(bar.get_x() + bar.get_width() / 2, ymin + 0.01, "待跑",
                    ha="center", va="bottom", fontsize=11, color="#888888")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=11)

    # ②相对①的落差标注（两者都有数时）
    if values[0] is not None and values[1] is not None:
        drop = values[0] - values[1]
        ax.annotate("", xy=(1, values[1]), xytext=(1, values[0]),
                    arrowprops=dict(arrowstyle="<->", color="#E45756"))
        ax.text(1.08, (values[0] + values[1]) / 2, f"掉 {drop:.3f}",
                color="#E45756", fontsize=11, va="center")
    ax.set_ylabel("test macro-F1")
    ax.set_title("E4 跨域实验：模型在多大程度上依赖背景？")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "e4_cross_domain.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figure] {out.relative_to(ROOT)}")


def main() -> int:
    if not RESULTS.exists():
        print(f"缺少 {RESULTS}", file=sys.stderr)
        return 1
    df = pd.read_csv(RESULTS)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]  # 去 BOM
    fig_e4_cross_domain(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
