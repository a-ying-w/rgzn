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


def fig_e3_robustness(df: pd.DataFrame) -> None:
    """E3：4 模型 × 3 扰动 × 5 档严重度的鲁棒性曲线。

    results.csv 中 E3 行的 notes 形如 "eval from e1b_resnet18_ft_r100_s42 variant=color"，
    从中解析 base 模型名（e1a/e1b/e1c/e1d）。
    取数约定：exp_id=e3, split=test, severity∈[1..5]。
    """
    import re

    e3 = df[(df["exp_id"] == "e3") & (df["split"] == "test")].copy()
    if e3.empty:
        print("[figure] e3 暂无数据，跳过鲁棒性曲线")
        return

    # 从 notes 解析 base 模型："eval from e1b_resnet18_ft_r100_s42 variant=color"
    def parse_base(notes) -> str | None:
        if not isinstance(notes, str):
            return None
        m = re.search(r"eval from (e1[abcd])_", notes)
        return m.group(1) if m else None

    e3["base"] = e3["notes"].apply(parse_base)
    e3 = e3[e3["base"].notna()]
    if e3.empty:
        print("[figure] e3 行无法解析 base 模型，跳过鲁棒性曲线")
        return

    # base → 展示标签
    BASE_LABEL = {
        "e1a": "resnet18-scratch",
        "e1b": "resnet18-ft",
        "e1c": "cnn-scratch",
        "e1d": "resnet18-ft-cw",
    }
    CORRS = ["brightness", "rotation", "blur"]
    SEVERITIES = [1, 2, 3, 4, 5]
    PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]  # 4 模型

    # 每个扰动一张子图，3 张横排
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150, sharey=True)
    fig.suptitle("E3 鲁棒性：随扰动严重度下降的 macro-F1", fontsize=13, y=1.02)

    for ax, corr in zip(axes, CORRS):
        for idx, (base, label) in enumerate(BASE_LABEL.items()):
            sub = e3[(e3["base"] == base) & (e3["corruption"] == corr)]
            if sub.empty:
                continue
            # severity → macro_f1 的有序点
            points = []
            for sev in SEVERITIES:
                hit = sub[sub["severity"] == sev]
                if not hit.empty:
                    points.append((sev, float(hit["macro_f1"].mean())))
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", label=label, color=PALETTE[idx], linewidth=2)

        ax.set_xlabel("severity")
        ax.set_xticks(SEVERITIES)
        ax.set_title(f"扰动: {corr}")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("test macro-F1")
    # 把 legend 放在中间子图顶部外
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False)

    known_f1s = e3["macro_f1"].dropna()
    if not known_f1s.empty:
        axes[0].set_ylim(max(0.0, known_f1s.min() - 0.05), 1.02)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "e3_robustness_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {out.relative_to(ROOT)}")


def main() -> int:
    if not RESULTS.exists():
        print(f"缺少 {RESULTS}", file=sys.stderr)
        return 1
    df = pd.read_csv(RESULTS)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]  # 去 BOM
    fig_e4_cross_domain(df)
    fig_e3_robustness(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
