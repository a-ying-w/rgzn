"""D 的出图脚本：从 results/results.csv 生成报告用图到 results/figures/。

用法：python scripts/make_figures.py
数据未跑齐时也能画：缺失的组用灰色占位条标出，方便先定版式。

E4 跨域对比图取数约定（三组的 test macro-F1）：
  ① color→color      : exp_id=e1b, split=test, corruption=clean
  ② color→segmented  : exp_id=e4a, split=test, corruption=clean
  ③ segmented→segmented: exp_id=e4b, split=test, corruption=clean
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "results.csv"
FIG_DIR = ROOT / "results" / "figures"
E4A_EVAL = ROOT / "runs" / "e4a_color2seg_resnet18_ft_s42" / "eval.json"

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


CLASS_SHORT = {
    "Tomato___Bacterial_spot": "Bacterial spot",
    "Tomato___Early_blight": "Early blight",
    "Tomato___Late_blight": "Late blight",
    "Tomato___Leaf_Mold": "Leaf mold",
    "Tomato___Septoria_leaf_spot": "Septoria spot",
    "Tomato___healthy": "Healthy",
}


def fig_e4a_confusion() -> None:
    """E4a 跨域混淆矩阵：按行归一化着色，格内标原始样本数。

    取数：runs/e4a_color2seg_resnet18_ft_s42/eval.json 的 metrics.confusion_matrix。
    """
    if not E4A_EVAL.exists():
        print(f"[figure] 跳过 E4a 混淆矩阵：缺少 {E4A_EVAL.relative_to(ROOT)}")
        return
    data = json.loads(E4A_EVAL.read_text(encoding="utf-8"))
    classes = [CLASS_SHORT.get(c, c) for c in data["class_names"]]
    cm = np.asarray(data["metrics"]["confusion_matrix"], dtype=float)
    row = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7.4, 5.8), dpi=150)
    im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=32, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=8,
                    color="white" if row[i, j] > 0.55 else "#333333")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("E4a 跨域混淆矩阵（color→segmented，行归一化，格内为样本数）")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="行内占比")
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "e4a_confusion_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figure] {out.relative_to(ROOT)}")


def get_macro_f1_pt(df: pd.DataFrame, exp_id: str, pretrained: bool) -> float | None:
    """取指定 exp_id + 是否预训练 的 test clean macro-F1；多 seed 取均值；没有则返回 None。"""
    pt = df["pretrained"].astype(str).str.lower() == "true"
    hit = df[
        (df["exp_id"] == exp_id)
        & (pt == pretrained)
        & (df["split"] == "test")
        & (df["corruption"] == "clean")
    ]
    if hit.empty:
        return None
    return float(hit["macro_f1"].mean())


def fig_e2_aug(df: pd.DataFrame) -> None:
    """E2：none/basic/strong 三档增强 × {微调, 从零} 的 test macro-F1。

    basic 臂直接复用既有结果（微调=e1b，从零=e1a），其余两档取 e2a/e2c。
    """
    specs = [
        ("none", "e2a", "e2a"),
        ("basic", "e1b", "e1a"),
        ("strong", "e2c", "e2c"),
    ]
    labels = [s[0] for s in specs]
    ft = [get_macro_f1_pt(df, s[1], True) for s in specs]
    sc = [get_macro_f1_pt(df, s[2], False) for s in specs]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    x = range(len(labels))
    w = 0.36
    bars_ft = ax.bar([i - w / 2 for i in x], [v if v is not None else 0 for v in ft],
                     width=w, color="#4C78A8", label="预训练微调")
    bars_sc = ax.bar([i + w / 2 for i in x], [v if v is not None else 0 for v in sc],
                     width=w, color="#F58518", label="从零训练")

    known = [v for v in ft + sc if v is not None]
    ymin = max(0.0, min(known) - 0.02) if known else 0.0
    ax.set_ylim(ymin, 1.005)

    for bars, vals in ((bars_ft, ft), (bars_sc, sc)):
        for bar, v in zip(bars, vals):
            if v is None:
                ax.text(bar.get_x() + bar.get_width() / 2, ymin + 0.002, "待跑",
                        ha="center", va="bottom", fontsize=9, color="#888888")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0015, f"{v:.4f}",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("增强档位")
    ax.set_ylabel("test macro-F1")
    ax.set_title("E2 数据增强消融：干净测试集上的精度")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "e2_aug_ablation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[figure] {out.relative_to(ROOT)}")


CORRUPTION_CN = {
    "brightness": "亮度（0.90→0.30）",
    "rotation": "旋转（10°→60°）",
    "blur": "模糊（0.6→3.2）",
}

MODEL_CN = {
    "e1a": "resnet18-从零",
    "e1b": "resnet18-微调(basic)",
    "e1c": "自建CNN",
    "e1d": "resnet18-微调+类权重",
    "e2c": "resnet18-微调(strong)",
}


def fig_e3_robustness(df: pd.DataFrame) -> None:
    """E3：三种扰动 × 5 档严重度下的 test macro-F1 曲线（每个模型一条）。"""
    d = df[
        df["exp_id"].astype(str).str.match(r"^e3[a-z0-9_]*$")
        & (df["split"] == "test")
        & (df["corruption"].isin(CORRUPTION_CN))
    ].copy()
    if d.empty:
        print("[figure] 跳过 E3：results.csv 里还没有 E3 行")
        return

    # 模型来源：从 notes 里的 "eval from e1x_" 或 "eval from e2c_" 提取
    src = d["notes"].astype(str).str.extract(r"eval from (e[12][a-z0-9]+)_")[0]
    fallback = d["exp_id"].astype(str).str.extract(r"^e3([a-z0-9]+)$")[0]
    d["src"] = src.fillna(fallback).fillna("")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150, sharey=True)
    for ax, corr in zip(axes, CORRUPTION_CN):
        sub = d[d["corruption"] == corr]
        for src_key in sorted(sub["src"].unique()):
            g = sub[sub["src"] == src_key].groupby("severity")["macro_f1"].mean().sort_index()
            ax.plot(g.index, g.values, marker="o", label=MODEL_CN.get(src_key, src_key))
        ax.set_title(CORRUPTION_CN[corr], fontsize=11)
        ax.set_xlabel("扰动严重度")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("test macro-F1")
    axes[0].set_ylim(0, 1.02)
    axes[-1].legend(fontsize=9, frameon=False)
    fig.suptitle("E3 扰动鲁棒性：模型在不同退化强度下的表现", fontsize=12)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "e3_robustness_curves.png"
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
    fig_e4a_confusion()
    fig_e3_robustness(df)
    fig_e2_aug(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
