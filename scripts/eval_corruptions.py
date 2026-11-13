"""E3 扰动鲁棒性批量评测：3 个模型 × 3 种扰动 × 5 档严重度 = 45 次 eval。

用法：
  # 先把 B 给的 4 个 best.pth 放进 runs/<exp_name>/best.pth，或改下面的 CKPT_BUNDLES 字典
  python scripts/eval_corruptions.py

  # 只跑某个模型做冒烟
  python scripts/eval_corruptions.py --only e1b

  # dry-run 预览命令不执行
  python scripts/eval_corruptions.py --dry-run

评测结果自动 append 到 results/results.csv（走 eval.py 自带的 append_row）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 每个模型的配置文件 + checkpoint 路径
# B 的 4 个模型：e1a(scratch) / e1b(ft) / e1c(cnn) / e1d(cw)
# ckpt 路径按 runs/<run_name>/best.pth 约定放好即可
CKPT_BUNDLES: dict[str, dict[str, str]] = {
    "e1a": {
        "label": "resnet18-scratch",
        "config": "configs/e1_resnet18_scratch.yaml",
        "ckpt": "runs/e1a_resnet18_scratch_r100_s42/best.pth",
    },
    "e1b": {
        "label": "resnet18-ft",
        "config": "configs/e1_resnet18_ft.yaml",
        "ckpt": "runs/e1b_resnet18_ft_r100_s42/best.pth",
    },
    "e1c": {
        "label": "cnn-scratch",
        "config": "configs/e1_cnn_scratch.yaml",
        "ckpt": "runs/e1c_cnn_scratch_r100_s42/best.pth",
    },
    "e1d": {
        "label": "resnet18-ft-cw",
        "config": "configs/e1_resnet18_ft_cw.yaml",
        "ckpt": "runs/e1d_resnet18_ft_cw_r100_s42/best.pth",
    },
}

# 扰动类型 × 严重度档位（与 data_and_setup.md 一致：5 档）
CORRUPTIONS = ["brightness", "rotation", "blur"]
SEVERITIES = [1, 2, 3, 4, 5]


def build_cmd(bundle_key: str, corruption: str, severity: int) -> list[str]:
    b = CKPT_BUNDLES[bundle_key]
    return [
        sys.executable, "-m", "src.eval",
        "--config", str(ROOT / b["config"]),
        "--ckpt", str(ROOT / b["ckpt"]),
        "--split", "test",
        "--corruption", corruption,
        "--severity", str(severity),
        "--exp-id", "e3",
        "--run-name", f"e3_{bundle_key}_{corruption}{severity}",
    ]


def run_batch(only: str | None, dry_run: bool) -> int:
    targets = {only: CKPT_BUNDLES[only]} if only else CKPT_BUNDLES
    total = len(targets) * len(CORRUPTIONS) * len(SEVERITIES)
    done = 0
    failed = 0

    for bkey in targets:
        b = CKPT_BUNDLES[bkey]
        ckpt_path = ROOT / b["ckpt"]
        if not ckpt_path.exists():
            print(f"[skip] {bkey}: ckpt 不存在 {ckpt_path}")
            continue

        for corr in CORRUPTIONS:
            for sev in SEVERITIES:
                cmd = build_cmd(bkey, corr, sev)
                done += 1
                tag = f"[{done}/{total}] {bkey:4s} {corr:10s} sev={sev}"
                if dry_run:
                    print(f"[dry-run] {tag}")
                    print("   ", " ".join(cmd))
                    continue

                print(f"\n{tag}")
                env = dict(__import__("os").environ)
                env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                proc = subprocess.run(cmd, cwd=ROOT, env=env)
                if proc.returncode != 0:
                    failed += 1
                    print(f"   FAILED (exit {proc.returncode})")

    print(f"\n=== 完成：{done - failed}/{done} 成功，{failed} 失败 ===")
    return 0 if failed == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(CKPT_BUNDLES), help="只跑某一个模型做冒烟")
    ap.add_argument("--dry-run", action="store_true", help="预览命令不执行")
    args = ap.parse_args()

    # 确认仓库根目录
    if not (ROOT / "src" / "eval.py").exists():
        print(f"找不到 src/eval.py，ROOT={ROOT}", file=sys.stderr)
        return 1
    return run_batch(args.only, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
