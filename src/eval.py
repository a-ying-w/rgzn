"""评测入口：拿已训好的 best.pth 在指定 split / 扰动下重算指标（D 的 E3/E4 用）。

用法：
  python -m src.eval --config configs/base.yaml --ckpt runs/e0_resnet18_ft_r100_s42/best.pth
  python -m src.eval --config configs/base.yaml --ckpt runs/.../best.pth \\
      --corruption blur --severity 3 --exp-id e3 --run-name e3_resnet18_ft_r100_s42_blur3

产物：runs/<run_name>/eval.json，并 append 一行到 results.csv。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.config import load_with_overrides
from src.data.dataset import build_loaders
from src.models.build import build_model
from src.train import evaluate, set_seed
from src.utils.table import append_row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt", required=True, help="runs/<run_name>/best.pth")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--corruption", default=None, help="brightness | rotation | blur")
    ap.add_argument("--severity", type=int, default=0)
    ap.add_argument("--exp-id", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-append", action="store_true")
    args = ap.parse_args()

    cfg = load_with_overrides(args.config, args.set)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = list(ckpt["class_names"])
    num_classes = int(ckpt["num_classes"])

    model = build_model(
        ckpt["arch"], num_classes=num_classes, pretrained=bool(ckpt["pretrained"])
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

    corruption = args.corruption if args.severity > 0 else None
    loaders = build_loaders(
        cfg, corruption=corruption, severity=args.severity, splits=(args.split,)
    )
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loaders[args.split], criterion, device, cfg, class_names)

    exp_id = args.exp_id or cfg.get("exp_id") or ckpt_path.parent.name.split("_")[0]
    cond = "clean" if corruption is None else f"{corruption}{args.severity}"
    run_name = args.run_name or f"{ckpt_path.parent.name}_eval_{cond}"
    run_dir = Path(cfg.get("output_dir", "runs")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "run_name": run_name,
        "exp_id": exp_id,
        "ckpt": str(ckpt_path),
        "arch": ckpt["arch"],
        "pretrained": bool(ckpt["pretrained"]),
        "split": args.split,
        "corruption": "clean" if corruption is None else corruption,
        "severity": args.severity if corruption else 0,
        "class_names": class_names,
        "metrics": metrics,
        "cfg": cfg,
    }
    (run_dir / "eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[eval] {run_name} split={args.split} cond={cond} "
        f"acc={metrics['acc']:.4f} macroF1={metrics['macro_f1']:.4f}"
    )

    if not args.no_append:
        append_row(
            "results/results.csv",
            {
                "exp_id": exp_id,
                "model": ckpt["arch"],
                "pretrained": bool(ckpt["pretrained"]),
                "aug": cfg["train"].get("aug", "basic"),
                "seed": int(cfg.get("seed", 42)),
                "data_ratio": float(cfg["train"].get("subset_ratio", 1.0)),
                "split": args.split,
                "corruption": "clean" if corruption is None else corruption,
                "severity": args.severity if corruption else 0,
                "acc": round(metrics["acc"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "params_m": "",
                "notes": f"eval from {ckpt_path.parent.name}",
            },
        )


if __name__ == "__main__":
    main()
