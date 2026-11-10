"""C 岗模型效率对比：参数量 / FLOPs / 单张推理延迟。

    python scripts/profile_models.py
    python scripts/profile_models.py --models mobilenet_v3_small,resnet18 --device cpu

产出 results/model_cost.csv，可直接贴进报告的"模型效率"小节。
FLOPs 用 torch.utils.flop_counter 统计（torch>=2.2），不额外装 thop/fvcore。
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.models.build import available_models, build_model, count_parameters  # noqa: E402


def measure_flops(model: torch.nn.Module, image_size: int) -> float:
    """单张 3×H×W 前向的 FLOPs（单位 G）。"""
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    return counter.get_total_flops() / 1e9


def measure_latency(model: torch.nn.Module, device: torch.device, image_size: int, warmup: int, iters: int):
    x = torch.randn(1, 3, image_size, image_size, device=device)
    model.to(device).eval()
    sync = device.type == "cuda"
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if sync:
            torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            if sync:
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(x)
            if sync:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    mean = statistics.fmean(times)
    return mean, statistics.pstdev(times), statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser(description="参数量 / FLOPs / 延迟对比")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--models", default=",".join(available_models()))
    parser.add_argument("--device", default=None, help="cpu / cuda / 默认自动")
    parser.add_argument("--image-size", type=int, default=None, help="默认取 config 的 data.image_size")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--out", default="results/model_cost.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_size = args.image_size or cfg["data"]["image_size"]
    num_classes = len(cfg["data"]["classes"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "cpu"

    rows = []
    for name in [n.strip() for n in args.models.split(",") if n.strip()]:
        try:
            model = build_model(name, num_classes=num_classes, pretrained=False)
        except (ImportError, KeyError, ValueError) as error:
            print(f"[SKIP] {name}: {error}")
            continue

        params_m = count_parameters(model) / 1e6
        flops_g = measure_flops(model, image_size)
        mean, std, p50 = measure_latency(model, device, image_size, args.warmup, args.iters)
        rows.append(
            {
                "model": name,
                "params_m": round(params_m, 3),
                "flops_g": round(flops_g, 3),
                "latency_ms_mean": round(mean, 3),
                "latency_ms_std": round(std, 3),
                "latency_ms_p50": round(p50, 3),
                "fps": round(1000 / mean, 1) if mean > 0 else float("nan"),
                "device": f"{device.type} / {device_name}",
                "batch_size": 1,
                "notes": f"torch {torch.__version__}, threads={torch.get_num_threads()}",
            }
        )
        print(
            f"{name:20s} params={params_m:7.2f}M  flops={flops_g:6.3f}G  "
            f"latency={mean:7.2f}±{std:.2f}ms  fps={1000 / mean:7.1f}"
        )

    if not rows:
        print("没有成功构建任何模型")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")

    print("\n| 模型 | 参数量(M) | FLOPs(G) | 延迟(ms, mean±std) | FPS |")
    print("|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['model']} | {row['params_m']} | {row['flops_g']} | "
            f"{row['latency_ms_mean']}±{row['latency_ms_std']} | {row['fps']} |"
        )
    print(f"\n设备: {device.type} / {device_name}；图像 {image_size}×{image_size}，batch=1")
    print(f"已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
