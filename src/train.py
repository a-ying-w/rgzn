"""训练入口（README §5 约定）。

用法：
  python -m src.train --config configs/e0_resnet18_ft.yaml
  python -m src.train --config configs/base.yaml --set train.epochs=30 --seed 0 --run-name e1_resnet18_ft_s0

产物：runs/<run_name>/best.pth + metrics.json，并自动 append val / test 两行到 results.csv。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import load_with_overrides
from src.data.dataset import build_loaders
from src.models.build import build_model
from src.utils.metrics import (
    confusion_matrix,
    per_class_precision,
    per_class_recall,
    summarize,
)
from src.utils.table import append_row


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """backbone / head 分组学习率；BN 与 bias 不进 weight_decay；冻结参数不进 optimizer。"""
    tcfg = cfg["train"]
    wd_all = float(tcfg.get("weight_decay", 0.0))
    buckets: dict[tuple, list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "backbone" if name.startswith("backbone.") else "head"
        wd = 0.0 if p.ndim <= 1 else wd_all
        lr = float(tcfg["lr_backbone"] if part == "backbone" else tcfg["lr_head"])
        buckets.setdefault((part, wd, lr), []).append(p)
    if not buckets:
        raise RuntimeError("没有可训练参数：检查 freeze_backbone 与模型结构")
    groups = [
        {"params": ps, "lr": lr, "weight_decay": wd, "tag": part}
        for (part, wd, lr), ps in buckets.items()
    ]
    return torch.optim.AdamW(groups)


def evaluate(model, loader, criterion, device, cfg, class_names) -> dict:
    """返回 A 的 summarize 指标 + 损失 + 逐类指标 + 混淆矩阵。"""
    amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    model.eval()
    total, n = 0.0, 0
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            total += float(loss) * y.size(0)
            n += y.size(0)
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())

    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(preds).numpy()
    num_classes = len(class_names)
    out = summarize(y_true, y_pred, num_classes, class_names)
    out["loss"] = total / max(n, 1)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    out["per_class_recall"] = {
        class_names[i]: float(r) for i, r in enumerate(per_class_recall(cm))
    }
    out["per_class_precision"] = {
        class_names[i]: float(p) for i, p in enumerate(per_class_precision(cm))
    }
    out["confusion_matrix"] = cm.tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="临时覆盖，如 train.epochs=5")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-append", action="store_true")
    args = ap.parse_args()

    cfg = load_with_overrides(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    tcfg, mcfg = cfg["train"], cfg["model"]
    class_names = list(cfg["data"]["classes"])
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_id = cfg.get("exp_id") or Path(args.config).stem
    ratio = float(tcfg.get("subset_ratio", 1.0))
    tag = "ft" if mcfg["pretrained"] else "scratch"
    run_name = args.run_name or f"{exp_id}_{mcfg['name']}_{tag}_r{int(round(ratio * 100))}_s{seed}"
    run_dir = Path(cfg.get("output_dir", "runs")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_loaders(cfg, splits=("train", "val", "test"))
    model = build_model(
        mcfg["name"], num_classes=num_classes, pretrained=bool(mcfg["pretrained"])
    ).to(device)

    # README：pretrained=false 时 freeze_backbone 忽略
    freeze = bool(tcfg.get("freeze_backbone", False)) and bool(mcfg["pretrained"])
    if freeze:
        for p in model.backbone.parameters():
            p.requires_grad = False

    weight = None
    if str(tcfg.get("class_weight", "none")).lower() == "balanced":
        counts = np.bincount(loaders["train"].dataset.labels, minlength=num_classes)
        weight = torch.tensor(
            counts.sum() / (num_classes * np.clip(counts, 1, None)), dtype=torch.float32, device=device
        )
    criterion = nn.CrossEntropyLoss(
        weight=weight, label_smoothing=float(tcfg.get("label_smoothing", 0.0))
    )

    optimizer = build_optimizer(model, cfg)
    epochs = int(tcfg["epochs"])
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if str(tcfg.get("scheduler", "cosine")).lower() == "cosine"
        else None
    )
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    patience = int(tcfg.get("patience", 0))
    monitor = str(tcfg.get("monitor", "macro_f1"))

    params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    print(
        f"[run] {run_name} device={device} params={params_m}M pretrained={mcfg['pretrained']} "
        f"freeze={freeze} train/val/test={len(loaders['train'].dataset)}/"
        f"{len(loaders['val'].dataset)}/{len(loaders['test'].dataset)}"
    )

    history, best, best_epoch, stale = [], -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        if freeze:
            model.backbone.eval()  # 冻结时 BN 必须用 running stats

        tr_total, tr_n = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_total += float(loss) * y.size(0)
            tr_n += y.size(0)

        val = evaluate(model, loaders["val"], criterion, device, cfg, class_names)
        if scheduler is not None:
            scheduler.step()

        lrs = {g.get("tag", "?"): g["lr"] for g in optimizer.param_groups}
        history.append(
            {
                "epoch": epoch,
                "lr_backbone": lrs.get("backbone", 0.0),
                "lr_head": lrs.get("head", 0.0),
                "train_loss": tr_total / max(tr_n, 1),
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "val_macro_f1": val["macro_f1"],
            }
        )
        print(
            f"[{epoch:02d}/{epochs}] loss={tr_total / max(tr_n, 1):.4f} val_loss={val['loss']:.4f} "
            f"acc={val['acc']:.4f} macroF1={val['macro_f1']:.4f} ({time.time() - t0:.0f}s)"
        )

        if val[monitor] > best:
            best, best_epoch, stale = val[monitor], epoch, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": mcfg["name"],
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "pretrained": bool(mcfg["pretrained"]),
                    "epoch": epoch,
                    "val_metrics": val,
                    "cfg": cfg,
                },
                run_dir / "best.pth",
            )
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                print(f"[stop] 早停于 epoch {epoch}，best {monitor}={best:.4f} @ {best_epoch}")
                break

    ckpt = torch.load(run_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    val_best = evaluate(model, loaders["val"], criterion, device, cfg, class_names)
    test = evaluate(model, loaders["test"], criterion, device, cfg, class_names)

    metrics = {
        "run_name": run_name,
        "exp_id": exp_id,
        "arch": mcfg["name"],
        "pretrained": bool(mcfg["pretrained"]),
        "freeze_backbone": freeze,
        "class_weight": str(tcfg.get("class_weight", "none")),
        "aug": tcfg.get("aug", "basic"),
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "monitor": monitor,
        "params_m": params_m,
        "class_names": class_names,
        "val": val_best,
        "test": test,
        "history": history,
        "cfg": cfg,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not args.no_append:
        for split_name, m in (("val", val_best), ("test", test)):
            append_row(
                "results/results.csv",
                {
                    "exp_id": exp_id,
                    "model": mcfg["name"],
                    "pretrained": bool(mcfg["pretrained"]),
                    "aug": tcfg.get("aug", "basic"),
                    "seed": seed,
                    "data_ratio": ratio,
                    "split": split_name,
                    "corruption": "clean",
                    "severity": 0,
                    "acc": round(m["acc"], 4),
                    "macro_f1": round(m["macro_f1"], 4),
                    "params_m": params_m,
                    "notes": f"freeze={freeze} cw={tcfg.get('class_weight', 'none')}",
                },
            )

    print(
        f"[done] best_epoch={best_epoch} val macroF1={val_best['macro_f1']:.4f} "
        f"test macroF1={test['macro_f1']:.4f} -> {run_dir}"
    )


if __name__ == "__main__":
    main()
