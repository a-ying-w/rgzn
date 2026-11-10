"""Grad-CAM：看模型到底在盯叶片的哪一块（C 岗可解释性交付）。

两种用法：

1) 命令行批量出图（答辩用，扰动图直接来自 build_loaders）：

    python -m src.models.gradcam --ckpt runs/e1e_mobilenet_v3_small_ft_r100_s42/best.pth \
        --corruption blur --severity 4 --num 8

2) 被 app.py 调用：GradCAM(model).explain(x) -> (cam, probs)

注意：Grad-CAM 要反传梯度，调用方不要把它包在 torch.inference_mode() 里。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.config import load_with_overrides  # noqa: E402
from src.data.corruptions import CORRUPTIONS  # noqa: E402
from src.data.dataset import build_loaders  # noqa: E402
from src.models.build import build_model, get_target_layer, guess_model_name, load_checkpoint  # noqa: E402

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def denormalize(tensor: torch.Tensor, mean, std) -> np.ndarray:
    """把 eval 变换后的张量还原成 [H, W, 3] 的 0-1 图像，用于叠加显示。"""
    mean_t = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std_t = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    image = (tensor * std_t + mean_t).clamp(0, 1)
    return image.permute(1, 2, 0).cpu().numpy()


def overlay_cam(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """把 0-1 的热力图叠到 0-1 的 RGB 图上，返回 uint8 供 Gradio / matplotlib 显示。"""
    heat = plt.get_cmap("jet")(cam)[..., :3]
    blended = (1 - alpha) * image + alpha * heat
    return (blended.clip(0, 1) * 255).astype(np.uint8)


class GradCAM:
    """最小实现：forward hook 拿特征图，tensor hook 拿梯度，不依赖第三方 cam 库。

    model 需处于 eval()，并且只在单张图（batch=1）上调用。
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer or get_target_layer(model)
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self._acts = output

        def _save_grad(grad):
            self._grads = grad.detach()

        output.register_hook(_save_grad)

    def explain(self, x: torch.Tensor, class_idx: int | None = None):
        """返回 (cam[H,W] 0-1, probs[K])；class_idx 为空时取预测类。"""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = logits.softmax(dim=1)
        if class_idx is None:
            class_idx = int(probs.argmax(dim=1)[0])
        logits[0, class_idx].backward()

        if self._acts is None or self._grads is None:
            raise RuntimeError("没拿到特征图梯度，检查 target_layer 或模型是否被 no_grad 包裹")

        weights = self._grads.mean(dim=(2, 3), keepdim=True)  # 每个通道的权重
        cam = torch.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), probs.detach().cpu().numpy()[0]


def _short_name(class_name: str) -> str:
    return class_name.split("___")[-1]


def plot_grid(samples: list[dict], out_path: str | Path, suptitle: str) -> Path:
    """一行一个样本：输入（扰动后） / 热力图 / 叠加。标题用英文避免字体缺字。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 2.8 * rows), squeeze=False)
    for row, axes_row in zip(samples, axes):
        panels = (row["image"], row["heat"], row["overlay"])
        for ax, panel in zip(axes_row, panels):
            ax.imshow(panel)
            ax.axis("off")
        axes_row[0].set_title(
            f"true: {row['true']}\npred: {row['pred']} ({row['prob']:.2f})",
            fontsize=8,
        )
    fig.suptitle(f"{suptitle}   (left: input / middle: Grad-CAM / right: overlay)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def pick_balanced_indices(dataset, num: int, seed: int = 42) -> list[int]:
    """test.csv 是按类别排好序的，直接取前 N 张会全是同一个类；这里按类轮转抽。

    轮转而不是"每类取 num/K 张再截断"，保证 num >= 类别数时每个类至少出一张。
    """
    labels = np.asarray(dataset.labels)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    pools = {cls: list(rng.permutation(np.flatnonzero(labels == cls))) for cls in classes}
    chosen: list[int] = []
    while len(chosen) < num and any(pools.values()):
        for cls in classes:
            if pools[cls] and len(chosen) < num:
                chosen.append(int(pools[cls].pop()))
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 Grad-CAM 对比图")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None, help="覆盖配置，如 data.root=...")
    parser.add_argument("--ckpt", required=True, help="runs/<run_name>/best.pth")
    parser.add_argument("--model", default=None, help="默认从 ckpt 路径猜，猜不到用 config 的 model.name")
    parser.add_argument("--corruption", default="clean", choices=("clean",) + CORRUPTIONS)
    parser.add_argument("--severity", type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--num", type=int, default=8, help="出图张数")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="默认 results/figures/gradcam_<model>_<tag>.png")
    args = parser.parse_args()

    cfg = load_with_overrides(args.config, args.overrides)
    cfg["data"]["num_workers"] = 0  # 出图脚本单进程即可
    classes: list[str] = cfg["data"]["classes"]
    mean = cfg["data"].get("mean") or DEFAULT_MEAN
    std = cfg["data"].get("std") or DEFAULT_STD
    model_name = args.model or guess_model_name(args.ckpt, cfg["model"]["name"])
    corruption = None if args.corruption == "clean" else args.corruption
    severity = args.severity if corruption else 0
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(model_name, num_classes=len(classes), pretrained=False)
    load_checkpoint(model, args.ckpt)
    model.to(device).eval()
    cam_engine = GradCAM(model)

    # 扰动图交给 build_loaders：它会把 Corruption 接在 eval 变换之前
    loader = build_loaders(cfg, corruption=corruption, severity=severity, splits=(args.split,))[args.split]
    dataset = loader.dataset
    indices = pick_balanced_indices(dataset, args.num, seed=args.seed)

    samples = []
    for index in indices:
        x, y = dataset[index]
        x = x.unsqueeze(0).to(device)
        cam, probs = cam_engine.explain(x)
        pred = int(probs.argmax())
        image = denormalize(x[0], mean, std)
        samples.append(
            {
                "image": image,
                "heat": plt.get_cmap("jet")(cam)[..., :3],
                "overlay": overlay_cam(image, cam),
                "true": _short_name(classes[y]),
                "pred": _short_name(classes[pred]),
                "prob": float(probs[pred]),
            }
        )
        flag = "OK " if pred == y else "ERR"
        print(f"[{flag}] {dataset.df.iloc[index]['path']}  pred={classes[pred]} p={probs[pred]:.3f}")

    tag = "clean" if corruption is None else f"{corruption}_s{severity}"
    out = args.out or f"results/figures/gradcam_{model_name}_{tag}.png"
    title = f"{model_name} @ {tag} (n={len(samples)})"
    saved = plot_grid(samples, out, title)
    print(f"\n已保存 {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
