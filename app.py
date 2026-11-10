"""Gradio Demo：上传叶片照片 → 病害预测 + Grad-CAM 热力图（C 岗交付）。

    python app.py --ckpt runs/e1_mobilenet_v3_small_s42/best.pth

预处理必须和评测完全一致：直接调用 src/data/transforms.py 的 build_eval_transform
（Resize(256) → CenterCrop(224) → ToTensor → Normalize(mean/std)），不要在这里另写一套。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.data.transforms import build_eval_transform
from src.models.build import build_model, count_parameters, guess_model_name, load_checkpoint
from src.models.gradcam import GradCAM, denormalize, overlay_cam

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

# 类别名 → 中文病名（只影响展示，不影响预测）
CN_NAMES = {
    "Tomato___Bacterial_spot": "番茄细菌性斑点病",
    "Tomato___Early_blight": "番茄早疫病",
    "Tomato___Late_blight": "番茄晚疫病",
    "Tomato___Leaf_Mold": "番茄叶霉病",
    "Tomato___Septoria_leaf_spot": "番茄斑枯病",
    "Tomato___healthy": "番茄健康叶片",
}


def display_name(class_name: str) -> str:
    return f"{CN_NAMES.get(class_name, class_name)}（{class_name.split('___')[-1]}）"


def topk_probs(probs: np.ndarray, classes: list[str], k: int = 3) -> dict[str, float]:
    order = np.argsort(probs)[::-1][:k]
    return {display_name(classes[i]): float(probs[i]) for i in order}


def main() -> int:
    parser = argparse.ArgumentParser(description="植物叶片病害识别 Demo")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--ckpt", default="runs/e1e_mobilenet_v3_small_ft_r100_s42/best.pth")
    parser.add_argument("--model", default=None, help="默认从 ckpt 路径猜，猜不到用 config 的 model.name")
    parser.add_argument("--device", default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成公网临时链接（答辩演示用）")
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError as error:  # noqa: BLE001
        raise SystemExit("缺少 gradio：pip install gradio") from error

    cfg = load_config(args.config)
    classes: list[str] = cfg["data"]["classes"]
    dcfg = cfg["data"]
    mean = dcfg.get("mean") or DEFAULT_MEAN
    std = dcfg.get("std") or DEFAULT_STD
    model_name = args.model or guess_model_name(args.ckpt, cfg["model"]["name"])
    if not Path(args.ckpt).exists():
        raise SystemExit(
            f"找不到权重 {args.ckpt}\n"
            f"先训练：python -m src.train --config configs/e1_mobilenet_v3_small.yaml --seed 42"
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(model_name, num_classes=len(classes), pretrained=False)
    load_checkpoint(model, args.ckpt)
    model.to(device).eval()
    cam_engine = GradCAM(model)

    # 与 eval 完全同一条变换，参数全部来自 config
    eval_tf = build_eval_transform(
        image_size=dcfg["image_size"],
        resize_size=dcfg["resize_size"],
        mean=mean,
        std=std,
    )

    def predict(pil_image):
        if pil_image is None:
            return {}, None
        x = eval_tf(pil_image.convert("RGB")).unsqueeze(0).to(device)
        cam, probs = cam_engine.explain(x)
        rgb = denormalize(x[0], mean, std)
        return topk_probs(probs, classes), overlay_cam(rgb, cam)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="上传叶片照片"),
        outputs=[
            gr.Label(num_top_classes=3, label="预测（Top-3）"),
            gr.Image(label="Grad-CAM 热力图"),
        ],
        title="植物叶片病害识别 · MobileNetV3-Small",
        description=(
            f"模型 {model_name}（{count_parameters(model) / 1e6:.2f}M 参数）。"
            "预处理与评测一致：Resize(256) → CenterCrop(224) → Normalize；"
            "热力图高亮模型判病时真正关注的区域。"
        ),
    )
    # server_name 显式绑定回环地址：本机开着系统代理时，Gradio 启动自检不会走代理导致 502
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
