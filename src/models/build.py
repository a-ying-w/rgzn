"""模型注册表（README §5 的 build_model 契约）。

全组共用两条硬约定：
1. 每个模型都暴露 .backbone / .head 两个子模块——train.py 的冻结与分组学习率依赖它；
2. 入口签名为 build_model(name, num_classes, pretrained)，C 的 MobileNet 也从这里取。

C 岗额外的工具（Grad-CAM / Demo 用）：count_parameters、guess_model_name、
get_target_layer、load_checkpoint。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as tvm

from .cnn import SmallCNN

NUM_CLASSES = 6

_REGISTRY = {}


def register_model(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


def available_models() -> list[str]:
    return sorted(_REGISTRY)


class BackboneHeadNet(nn.Module):
    """把 torchvision 模型拆成 backbone + head，满足统一契约。"""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


@register_model("cnn")
def _make_cnn(num_classes=6, pretrained=False, dropout=0.2):
    # 自建网络没有预训练权重，pretrained 参数忽略
    return SmallCNN(num_classes=num_classes, dropout=dropout)


@register_model("resnet18")
def _make_resnet18(num_classes=6, pretrained=False, dropout=0.2):
    m = tvm.resnet18(weights="DEFAULT" if pretrained else None)
    dim = m.fc.in_features
    m.fc = nn.Identity()  # resnet 自带 avgpool + flatten，这里输出 (B, 512)
    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes))
    return BackboneHeadNet(m, head)


@register_model("mobilenet_v3_small")
def _make_mobilenet_v3_small(num_classes=6, pretrained=False, dropout=0.2):
    m = tvm.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
    dim = m.classifier[-1].in_features
    backbone = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1))
    # 保留预训练的 576->1024 隐层，只换最后一层线性头（参数量与官方一致，约 1.5M）
    head = nn.Sequential(nn.Flatten(1), m.classifier[:-1], nn.Linear(dim, num_classes))
    return BackboneHeadNet(backbone, head)


def build_model(name: str, num_classes: int = 6, pretrained: bool = False, dropout: float = 0.2):
    if name not in _REGISTRY:
        raise KeyError(f"未注册的模型 {name!r}，可选: {available_models()}")
    return _REGISTRY[name](num_classes=num_classes, pretrained=pretrained, dropout=dropout)


# ---------------------------------------------------------------- C 岗工具函数


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """模型参数量（trainable_only=True 时只数 requires_grad 的）。"""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def guess_model_name(ckpt_path: str | Path, fallback: str) -> str:
    """从 runs/<run_name>/best.pth 的路径猜模型名，避免 --model 与权重不匹配。"""
    text = Path(ckpt_path).as_posix().lower()
    for name in _REGISTRY:
        if name in text:
            return name
    aliases = {"mobilenet": "mobilenet_v3_small", "resnet": "resnet18"}
    for alias, name in aliases.items():
        if alias in text:
            return name
    return fallback


def get_target_layer(model: nn.Module) -> nn.Module:
    """Grad-CAM 目标层：ResNet 取 layer4 末块，其余取最后一个 Conv2d。

    兼容 BackboneHeadNet 包装（此时真实主体在 model.backbone 里）。
    """
    base = model.backbone if hasattr(model, "backbone") else model
    if hasattr(base, "layer4"):
        return base.layer4[-1]
    if hasattr(base, "features"):
        return base.features[-1]
    convs = [m for m in base.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("模型里找不到 Conv2d，Grad-CAM 需要显式指定 target_layer")
    return convs[-1]


def load_checkpoint(model: nn.Module, path: str | Path) -> nn.Module:
    """加载 train.py 产出的 best.pth；兼容 state_dict / {'state_dict': ...} / DataParallel。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到权重 {path}")
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 — 老 checkpoint 可能夹带非张量对象
        blob = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(blob, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in blob and isinstance(blob[key], dict):
                blob = blob[key]
                break

    state = {k: v for k, v in blob.items() if torch.is_tensor(v)}
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if state and all(k.startswith("model.") for k in state):
        state = {k[len("model.") :]: v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"权重与模型结构不匹配：missing={missing[:5]} unexpected={unexpected[:5]} "
            f"（检查 --model 是否与训练时一致）"
        )
    return model


if __name__ == "__main__":
    for model_name in available_models():
        net = build_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
        print(f"{model_name:20s} params={count_parameters(net) / 1e6:.3f}M")
