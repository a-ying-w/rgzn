"""模型注册表（README §5 的 build_model 契约）。

全组共用两条硬约定：
1. 每个模型都暴露 .backbone / .head 两个子模块——train.py 的冻结与分组学习率依赖它；
2. 入口签名为 build_model(name, num_classes, pretrained)，C 的 MobileNet 也从这里取。
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models as tvm

from .cnn import SmallCNN

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
    head = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout), nn.Linear(dim, num_classes))
    return BackboneHeadNet(backbone, head)


def build_model(name: str, num_classes: int = 6, pretrained: bool = False, dropout: float = 0.2):
    if name not in _REGISTRY:
        raise KeyError(f"未注册的模型 {name!r}，可选: {available_models()}")
    return _REGISTRY[name](num_classes=num_classes, pretrained=pretrained, dropout=dropout)
