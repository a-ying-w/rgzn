"""自建 CNN：4 个 conv block（每块 conv-bn-relu ×2 + maxpool），GAP 后接 head。"""

from __future__ import annotations

import torch.nn as nn


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """224 -> 112 -> 56 -> 28 -> 14 -> GAP -> head。"""

    def __init__(self, num_classes: int = 6, dropout: float = 0.2, widths=(32, 64, 128, 256)):
        super().__init__()
        blocks = []
        cin = 3
        for w in widths:
            blocks.append(_conv_block(cin, w))
            cin = w
        self.backbone = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(cin, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.backbone(x))
