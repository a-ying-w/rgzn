"""测试时扰动（鲁棒性实验 E3）。

只依赖 Pillow，不依赖 torch，因此可以直接在本地验证效果。
设计原则：扰动作用在"原始图片"上，之后照常走 eval 的 Resize/CenterCrop/Normalize。

severity 1-5，与 ImageNet-C 的思路一致：级数越高越强。
"""

from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter

CORRUPTIONS = ("brightness", "rotation", "blur")

# 各级别参数（数字含义见各类别名）
SEVERITY_MAP: dict[str, dict[int, float]] = {
    # 亮度缩放系数，1.0 表示不变；越小越暗
    "brightness": {1: 0.90, 2: 0.75, 3: 0.60, 4: 0.45, 5: 0.30},
    # 旋转角度（度）
    "rotation": {1: 10, 2: 20, 3: 30, 4: 45, 5: 60},
    # 高斯模糊半径（约等于 sigma）
    "blur": {1: 0.6, 2: 1.0, 3: 1.6, 4: 2.4, 5: 3.2},
}

_FILL = (124, 124, 124)


def _center_crop(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = img.size
    tw, th = size
    left = max(0, (w - tw) // 2)
    top = max(0, (h - th) // 2)
    return img.crop((left, top, left + tw, top + th))


class Corruption:
    """可当 torchvision.transforms 的普通 callable 用：PIL in -> PIL out。"""

    def __init__(self, name: str, severity: int):
        if name not in SEVERITY_MAP:
            raise ValueError(f"未知扰动 {name!r}，可选: {CORRUPTIONS}")
        if severity not in SEVERITY_MAP[name]:
            raise ValueError(f"severity 必须是 0-5，收到 {severity}")
        self.name = name
        self.severity = severity

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.severity == 0:
            return img
        value = SEVERITY_MAP[self.name][self.severity]
        if self.name == "brightness":
            return ImageEnhance.Brightness(img).enhance(value)
        if self.name == "rotation":
            rotated = img.rotate(value, resample=Image.BILINEAR, expand=True, fillcolor=_FILL)
            return _center_crop(rotated, img.size)
        if self.name == "blur":
            return img.filter(ImageFilter.GaussianBlur(radius=value))
        raise ValueError(self.name)

    def __repr__(self) -> str:
        return f"Corruption({self.name!r}, severity={self.severity})"


def apply_corruption(img: Image.Image, name: str, severity: int) -> Image.Image:
    return Corruption(name, severity)(img)


def describe() -> str:
    lines = []
    for name in CORRUPTIONS:
        params = ", ".join(f"s{k}={v}" for k, v in SEVERITY_MAP[name].items())
        lines.append(f"{name:11s} {params}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
