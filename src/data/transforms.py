"""增强配方注册表。

全组只通过 aug 名字（none / basic / strong）来选择增强，
不要在各自脚本里手写 transform，否则 E2 的消融就不成立。
"""

from __future__ import annotations

from torchvision import transforms as T

from .corruptions import Corruption

AUG_RECIPES = ("none", "basic", "strong")


def build_train_transform(
    recipe: str = "basic",
    image_size: int = 224,
    resize_size: int = 256,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> T.Compose:
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    if recipe == "none":
        ops = [T.Resize(resize_size), T.CenterCrop(image_size)]
    elif recipe == "basic":
        ops = [
            T.Resize(resize_size),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.2),
            T.RandomRotation(15),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        ]
    elif recipe == "strong":
        ops = [
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(0.5),
            T.RandAugment(num_ops=2, magnitude=9),
        ]
    else:
        raise ValueError(f"未知增强配方 {recipe!r}，可选: {AUG_RECIPES}")

    tail = [T.ToTensor(), T.Normalize(mean, std)]
    if recipe == "strong":
        tail.append(T.RandomErasing(p=0.25))
    return T.Compose(ops + tail)


def build_eval_transform(
    image_size: int = 224,
    resize_size: int = 256,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> T.Compose:
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    return T.Compose(
        [
            T.Resize(resize_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def build_corrupted_eval_transform(
    name: str,
    severity: int,
    image_size: int = 224,
    resize_size: int = 256,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> T.Compose:
    """先施加扰动，再走正常 eval 流程。severity=0 等价于干净测试集。"""
    return T.Compose(
        [
            Corruption(name, severity),
            build_eval_transform(image_size, resize_size, mean, std),
        ]
    )
