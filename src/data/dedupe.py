"""pHash 近重复检测。

纯 numpy + Pillow 实现，不依赖 torch / imagehash / scipy，
因此本地 Python 3.14 也能跑（正式训练环境反而装不了 torch）。

PlantVillage 里同一片叶子会被拍很多张，随机划分会让测试集虚高。
这里把"视觉上几乎一样"的图片归到同一个 group，划分时按 group 整体划，
保证同一组图片不会同时出现在 train 和 test。
"""

from __future__ import annotations

import itertools

import numpy as np
from PIL import Image

_HASH_SIZE = 8
_IMG_SIZE = 32


def _dct_matrix(n: int = _IMG_SIZE) -> np.ndarray:
    """DCT-II 变换矩阵（正交归一化），用来近似的取低频分量。"""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[0, :] *= 1.0 / np.sqrt(2.0)
    return m * np.sqrt(2.0 / n)


_DCT = _dct_matrix()


def phash(img: Image.Image, hash_size: int = _HASH_SIZE) -> int:
    """返回 64 位感知哈希（int）。"""
    gray = img.convert("L").resize((_IMG_SIZE, _IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    coef = _DCT @ arr @ _DCT.T
    block = coef[:hash_size, :hash_size].ravel()
    # 去掉直流分量后取中位数作为阈值
    median = np.median(block[1:])
    value = 0
    for bit in block > median:
        value = (value << 1) | int(bit)
    return value


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def hamming_pairs(values: np.ndarray, max_dist: int, chunk: int = 512) -> list[tuple[int, int]]:
    """找出所有 hamming 距离 <= max_dist 的下标对 (i, j)，i < j。"""
    arr = np.asarray(values, dtype=np.uint64)
    n = arr.size
    pairs: list[tuple[int, int]] = []
    for start in range(0, n, chunk):
        left = arr[start:start + chunk, None] ^ arr[None, :]
        # uint64 -> 8 个字节 -> 逐字节 popcount 求和
        dist = _POPCOUNT[left.view(np.uint8).reshape(left.shape + (8,))].sum(axis=-1)
        idx = np.argwhere(dist <= max_dist)
        if idx.size == 0:
            continue
        idx[:, 0] += start
        pairs.extend(map(tuple, idx[idx[:, 0] < idx[:, 1]]))
    return pairs


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def assign_group_ids(records: list[dict], max_dist: int = 8) -> dict[str, str]:
    """按类别内部做近重复聚类。

    records: [{"path": ..., "class_name": ..., "phash": int}, ...]
    返回: {path: group_id}
    """
    out: dict[str, str] = {}
    ordered = sorted(records, key=lambda r: (r["class_name"], r["path"]))
    for cls, chunk in itertools.groupby(ordered, key=lambda r: r["class_name"]):
        items = list(chunk)
        values = np.array([r["phash"] for r in items], dtype=np.uint64)
        parent = list(range(len(items)))
        for i, j in hamming_pairs(values, max_dist):
            _union(parent, int(i), int(j))
        root_to_gid: dict[int, str] = {}
        for i, rec in enumerate(items):
            root = _find(parent, i)
            gid = root_to_gid.setdefault(root, f"{cls}_g{len(root_to_gid):04d}")
            out[rec["path"]] = gid
    return out
