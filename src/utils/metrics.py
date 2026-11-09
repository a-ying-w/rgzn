"""评价指标。纯 numpy 实现，不依赖 sklearn，方便在本地随便跑。"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), 0.0)
    return recall


def per_class_precision(cm: np.ndarray) -> np.ndarray:
    pred = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pred > 0, np.diag(cm) / np.maximum(pred, 1), 0.0)


def macro_f1(cm: np.ndarray) -> float:
    p = per_class_precision(cm)
    r = per_class_recall(cm)
    f1 = np.where((p + r) > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return float(f1.mean())


def summarize(y_true, y_pred, num_classes: int, class_names: list[str] | None = None) -> dict:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    recall = per_class_recall(cm)
    out = {
        "acc": float(np.trace(cm) / max(cm.sum(), 1)),
        "macro_f1": macro_f1(cm),
        "macro_recall": float(recall.mean()),
        "worst_recall": float(recall.min()),
    }
    if class_names:
        out["worst_class"] = class_names[int(recall.argmin())]
    return out


if __name__ == "__main__":
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    print(summarize(y_true, y_pred, 3, ["a", "b", "c"]))
