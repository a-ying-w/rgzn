"""为 web/ 前端汇总只读数据与图片素材。

设计原则：
1. 只读 results/、runs/、data/splits/、data/raw/、report/，不写回任何实验产物；
2. 所有数字都来自真实文件（results.csv / summary.csv / metrics.json / meta.json），不编数据；
3. 产出全部落在 web/data/ 与 web/static/assets/，可重复执行。

用法：

    python scripts/build_web_data.py                 # 汇总数据 + 生成图片
    python scripts/build_web_data.py --with-eval     # 额外补跑 e1a/e1c/e1d 的逐类指标（CPU，几分钟）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

from PIL import Image

# 本地 torch + matplotlib 同时加载会触发 OpenMP 重复链接（见项目经验记录）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WEB = ROOT / "web"
DATA_DIR = WEB / "data"
ASSETS = WEB / "static" / "assets"
IMG_DIR = ASSETS / "img"
FIG_DIR = ROOT / "report" / "figures"


# ---------------------------------------------------------------- 类别元信息

DISEASES = [
    {
        "id": "bacterial-spot",
        "en": "Bacterial Spot",
        "cn": "番茄细菌性斑点病",
        "cls": "Tomato___Bacterial_spot",
        "kind": "细菌性病害",
        "tag": "高传染",
        "summary": "由 Xanthomonas 属细菌引起的叶、茎、果均可受害的病害，高温高湿、雨水飞溅时传播最快。",
        "symptoms": [
            "叶片初现水浸状小斑点，后扩大为深褐色近圆形病斑",
            "病斑周围常有明显的黄色晕圈，是区别于真菌性斑点的重要特征",
            "严重时病斑连片，叶片枯黄卷曲并大量脱落",
            "果实表面形成黑褐色、略微隆起的疮痂状小斑",
        ],
        "causes": "病原细菌随病残体、种子和土壤越冬，靠雨水飞溅、灌溉水和农事操作传播；气温 24–30℃ 且叶面结露时发病最重。",
        "prevention": [
            "使用无病种子并进行温汤浸种处理",
            "避免顶部喷灌，改用滴灌降低叶面湿度",
            "及时清除病叶病果，收获后深翻病残体",
            "发病初期可选用铜制剂或抗生素类药剂，注意轮换用药",
        ],
        "similar": ["septoria-leaf-spot"],
        "similar_note": "与斑枯病同为小型斑点，但本病的病斑带黄色晕圈、中央不呈灰白色，且叶背少见黑色小点。",
    },
    {
        "id": "early-blight",
        "en": "Early Blight",
        "cn": "番茄早疫病",
        "cls": "Tomato___Early_blight",
        "kind": "真菌性病害",
        "tag": "常见",
        "summary": "由链格孢菌（Alternaria solani）引起，多从下部老叶开始发病，病斑具同心轮纹，是番茄生产中最常见的叶部病害之一。",
        "symptoms": [
            "下部老叶先出现褐色小点，逐渐扩大为圆形病斑",
            "病斑上有明显的同心轮纹，呈“靶心”状，外围常伴黄化",
            "多个病斑融合后叶片干枯，但通常不会整株迅速萎蔫",
            "茎部和果实也可受害，形成凹陷的褐色病斑",
        ],
        "causes": "病菌以菌丝和分生孢子在病残体上越冬，温暖潮湿（25–28℃）且植株长势弱时易流行，老叶和缺肥植株更敏感。",
        "prevention": [
            "与非茄科作物轮作 2 年以上，清理病残体",
            "合理密植、及时整枝打杈，改善通风透光",
            "平衡施肥，避免后期缺氮早衰",
            "发病初期喷施代森锰锌、苯醚甲环唑等登记药剂",
        ],
        "similar": ["late-blight"],
        "similar_note": "与晚疫病相比，早疫病病斑有清晰同心轮纹、扩展较慢，且很少在叶背形成白色霉层。",
    },
    {
        "id": "late-blight",
        "en": "Late Blight",
        "cn": "番茄晚疫病",
        "cls": "Tomato___Late_blight",
        "kind": "卵菌性病害",
        "tag": "高危险",
        "summary": "由致病疫霉（Phytophthora infestans）引起，流行速度极快，低温高湿条件下可在数天内毁园，是番茄和马铃薯的头号病害。",
        "symptoms": [
            "叶尖或叶缘出现不规则水浸状暗绿色斑，迅速扩大为褐色大斑",
            "湿度高时病斑边缘叶背产生一圈白色稀疏霉层",
            "茎部出现黑褐色条斑，严重时植株上部枯死",
            "青果出现油浸状褐色硬斑，表面不平整，不软腐",
        ],
        "causes": "病原随气流和雨水远距离传播，适宜温度 18–22℃、相对湿度 85% 以上；昼夜温差大、结露时间长时爆发风险最高。",
        "prevention": [
            "选用抗病品种，避免与马铃薯邻作",
            "控制棚内湿度，及时通风、避免叶面长时间结露",
            "发现中心病株立即拔除并带出田外销毁",
            "流行期前喷施氟啶胺、烯酰吗啉等登记药剂预防",
        ],
        "similar": ["early-blight", "leaf-mold"],
        "similar_note": "与早疫病的区别在于病斑无同心轮纹、呈水浸状且扩展迅猛；与叶霉病的区别是后者叶背为橄榄色绒霉而非白色霉层。",
    },
    {
        "id": "leaf-mold",
        "en": "Leaf Mold",
        "cn": "番茄叶霉病",
        "cls": "Tomato___Leaf_Mold",
        "kind": "真菌性病害",
        "tag": "棚室高发",
        "summary": "由黄枝孢菌（Passalora fulva）引起，典型特征是叶正面褪绿、叶背长出橄榄色绒状霉层，保护地栽培中尤为常见。",
        "symptoms": [
            "叶正面出现边界不清的淡黄色斑块，逐渐扩大",
            "叶背对应位置长出灰褐色至橄榄色绒状霉层",
            "病斑增多后叶片卷曲、干枯，但一般不形成明显坏死大斑",
            "严重时蔓延至茎、花和果实，影响坐果",
        ],
        "causes": "病菌在病残体和种子上越冬，孢子借气流传播；相对湿度 90% 以上、棚内通风不良、种植过密时发病严重。",
        "prevention": [
            "选用抗叶霉病的品种，种子消毒后播种",
            "加强棚室通风，控制湿度在 85% 以下",
            "及时摘除下部病叶，避免密植",
            "发病初期选用嘧菌酯、氟硅唑等登记药剂交替喷施",
        ],
        "similar": ["late-blight"],
        "similar_note": "与晚疫病的关键差异：叶霉病叶背是橄榄色绒霉、病斑黄化但不水浸状；晚疫病则是水浸状褐斑加白色霉层。",
    },
    {
        "id": "septoria-leaf-spot",
        "en": "Septoria Leaf Spot",
        "cn": "番茄斑枯病",
        "cls": "Tomato___Septoria_leaf_spot",
        "kind": "真菌性病害",
        "tag": "多雨高发",
        "summary": "由番茄壳针孢（Septoria lycopersici）引起，病斑小而多，中央灰白并散生黑色小点，多雨季节常自下而上迅速蔓延。",
        "symptoms": [
            "下部叶片先出现大量直径 2–3 mm 的圆形小斑",
            "病斑中央灰白色、边缘深褐色，界限清晰",
            "病斑中央可见细小黑点，即病原菌的分生孢子器",
            "严重时叶片黄化、干枯脱落，植株早衰",
        ],
        "causes": "病菌以分生孢子器在病残体上越冬，靠雨水飞溅传播；温暖（20–25℃）多雨、田间积水时扩展迅速。",
        "prevention": [
            "清除病残体并深翻土壤，与非茄科作物轮作",
            "采用滴灌或地膜覆盖，减少雨水飞溅",
            "及时摘除下部病叶，保持田间通风",
            "发病初期喷施代森锰锌、苯醚甲环唑等登记药剂",
        ],
        "similar": ["bacterial-spot"],
        "similar_note": "与细菌性斑点病相比，本病病斑中央灰白、可见黑色小点，且通常没有黄色晕圈。",
    },
    {
        "id": "healthy",
        "en": "Healthy",
        "cn": "番茄健康叶片",
        "cls": "Tomato___healthy",
        "kind": "健康状态",
        "tag": "无病害",
        "summary": "未感染上述病害的正常番茄叶片，叶色均匀浓绿、叶面平展，是识别模型的重要对照类别。",
        "symptoms": [
            "叶片舒展、颜色均匀浓绿，无黄化或褪绿斑块",
            "叶面平整，无水浸状、坏死斑或霉层",
            "叶脉清晰，叶缘完整，无卷曲或焦枯",
            "植株整体长势健壮，新叶老叶状态一致",
        ],
        "causes": "健康的栽培管理结果：光照充足、水肥均衡、通风良好，且未接触病原。",
        "prevention": [
            "保持合理的水肥与光照管理",
            "定期巡查，发现异常叶片及早处理",
            "做好通风与湿度控制，预防病害发生",
        ],
        "similar": [],
        "similar_note": "健康叶片不含病斑与霉层；若出现局部黄化，应重点排查叶霉病早期与缺素症。",
    },
]

DISEASE_BY_CLASS = {d["cls"]: d for d in DISEASES}

ROBUSTNESS_ARMS = OrderedDict(
    [
        ("e3a", {"label": "ResNet-18 从零训练", "short": "ResNet-18 Scratch", "model": "resnet18", "pretrained": False}),
        ("e3b", {"label": "ResNet-18 迁移学习", "short": "ResNet-18 Transfer", "model": "resnet18", "pretrained": True}),
        ("e3c", {"label": "自建 CNN 从零训练", "short": "Small CNN", "model": "cnn", "pretrained": False}),
        ("e3d", {"label": "ResNet-18 类别加权", "short": "ResNet-18 Class-Weighted", "model": "resnet18", "pretrained": True}),
    ]
)

MODEL_LABELS = {
    "cnn": {"cn": "自建 CNN", "en": "Small CNN"},
    "resnet18": {"cn": "ResNet-18", "en": "ResNet-18"},
    "mobilenet_v3_small": {"cn": "MobileNetV3-Small", "en": "MobileNetV3-Small"},
}


# ---------------------------------------------------------------- 读数据


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_results() -> list[dict]:
    rows = []
    for row in read_csv(ROOT / "results" / "results.csv"):
        rows.append(
            {
                "exp_id": row["exp_id"],
                "model": row["model"],
                "pretrained": row["pretrained"].strip().lower() == "true",
                "aug": row["aug"],
                "seed": parse_int(row.get("seed")),
                "data_ratio": parse_float(row.get("data_ratio"), 1.0),
                "split": row["split"],
                "corruption": row["corruption"],
                "severity": parse_int(row.get("severity"), 0),
                "acc": parse_float(row.get("acc")),
                "macro_f1": parse_float(row.get("macro_f1")),
                "params_m": parse_float(row.get("params_m")),
                "notes": row.get("notes", ""),
            }
        )
    return rows


def build_model_table(results: list[dict]) -> list[dict]:
    """取每个模型在 test/clean 上的一行，构造性能对比表。"""
    order = ["e0", "e1a", "e1b", "e1c", "e1d", "e1e"]
    picked = {}
    for row in results:
        if row["split"] != "test" or row["corruption"] != "clean":
            continue
        if row["exp_id"] in order and row["exp_id"] not in picked:
            picked[row["exp_id"]] = row

    table = []
    for exp_id in order:
        row = picked.get(exp_id)
        if not row:
            continue
        label = MODEL_LABELS.get(row["model"], {"cn": row["model"], "en": row["model"]})
        note = {
            "e0": "冻结主干，仅训练分类头",
            "e1a": "从零训练，不使用预训练权重",
            "e1b": "ImageNet 预训练后整体微调（主模型）",
            "e1c": "轻量自建网络，从零训练",
            "e1d": "预训练微调 + 类别加权损失",
            "e1e": "轻量预训练模型，整体微调",
        }.get(exp_id, row["notes"])
        table.append(
            {
                "exp_id": exp_id,
                "model": row["model"],
                "name": label["cn"],
                "en": label["en"],
                "pretrained": row["pretrained"],
                "acc": round(row["acc"], 4) if row["acc"] is not None else None,
                "macro_f1": round(row["macro_f1"], 4) if row["macro_f1"] is not None else None,
                "params_m": row["params_m"],
                "note": note,
                "variant": "segmented" if "segmented" in (row["notes"] or "") else "color",
            }
        )
    return table


def build_robustness(results: list[dict]) -> dict:
    wanted = {"brightness", "rotation", "blur"}
    series = {exp: {"series": defaultdict(list)} for exp in ROBUSTNESS_ARMS}
    for row in results:
        exp = row["exp_id"]
        if exp not in ROBUSTNESS_ARMS:
            continue
        if row["split"] != "test" or row["corruption"] not in wanted:
            continue
        if row["severity"] < 1:
            continue
        series[exp]["series"][row["corruption"]].append(
            [row["severity"], round(row["acc"], 4), round(row["macro_f1"], 4)]
        )

    out = {}
    for exp, arm in ROBUSTNESS_ARMS.items():
        item = dict(arm)
        item["series"] = {
            name: sorted(points, key=lambda p: p[0]) for name, points in series[exp]["series"].items()
        }
        out[exp] = item
    return out


def build_crossdomain(results: list[dict]) -> dict:
    clean = [r for r in results if r["split"] == "test" and r["corruption"] == "clean"]
    color = next((r for r in clean if r["exp_id"] == "e1b"), None)
    seg = next((r for r in clean if r["exp_id"] == "e4a"), None)
    upper = next((r for r in clean if r["exp_id"] == "e4b"), None)
    return {
        "color": {"acc": round(color["acc"], 4), "macro_f1": round(color["macro_f1"], 4)} if color else None,
        "segmented": {"acc": round(seg["acc"], 4), "macro_f1": round(seg["macro_f1"], 4)} if seg else None,
        "segmented_upper": {"acc": round(upper["acc"], 4), "macro_f1": round(upper["macro_f1"], 4)} if upper else None,
        "drop": round(color["acc"] - seg["acc"], 4) if color and seg else None,
    }


def build_aug_ablation(results: list[dict]) -> dict:
    wanted = {"e2a", "e2c", "e1a", "e1b"}
    out = defaultdict(dict)
    for row in results:
        if row["split"] != "test" or row["corruption"] != "clean":
            continue
        if row["exp_id"] not in wanted:
            continue
        key = "pretrained" if row["pretrained"] else "scratch"
        out[key][row["aug"]] = {
            "acc": round(row["acc"], 4),
            "macro_f1": round(row["macro_f1"], 4),
            "exp_id": row["exp_id"],
        }
    return dict(out)


def class_split_counts() -> dict:
    """按 split 统计每个类别的样本数，供数据中心展示。"""
    out = {}
    for split in ("train", "val", "test"):
        path = ROOT / "data" / "splits" / f"{split}.csv"
        if not path.exists():
            continue
        counts = defaultdict(int)
        for row in read_csv(path):
            counts[row["class_name"]] += 1
        out[split] = dict(counts)
    return out


def load_metrics_json() -> dict:
    """读取 runs/*/metrics.json（只有部分 run 有），用于混淆矩阵与逐类指标。"""
    out = {}
    for path in sorted((ROOT / "runs").glob("*/metrics.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out[blob.get("run_name", path.parent.name)] = {
            "exp_id": blob.get("exp_id"),
            "arch": blob.get("arch"),
            "params_m": blob.get("params_m"),
            "class_names": blob.get("class_names"),
            "test": blob.get("test"),
            "val": blob.get("val"),
            "history": blob.get("history"),
        }
    return out


# ---------------------------------------------------------------- 图片素材


def save_jpg(img: Image.Image, out: Path, max_side: int = 1000, quality: int = 86) -> None:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=quality, optimize=True, progressive=True)


def dataset_root() -> Path:
    return ROOT / "data" / "raw" / "plantvillage" / "raw"


def collect_class_samples(per_class: int = 2) -> dict[str, list[Path]]:
    """按 test.csv 顺序，每个类别取前 N 张能读到的真实图片。"""
    split_csv = ROOT / "data" / "splits" / "test.csv"
    if not split_csv.exists():
        return {}
    root = dataset_root() / "color"
    picked: dict[str, list[Path]] = defaultdict(list)
    for row in read_csv(split_csv):
        cls = row["class_name"]
        if len(picked[cls]) >= per_class:
            continue
        path = root / row["path"]
        if path.exists():
            picked[cls].append(path)
        if all(len(picked[c]) >= per_class for c in DISEASE_BY_CLASS):
            break
    return picked


def build_disease_images() -> dict[str, str]:
    samples = collect_class_samples(per_class=2)
    out_map = {}
    for cls, paths in samples.items():
        meta = DISEASE_BY_CLASS.get(cls)
        if not meta or not paths:
            continue
        for i, path in enumerate(paths, start=1):
            name = f"{meta['id']}-{i}.jpg"
            save_jpg(Image.open(path), IMG_DIR / "diseases" / name)
        out_map[meta["id"]] = f"assets/img/diseases/{meta['id']}-1.jpg"
        if len(paths) > 1:
            out_map[f"{meta['id']}-alt"] = f"assets/img/diseases/{meta['id']}-2.jpg"
    return out_map


def build_corruption_images() -> dict:
    """用同一张泛化性较好的测试图生成 3 类 × 5 级扰动示例。"""
    import sys

    sys.path.insert(0, str(ROOT))
    from src.data.corruptions import CORRUPTIONS, Corruption

    candidates = collect_class_samples(per_class=1)
    base_path = None
    for cls in ("Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Septoria_leaf_spot"):
        if candidates.get(cls):
            base_path = candidates[cls][0]
            break
    if base_path is None:
        return {}

    base = Image.open(base_path)
    save_jpg(base, IMG_DIR / "corruptions" / "clean.jpg", max_side=800)
    out = {"clean": "assets/img/corruptions/clean.jpg", "classes": {}}
    for name in CORRUPTIONS:
        points = {}
        for sev in range(1, 6):
            img = Corruption(name, sev)(base)
            fname = f"{name}-{sev}.jpg"
            save_jpg(img, IMG_DIR / "corruptions" / fname, max_side=800)
            points[str(sev)] = f"assets/img/corruptions/{fname}"
        out["classes"][name] = points
    return out


def build_crossdomain_images() -> dict:
    """找一张 color / segmented 都存在的测试图，用于跨域对比。"""
    split_csv = ROOT / "data" / "splits" / "test.csv"
    if not split_csv.exists():
        return {}
    color_root = dataset_root() / "color"
    seg_root = dataset_root() / "segmented"
    for row in read_csv(split_csv):
        rel = Path(row["path"])
        color_path = color_root / rel
        seg_path = seg_root / rel.with_name(rel.stem + "_final_masked.jpg")
        if color_path.exists() and seg_path.exists():
            save_jpg(Image.open(color_path), IMG_DIR / "crossdomain" / "color.jpg", max_side=800)
            save_jpg(Image.open(seg_path), IMG_DIR / "crossdomain" / "segmented.jpg", max_side=800)
            return {
                "color": "assets/img/crossdomain/color.jpg",
                "segmented": "assets/img/crossdomain/segmented.jpg",
                "class_name": row["class_name"],
                "path": row["path"],
            }
    return {}


def build_sample_gallery(per_class: int = 4) -> dict[str, list[str]]:
    """数据中心样本墙：每类 4 张缩略图。"""
    samples = collect_class_samples(per_class=per_class)
    gallery = {}
    for cls, paths in samples.items():
        meta = DISEASE_BY_CLASS.get(cls)
        if not meta:
            continue
        names = []
        for i, path in enumerate(paths, start=1):
            name = f"sample-{i}.jpg"
            save_jpg(Image.open(path), IMG_DIR / "gallery" / meta["id"] / name, max_side=600, quality=82)
            names.append(f"assets/img/gallery/{meta['id']}/{name}")
        gallery[meta["id"]] = names
    return gallery


def copy_figures() -> list[dict]:
    out = []
    if not FIG_DIR.exists():
        return out
    target = IMG_DIR / "figures"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(FIG_DIR.glob("*.png")):
        shutil.copy2(path, target / path.name)
        out.append({"name": path.stem, "src": f"assets/img/figures/{path.name}"})
    return out


def copy_docs() -> list[dict]:
    out = []
    docs_dir = ASSETS / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("full_report.docx",):
        src = ROOT / "report" / name
        if src.exists():
            shutil.copy2(src, docs_dir / name)
            out.append({"name": name, "src": f"assets/docs/{name}"})
    return out


def copy_vendor() -> None:
    vendor = ASSETS / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    src = ROOT / "defense" / "_shared" / "echarts.min.js"
    if src.exists():
        shutil.copy2(src, vendor / "echarts.min.js")


# ---------------------------------------------------------------- 可选：补跑逐类指标


def run_extra_eval(exp_to_ckpt: dict[str, Path]) -> dict:
    """在 test 上补跑评测，只取混淆矩阵与逐类指标；不写 results.csv。"""
    import torch

    from src.config import load_config
    from src.data.dataset import build_loaders
    from src.models.build import build_model, guess_model_name, load_checkpoint
    from src.utils.metrics import confusion_matrix, per_class_precision, per_class_recall

    cfg = load_config(str(ROOT / "configs" / "base.yaml"))
    cfg["data"]["num_workers"] = 0
    classes = cfg["data"]["classes"]
    loader = build_loaders(cfg, splits=("test",))["test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = {}
    for exp, ckpt in exp_to_ckpt.items():
        model_name = guess_model_name(ckpt, cfg["model"]["name"])
        model = build_model(model_name, num_classes=len(classes), pretrained=False)
        load_checkpoint(model, ckpt)
        model.to(device).eval()

        y_true, y_pred = [], []
        with torch.inference_mode():
            for x, y in loader:
                pred = model(x.to(device)).argmax(dim=1).cpu()
                y_pred.extend(pred.tolist())
                y_true.extend(y.tolist())
        cm = confusion_matrix(y_true, y_pred, len(classes))
        out[exp] = {
            "exp_id": exp,
            "model": model_name,
            "ckpt": str(ckpt.relative_to(ROOT)),
            "acc": float((cm.diagonal().sum()) / max(cm.sum(), 1)),
            "confusion_matrix": cm.tolist(),
            "per_class_precision": dict(zip(classes, per_class_precision(cm).round(6).tolist())),
            "per_class_recall": dict(zip(classes, per_class_recall(cm).round(6).tolist())),
        }
        print(f"  [eval] {exp:<5} {model_name:<20} acc={out[exp]['acc']:.4f}")
    return out


# ---------------------------------------------------------------- 主流程


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 web 前端所需的只读数据与图片")
    parser.add_argument("--with-eval", action="store_true", help="补跑 e1a/e1c/e1d 的逐类指标（CPU）")
    parser.add_argument("--skip-images", action="store_true", help="只更新 JSON，不动图片")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    meta = json.loads((ROOT / "data" / "splits" / "meta.json").read_text(encoding="utf-8"))
    models = build_model_table(results)
    robustness = build_robustness(results)
    crossdomain = build_crossdomain(results)
    aug = build_aug_ablation(results)
    metrics = load_metrics_json()

    # 用 e1b 的 metrics.json 补全对比表里的逐类指标
    e1b_key = next((k for k, v in metrics.items() if v.get("exp_id") == "e1b" and v.get("test")), None)
    if e1b_key:
        blob = metrics[e1b_key]["test"]
        for item in models:
            if item["exp_id"] == "e1b":
                item["per_class_precision"] = blob.get("per_class_precision")
                item["per_class_recall"] = blob.get("per_class_recall")
                item["confusion_matrix"] = blob.get("confusion_matrix")
                break

    extra_path = DATA_DIR / "eval_extra.json"
    if args.with_eval:
        ckpts = OrderedDict()
        for exp, run_dir in (
            ("e1a", "e1a_resnet18_scratch_r100_s42"),
            ("e1c", "e1c_cnn_scratch_r100_s42"),
            ("e1d", "e1d_resnet18_ft_cw_r100_s42"),
        ):
            path = ROOT / "runs" / run_dir / "best.pth"
            if path.exists():
                ckpts[exp] = path
        if ckpts:
            print("补跑逐类指标（test split，CPU）...")
            extra = run_extra_eval(ckpts)
            extra_path.write_text(json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8")

    extra = {}
    if extra_path.exists():
        extra = json.loads(extra_path.read_text(encoding="utf-8"))
        for item in models:
            blob = extra.get(item["exp_id"])
            if blob:
                item["per_class_precision"] = blob["per_class_precision"]
                item["per_class_recall"] = blob["per_class_recall"]
                item["confusion_matrix"] = blob["confusion_matrix"]

    images = {"diseases": {}, "corruptions": {}, "crossdomain": {}, "gallery": {}}
    figures = []
    docs = []
    out_path = DATA_DIR / "site_data.json"
    if args.skip_images and out_path.exists():
        old = json.loads(out_path.read_text(encoding="utf-8"))
        images = old.get("images", images)
        figures = old.get("figures", [])
        docs = old.get("docs", [])
    if not args.skip_images:
        print("生成图片素材 ...")
        images["diseases"] = build_disease_images()
        images["corruptions"] = build_corruption_images()
        images["crossdomain"] = build_crossdomain_images()
        images["gallery"] = build_sample_gallery()
        figures = copy_figures()
        docs = copy_docs()
        copy_vendor()

    n_color = len(list((dataset_root() / "color").rglob("*.JPG"))) + len(list((dataset_root() / "color").rglob("*.jpg")))
    n_seg = len(list((dataset_root() / "segmented").rglob("*.JPG"))) + len(list((dataset_root() / "segmented").rglob("*.jpg")))

    site = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "meta": {
            **meta,
            "class_cn": {d["cls"]: d["cn"] for d in DISEASES},
            "class_en": {d["cls"]: d["en"] for d in DISEASES},
            "n_color": n_color,
            "n_segmented": n_seg,
            "per_class_split": class_split_counts(),
        },
        "models": models,
        "robustness": robustness,
        "crossdomain": crossdomain,
        "aug_ablation": aug,
        "diseases": DISEASES,
        "images": images,
        "figures": figures,
        "docs": docs,
        "cost": read_csv(ROOT / "results" / "model_cost.csv"),
        "cost_cpu": read_csv(ROOT / "results" / "model_cost_cpu.csv")
        if (ROOT / "results" / "model_cost_cpu.csv").exists()
        else [],
        "history_best": next(
            (v["history"] for v in metrics.values() if v.get("exp_id") == "e1b"),
            [],
        ),
        "figures_e1b": {
            "confusion_matrix": metrics.get(e1b_key, {}).get("test", {}).get("confusion_matrix") if e1b_key else None
        },
        "class_order": [d["cls"] for d in DISEASES],
    }

    out_path = DATA_DIR / "site_data.json"
    out_path.write_text(json.dumps(site, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {out_path.relative_to(ROOT)}")
    print(f"  模型 {len(models)} 个 / 病害 {len(DISEASES)} 类 / 图片素材 {sum(len(v) if isinstance(v, list) else len(v) for v in images.values())} 组")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
