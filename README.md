# rgzn · 基于迁移学习的植物叶片病害识别与鲁棒性分析

PlantVillage 番茄 6 类叶片病害分类，三组核心实验：**从零 vs 迁移学习**、**增强策略消融**、**亮度/旋转/模糊扰动下的鲁棒性**。

本项目不是"调一个模型"，而是"在统一协议下做可复现的对比实验"。因此**数据划分、评测指标、结果表是全组共用的**，任何人不得自行改动。

---

## 0. 环境说明

| 场景 | 环境 | 能做什么 |
|---|---|---|
| 本地 | Python 3.14（Windows） | 跑数据处理、划分、扰动、指标 —— **不需要 torch** |
| Kaggle / Colab | Python 3.11 + 预装 torch | 训练、评测 |

> 本地 Python 3.14 目前装不了 torch，这是正常的，不要试图在本地训练。本地只做数据准备，训练一律在 Kaggle（每账号每周 30 小时免费 GPU）。

### 0.1 仓库里没有图片

Clone 下来只有代码和 `data/splits/*.csv`，csv 里存的是**相对路径字符串**（如 `Tomato___Early_blight/xxx.JPG`），不是图片本身。所以：

| 你要做的事 | 需要真实数据集？ | 怎么做 |
|---|---|---|
| 写模型 / 调通链路 | 不需要 | 用 §1 的假数据，`make_dummy_data.py` 生成，本地 CPU 可跑 |
| 本地跑真实数据 | 需要 | 用 §2 的 `sparse-checkout`，约 224MB |
| Kaggle 出实验结果 | 需要 | Add Data 挂 `abdallahalidev/plantvillage-dataset`，见 §2.1 |

⚠️ 划分里的路径是**具体文件名**，数据必须与 spMohanty 那份同源（Kaggle mirror 是它的衍生版，文件名一致）。用了裁剪过或改名过的版本，`check_pipeline.py` 会直接报"图片文件存在 — 缺失 N 张"。

```bash
pip install numpy pandas pillow pyyaml matplotlib
```

---

## 1. 快速开始（写代码阶段，用假数据）

```bash
# 1) 生成假数据，让训练脚本当天就能跑通
python scripts/make_dummy_data.py --per-class 20

# 2) 用假数据生成划分
python scripts/prepare_data.py --root data/raw/dummy --out data/splits_dummy

# 3) 验收数据管线
python scripts/check_pipeline.py --splits data/splits_dummy --root data/raw/dummy
```

全部 `PASS` 说明数据侧可用（本地没装 torch 时最后一项会 `SKIP`）。

## 2. 真实数据

官方仓库：[spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
目录结构固定为 `raw/{color,grayscale,segmented}/<类别>/*.JPG`，全库约 54k 张。
训练用 `color`；`segmented`（去背景版）留给 E4 的跨域实验。

只拉番茄 6 类，不下全库（全库约 2GB）：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git data/raw/plantvillage
cd data/raw/plantvillage
git sparse-checkout set raw/color/Tomato___Bacterial_spot raw/color/Tomato___Early_blight raw/color/Tomato___Late_blight raw/color/Tomato___Leaf_Mold raw/color/Tomato___Septoria_leaf_spot raw/color/Tomato___healthy raw/segmented/Tomato___Bacterial_spot raw/segmented/Tomato___Early_blight raw/segmented/Tomato___Late_blight raw/segmented/Tomato___Leaf_Mold raw/segmented/Tomato___Septoria_leaf_spot raw/segmented/Tomato___healthy
cd ../../..
```

PowerShell 不支持 `\` 续行，所以上面写成一整行。`segmented` 是 E4 跨域实验用的，必须一起拉。若想全库，把 `git sparse-checkout set` 后面换成 `raw/color`。

最终路径形如 `data/raw/plantvillage/raw/color/Tomato___Early_blight/*.JPG`。
`configs/base.yaml` 的 `data.root` 保持 `data/raw/plantvillage` 即可 —— 脚本会自动下探到 `raw/color`。

然后生成正式划分（约 1 万张，几分钟）：

```bash
python scripts/prepare_data.py
```

产出 `data/splits/{train,val,test}.csv` + `meta.json` + `class_distribution.png`。
**这三个 csv 提交到仓库后即冻结，此后任何人不得重新生成。**

### 2.1 Kaggle 上怎么用

`abdallahalidev/plantvillage-dataset` 是同一份数据的 Kaggle 镜像，目录直接是 `color/ segmented/`，不用自己上传。Notebook 右侧 Add Data 挂载后：

```python
from src.config import load_config

cfg = load_config("configs/base.yaml")
cfg["data"]["root"] = "/kaggle/input/plantvillage-dataset"   # 含 color/ 的那一层
```

脚本会自动识别 `color/` 与 `segmented/`，其余配置不用改。

---

## 3. 目录

```
configs/     base.yaml —— 所有实验从它继承
data/
  raw/       原始数据（不入库）
  splits/    train/val/test.csv + meta.json（入库、只读）
src/
  config.py  配置加载与 --set 覆盖
  data/      dedupe / splits / dataset / transforms / corruptions
  models/    (B、C 负责)
  train.py   (B 负责)
  eval.py    (B 负责，D 调用)
  utils/     metrics / table / seed
scripts/     prepare_data / make_dummy_data / check_pipeline
results/     results.csv（唯一结果表）+ figures/
report/      报告
```

---

## 4. 三条铁律

### 4.1 划分文件

`data/splits/{train,val,test}.csv`，列固定：

| 列 | 说明 |
|---|---|
| `path` | 相对 `color/` 的路径 |
| `class_name` | 类别目录名 |
| `group_id` | 近重复分组 id（同一片叶子的多张照片同组，不会跨集合） |

划分方式：**pHash 近重复聚类 + 类内分层 + 按组整体划分**（70/15/15）。
这避免了 PlantVillage 同叶多拍导致的测试集虚高。

想要"泄漏版"划分做对比（报告加分项）：`python scripts/prepare_data.py --dedupe off --out data/splits_leaky`。

### 4.2 结果表

`results/results.csv`，一行 = 一次评测，**长表格式**，只追加不改写：

```
exp_id, model, pretrained, aug, seed, data_ratio, split, corruption, severity, acc, macro_f1, params_m, notes
```

- 干净测试集：`corruption=clean, severity=0`
- 扰动测试集：`corruption=brightness|rotation|blur, severity=1..5`
- E1b 低数据量：`data_ratio=0.1|0.25|0.5|1.0`

汇总多个 seed：

```bash
python -m src.utils.table --path results/results.csv --out results/summary.csv
```

写结果统一用 `src.utils.table.append_row("results/results.csv", row)`，不要手写 csv。

### 4.3 配置

所有实验差异只能写在 yaml 里，或命令行临时覆盖：

```bash
python -m src.train --config configs/base.yaml --set train.epochs=5 model.name=mobilenet_v3_small
```

命名规范：`e1b_resnet18_ft_r10_s42`（实验_模型_是否预训练_数据比例_seed）。

---

## 5. 分工与接口

| | 负责 | 已交付 / 待交付 |
|---|---|---|
| **A** | 数据与基建 | ✅ `splits`、`dedupe`、`corruptions`、`dataset`、`transforms`、`metrics`、`table`、仓库骨架、`configs/e0_resnet18_ft.yaml` |
| **B** | 模型与训练 | ✅ `src/models/cnn.py`、`build.py`、`src/train.py`、`src/eval.py` |
| **C** | MobileNet + 可解释 + Demo | ⬜ `mobilenet_v3_small`、Grad-CAM、`app.py` |
| **D** | 实验编排与分析 | ⬜ E2/E3/E4、图表、报告 |

**B 的 `train.py` 约定**（D 会依赖）：

```bash
python -m src.train --config configs/xxx.yaml --seed 42
# 产物: runs/<run_name>/best.pth, metrics.json, 并自动 append 一行到 results.csv
```

- `train.freeze_backbone: true` 时冻结主干、只训练分类 head（E0 用，见 `configs/e0_resnet18_ft.yaml`）；`false` 时整体微调。`model.pretrained=false` 时该字段忽略。

**C 的 `build.py` 约定**：

```python
build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module
```

**A 提供的可直接调用接口**：

```python
from src.data.dataset import build_loaders
from src.data.transforms import build_corrupted_eval_transform
from src.utils.metrics import summarize
from src.utils.table import append_row
```

---

## 6. 扰动实验怎么跑

```python
loaders = build_loaders(cfg, corruption="blur", severity=3, splits=("test",))
```

`brightness` / `rotation` / `blur` 各 5 级，参数见 `src/data/corruptions.py` 的 `SEVERITY_MAP`。
**测试时扰动只在 eval 用，训练时不要动。**

---

## 6.1 跨域实验（E4：color → segmented）

`data.variant` 决定读哪一套图，默认 `color`，可填 `segmented`（去背景）。划分文件两套共用，保证同组图片不跨 split。

| 配置 | 用途 |
|---|---|
| `configs/e4a_color2seg.yaml` | color 训练的权重直接在 segmented 上评测（**不训练**，D 做） |
| `configs/e4b_seg2seg.yaml` | 训练与测试都在 segmented，作为上界对照（B 做） |

```bash
# E4a 零样本跨域：拿 E1b 的权重直接测
python -m src.eval --config configs/e4a_color2seg.yaml \
    --ckpt runs/e1b_resnet18_ft_r100_s42/best.pth --exp-id e4a

# E4b 域内上界：在 segmented 上重训一次
python -m src.train --config configs/e4b_seg2seg.yaml --seed 42
```

segmented 的文件名规则是 `<原文件名>_final_masked.jpg`，映射写在 `src/data/dataset.py` 的 `resolve_image_path` 里，9350 张已全部对齐。

---

## 7. 验收

```bash
python scripts/check_pipeline.py
```

8 项检查：划分文件 / 字段 / 无泄漏 / 图片可读 / 抽样可解码 / 扰动生效 / 指标正确 / Dataset 可用（本地无 torch 时最后一项 SKIP，须在 Kaggle 补验）。
全绿才算 A 岗完成，其他人才能开工。

---

## 8. C 岗：MobileNetV3 / Grad-CAM / Demo

模型注册表在 `src/models/build.py`：`build_model(name, num_classes=6, pretrained=True)`，
可选 `cnn | resnet18 | mobilenet_v3_small`。三个模型都拆成 `.backbone / .head`，
训练统一走 B 的 `src/train.py`。注册表里另有 C 岗用的 `get_target_layer / load_checkpoint /
guess_model_name / count_parameters`（Grad-CAM 与 Demo 依赖）。

```bash
# E1e：MobileNetV3 预训练全量微调（与 E1b 的 resnet18 ft 同配置，只换模型）
python -m src.train --config configs/e1_mobilenet_v3_small.yaml --seed 42
# 从零训练对照
python -m src.train --config configs/e1_mobilenet_v3_small.yaml --seed 42 \
    --set model.pretrained=false --run-name e1e_mobilenet_v3_small_scratch_r100_s42

# 本地没数据也能先把流程跑通（假数据，CPU 即可）
python scripts/make_dummy_data.py --per-class 20
python scripts/prepare_data.py --root data/raw/dummy --out data/splits_dummy
python -m src.train --config configs/e1_mobilenet_v3_small.yaml --seed 42 \
    --set data.root=data/raw/dummy data.split_dir=data/splits_dummy train.epochs=3 data.num_workers=0 \
    --run-name smoke_mobilenet --no-append

# 参数量 / FLOPs / 推理延迟 → results/model_cost.csv
python scripts/profile_models.py
# 轻量模型在 CPU 上的优势更明显，报告里可两份都放
python scripts/profile_models.py --device cpu --out results/model_cost_cpu.csv

# Grad-CAM：扰动图直接来自 build_loaders，不改数据管线
python -m src.models.gradcam --ckpt runs/e1e_mobilenet_v3_small_ft_r100_s42/best.pth \
    --corruption blur --severity 4 --num 8
python -m src.models.gradcam --ckpt runs/e1e_mobilenet_v3_small_ft_r100_s42/best.pth --corruption clean

# Demo：上传叶片 → 预测 + 热力图
python app.py --ckpt runs/e1e_mobilenet_v3_small_ft_r100_s42/best.pth
```

`app.py` 的预处理直接复用 `build_eval_transform`（Resize 256 → CenterCrop 224 → Normalize），
与评测同一条链路；改尺寸或 mean/std 只改 yaml，不要在 demo 里另写一套。
`app.py` 用 `server_name="127.0.0.1"` 启动，本机开着系统代理时也不会在启动自检阶段 502。
