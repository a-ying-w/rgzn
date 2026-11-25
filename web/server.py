"""TomatoGuard 前端后端：静态托管 + 真实模型推理。

    python web/server.py                    # http://127.0.0.1:8000
    python web/server.py --port 8010
    python web/server.py --ckpt runs/e1b_resnet18_ft_r100_s42/best.pth

只读复用 src/ 下的模型与预处理，不写 runs/、results/，也不改动 app.py（Gradio Demo）。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import secrets
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# 本地 torch + matplotlib 同时加载会触发 OpenMP 重复链接（见项目经验记录）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fastapi import Cookie, FastAPI, File, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.transforms import build_eval_transform  # noqa: E402
from src.models.build import build_model, count_parameters, guess_model_name, load_checkpoint  # noqa: E402
from src.models.gradcam import GradCAM, denormalize, overlay_cam  # noqa: E402

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

CLASS_CN = {
    "Tomato___Bacterial_spot": "番茄细菌性斑点病",
    "Tomato___Early_blight": "番茄早疫病",
    "Tomato___Late_blight": "番茄晚疫病",
    "Tomato___Leaf_Mold": "番茄叶霉病",
    "Tomato___Septoria_leaf_spot": "番茄斑枯病",
    "Tomato___healthy": "番茄健康叶片",
}
CLASS_EN = {
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato___healthy": "Healthy",
}
CLASS_DISEASE_ID = {
    "Tomato___Bacterial_spot": "bacterial-spot",
    "Tomato___Early_blight": "early-blight",
    "Tomato___Late_blight": "late-blight",
    "Tomato___Leaf_Mold": "leaf-mold",
    "Tomato___Septoria_leaf_spot": "septoria-leaf-spot",
    "Tomato___healthy": "healthy",
}

STATE: dict = {}
LOCK = threading.Lock()

# ---------------------------------------------------------------- Auth（文件 + 内存 session，课程项目级别）

AUTH_DIR = ROOT / "web" / "data"
USERS_FILE = AUTH_DIR / "users.json"
SESSIONS: dict[str, str] = {}  # sid -> username
SESSIONS_EXPIRE: dict[str, float] = {}  # sid -> expire timestamp
AUTH_COOKIE = "tg_sid"
AUTH_TTL_SEC = 7 * 24 * 3600


class _AuthRequest(BaseModel):
    username: str = Field(min_length=2, max_length=32)
    password: str = Field(min_length=4, max_length=64)


def _hash_pw(password: str, salt: bytes | None = None) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return f"pbkdf2_sha256$120000${salt.hex()}${dk.hex()}"


def _verify_pw(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, dk_hex = stored.split("$")
        assert algo == "pbkdf2_sha256"
        assert int(iters) == 120_000
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iters))
        return secrets.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_users(users: dict) -> None:
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def _seed_users() -> None:
    """首次启动时写入默认账号：admin/admin123、user/user123。"""
    if USERS_FILE.exists():
        return
    now = datetime.now(timezone.utc).isoformat()
    users = {
        "admin": {
            "username": "admin",
            "password": _hash_pw("admin123"),
            "role": "admin",
            "created_at": now,
        },
        "user": {
            "username": "user",
            "password": _hash_pw("user123"),
            "role": "user",
            "created_at": now,
        },
    }
    _save_users(users)


def _user_public(username: str) -> dict | None:
    users = _load_users()
    u = users.get(username)
    if not u:
        return None
    return {
        "username": u["username"],
        "role": u.get("role", "user"),
        "created_at": u.get("created_at"),
    }


def _current_user(sid: str | None) -> dict | None:
    if not sid or sid not in SESSIONS or SESSIONS_EXPIRE.get(sid, 0) < time.time():
        if sid and sid in SESSIONS:
            SESSIONS.pop(sid, None)
            SESSIONS_EXPIRE.pop(sid, None)
        return None
    return _user_public(SESSIONS[sid])


def _new_sid(username: str) -> str:
    sid = secrets.token_urlsafe(32)
    SESSIONS[sid] = username
    SESSIONS_EXPIRE[sid] = time.time() + AUTH_TTL_SEC
    return sid


def _cleanup_expired() -> None:
    now = time.time()
    dead = [sid for sid, exp in SESSIONS_EXPIRE.items() if exp < now]
    for sid in dead:
        SESSIONS.pop(sid, None)
        SESSIONS_EXPIRE.pop(sid, None)


_seed_users()


def init_model(ckpt: str, config_path: str, device_name: str | None, reject_threshold: float) -> None:
    cfg = load_config(str(ROOT / config_path))
    dcfg = cfg["data"]
    classes: list[str] = dcfg["classes"]
    ckpt_path = (ROOT / ckpt) if not Path(ckpt).is_absolute() else Path(ckpt)
    if not ckpt_path.exists():
        raise SystemExit(
            f"找不到权重 {ckpt_path}\n先训练：python -m src.train --config configs/e1_resnet18_ft.yaml --seed 42"
        )

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = guess_model_name(ckpt_path, cfg["model"]["name"])
    model = build_model(model_name, num_classes=len(classes), pretrained=False)
    load_checkpoint(model, ckpt_path)
    model.to(device).eval()

    STATE.update(
        {
            "cfg": cfg,
            "classes": classes,
            "model": model,
            "model_name": model_name,
            "params_m": count_parameters(model) / 1e6,
            "device": device,
            "ckpt": str(ckpt_path),
            "cam": GradCAM(model),
            "mean": dcfg.get("mean") or DEFAULT_MEAN,
            "std": dcfg.get("std") or DEFAULT_STD,
            "eval_tf": build_eval_transform(
                image_size=dcfg["image_size"],
                resize_size=dcfg["resize_size"],
                mean=dcfg.get("mean") or DEFAULT_MEAN,
                std=dcfg.get("std") or DEFAULT_STD,
            ),
            "reject_threshold": reject_threshold,
        }
    )


def to_data_url(array: np.ndarray, fmt: str = "JPEG", quality: int = 88) -> str:
    """uint8 [H, W, 3] → data URL。"""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, fmt, quality=quality, optimize=True)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_payload(filename: str, image: Image.Image) -> dict:
    classes = STATE["classes"]
    device = STATE["device"]
    x = STATE["eval_tf"](image).unsqueeze(0).to(device)

    started = time.perf_counter()
    with LOCK:
        # Grad-CAM 需要反传梯度，不能包在 inference_mode 里
        cam, probs = STATE["cam"].explain(x)
    elapsed_ms = (time.perf_counter() - started) * 1000

    order = np.argsort(probs)[::-1]
    top3 = [
        {
            "class_name": classes[int(i)],
            "cn": CLASS_CN.get(classes[int(i)], classes[int(i)]),
            "en": CLASS_EN.get(classes[int(i)], classes[int(i)].split("___")[-1]),
            "prob": float(probs[int(i)]),
            "disease_id": CLASS_DISEASE_ID.get(classes[int(i)], "healthy"),
        }
        for i in order[:3]
    ]

    rgb = denormalize(x[0], STATE["mean"], STATE["std"])
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    heat_rgb = (STATE["cmap"](cam)[..., :3] * 255).astype(np.uint8)
    overlay = overlay_cam(rgb, cam)

    top1 = top3[0]
    uncertain = top1["prob"] < STATE["reject_threshold"]
    note = None
    if uncertain:
        note = (
            f"不确定：图片特征与六类番茄叶片差异较大（最高置信度 {top1['prob']:.1%}），"
            "请尝试拍摄叶片正面、光照均匀、背景简洁的照片。"
        )

    return {
        "ok": True,
        "filename": filename,
        "model": STATE["model_name"],
        "device": str(device),
        "params_m": round(STATE["params_m"], 3),
        "inference_ms": round(elapsed_ms, 1),
        "top1": top1,
        "top3": top3,
        "uncertain": bool(uncertain),
        "note": note,
        "images": {
            "original": to_data_url(rgb_u8),
            "heatmap": to_data_url(heat_rgb),
            "overlay": to_data_url(overlay),
        },
        "input_size": [image.width, image.height],
    }


def create_app(static_dir: Path) -> FastAPI:
    app = FastAPI(title="TomatoGuard API", version="1.0")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "model": STATE["model_name"],
            "params_m": round(STATE["params_m"], 3),
            "device": str(STATE["device"]),
            "ckpt": STATE["ckpt"],
            "classes": [
                {"class_name": c, "cn": CLASS_CN.get(c, c), "en": CLASS_EN.get(c, c)}
                for c in STATE["classes"]
            ],
            "reject_threshold": STATE["reject_threshold"],
        }

    @app.get("/api/site-data")
    def site_data() -> JSONResponse:
        path = ROOT / "web" / "data" / "site_data.json"
        if not path.exists():
            raise HTTPException(500, "site_data.json 不存在，请先运行 python scripts/build_web_data.py")
        return JSONResponse(__import__("json").loads(path.read_text(encoding="utf-8")))

    @app.post("/api/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "空文件")
        if len(raw) > 15 * 1024 * 1024:
            raise HTTPException(413, "图片超过 15MB，请压缩后再上传")
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(400, "无法解析图片，请上传 JPG / PNG 格式")
        if min(image.size) < 32:
            raise HTTPException(400, "图片尺寸过小，至少 32×32")
        return build_payload(file.filename or "upload.jpg", image)

    # ---------------------------------------------------------------- Auth

    @app.post("/api/auth/register")
    def auth_register(body: _AuthRequest) -> dict:
        if body.username.lower() in ("admin", "user"):
            raise HTTPException(400, "该用户名已被占用")
        users = _load_users()
        if body.username in users:
            raise HTTPException(400, "该用户名已被占用")
        users[body.username] = {
            "username": body.username,
            "password": _hash_pw(body.password),
            "role": "user",  # 新注册一律普通用户
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_users(users)
        sid = _new_sid(body.username)
        resp = JSONResponse({"ok": True, "user": _user_public(body.username)})
        resp.set_cookie(AUTH_COOKIE, sid, httponly=True, samesite="lax",
                        max_age=AUTH_TTL_SEC, path="/")
        return resp

    @app.post("/api/auth/login")
    def auth_login(body: _AuthRequest) -> JSONResponse:
        users = _load_users()
        u = users.get(body.username)
        if not u or not _verify_pw(body.password, u.get("password", "")):
            raise HTTPException(401, "用户名或密码错误")
        sid = _new_sid(body.username)
        resp = JSONResponse({"ok": True, "user": _user_public(body.username)})
        resp.set_cookie(AUTH_COOKIE, sid, httponly=True, samesite="lax",
                        max_age=AUTH_TTL_SEC, path="/")
        return resp

    @app.post("/api/auth/logout")
    def auth_logout(tg_sid: Optional[str] = Cookie(default=None)) -> JSONResponse:
        if tg_sid and tg_sid in SESSIONS:
            SESSIONS.pop(tg_sid, None)
            SESSIONS_EXPIRE.pop(tg_sid, None)
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(AUTH_COOKIE, path="/")
        return resp

    @app.get("/api/auth/me")
    def auth_me(tg_sid: Optional[str] = Cookie(default=None)) -> dict:
        _cleanup_expired()
        user = _current_user(tg_sid)
        return {"user": user}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="TomatoGuard 前端与推理服务")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--ckpt", default="runs/e1b_resnet18_ft_r100_s42/best.pth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--reject-threshold", type=float, default=0.6)
    args = parser.parse_args()

    init_model(args.ckpt, args.config, args.device, args.reject_threshold)
    # jet 色表：Grad-CAM 热力图与 app.py 保持一致
    from matplotlib import colormaps

    STATE["cmap"] = colormaps["jet"]

    static_dir = ROOT / "web" / "static"
    app = create_app(static_dir)

    import uvicorn

    print(f"TomatoGuard · http://{args.host}:{args.port}")
    print(f"模型 {STATE['model_name']}（{STATE['params_m']:.2f}M） · {STATE['device']} · {STATE['ckpt']}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
