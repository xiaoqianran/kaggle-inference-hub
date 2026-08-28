from __future__ import annotations

import gc
import hashlib
import json
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass(frozen=True)
class HunyuanConfig:
    root: Path = Path("/kaggle/working/Hunyuan3D-2.1")
    image: Path = Path("/kaggle/working/Hunyuan3D-2.1/assets/demo.png")
    workdir: Path = Path("/kaggle/working")
    shape_steps: int = 20
    octree_resolution: int = 256
    paint_views: int = 4
    paint_resolution: int = 256
    render_size: int = 1024
    texture_size: int = 2048

    @property
    def shape_obj(self) -> Path:
        return self.workdir / "hunyuan21-shape.obj"

    @property
    def shape_glb(self) -> Path:
        return self.workdir / "hunyuan21-shape.glb"

    @property
    def pbr_obj(self) -> Path:
        return self.workdir / "hunyuan21-pbr.obj"

    @property
    def pbr_glb(self) -> Path:
        return self.workdir / "hunyuan21-pbr.glb"


CFG = HunyuanConfig()


def prepare_imports() -> None:
    sys.path[:0] = [str(CFG.root / "hy3dshape"), str(CFG.root / "hy3dpaint")]
    os.chdir(CFG.root)


def clear_cuda() -> None:
    gc.collect()
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()


def reset_peaks() -> None:
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.reset_peak_memory_stats()


def memory_snapshot(label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            free, _ = torch.cuda.mem_get_info()
            rows.append(
                {
                    "gpu": index,
                    "free_gib": round(free / 2**30, 2),
                    "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
                    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }
            )
    print(f"[HY21_MEM] {label} {json.dumps(rows)}", flush=True)
    return rows


def run_shape(cfg: HunyuanConfig = CFG) -> float:
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    started = time.perf_counter()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1",
        subfolder="hunyuan3d-dit-v2-1",
        use_safetensors=False,
        torch_dtype=torch.float16,
    )
    pipeline.to("cuda:0")
    if not callable(pipeline):
        raise TypeError("Hunyuan shape pipeline is not callable after device placement")
    print(f"[HY21] shape_load={time.perf_counter() - started:.2f}s", flush=True)
    memory_snapshot("shape-loaded")

    started = time.perf_counter()
    with torch.inference_mode():
        mesh = pipeline(
            image=str(cfg.image),
            num_inference_steps=cfg.shape_steps,
            octree_resolution=cfg.octree_resolution,
            guidance_scale=5.0,
        )[0]
    seconds = time.perf_counter() - started
    mesh.export(cfg.shape_obj)
    mesh.export(cfg.shape_glb)
    print(
        f"[HY21_SHAPE] seconds={seconds:.3f} glb={cfg.shape_glb.stat().st_size}",
        flush=True,
    )

    del mesh, pipeline
    clear_cuda()
    memory_snapshot("shape-released")
    return seconds


def run_paint(cfg: HunyuanConfig = CFG) -> float:
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    config = Hunyuan3DPaintConfig(
        max_num_view=cfg.paint_views, resolution=cfg.paint_resolution
    )
    config.render_size = cfg.render_size
    config.texture_size = cfg.texture_size
    realesrgan = cfg.root / "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    if not realesrgan.is_file():
        raise FileNotFoundError(realesrgan)
    config.realesrgan_ckpt_path = str(realesrgan)

    started = time.perf_counter()
    pipeline = Hunyuan3DPaintPipeline(config)
    print(f"[HY21] paint_load={time.perf_counter() - started:.2f}s", flush=True)
    memory_snapshot("paint-loaded")

    started = time.perf_counter()
    pipeline(
        mesh_path=str(cfg.shape_obj),
        image_path=str(cfg.image),
        output_mesh_path=str(cfg.pbr_obj),
        use_remesh=True,
        save_glb=False,
    )
    seconds = time.perf_counter() - started
    if not cfg.pbr_obj.is_file():
        raise FileNotFoundError(cfg.pbr_obj)
    print(
        f"[HY21_PAINT] seconds={seconds:.3f} obj={cfg.pbr_obj.stat().st_size}",
        flush=True,
    )
    memory_snapshot("paint-finished")
    return seconds


def export_pbr_glb(cfg: HunyuanConfig = CFG) -> None:
    from convert_utils import create_glb_with_pbr_materials

    base = cfg.pbr_obj.with_suffix("")
    assets = {
        "obj": cfg.pbr_obj,
        "mtl": base.with_suffix(".mtl"),
        "albedo": Path(f"{base}.jpg"),
        "metallic": Path(f"{base}_metallic.jpg"),
        "roughness": Path(f"{base}_roughness.jpg"),
    }
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing PBR assets: {missing}")

    textures = {name: str(assets[name]) for name in ("albedo", "metallic", "roughness")}
    create_glb_with_pbr_materials(str(cfg.pbr_obj), textures, str(cfg.pbr_glb))
    if not cfg.pbr_glb.is_file() or cfg.pbr_glb.stat().st_size < 1000:
        raise RuntimeError(
            "PBR GLB export failed or produced an unexpectedly small file"
        )
    print(f"[HY21_EXPORT] glb={cfg.pbr_glb.stat().st_size}", flush=True)


MODEL = "hunyuan3d-2.1"
POLL_TIMEOUT = 35
REQUEST_TIMEOUT = 180
HEARTBEAT_SECONDS = 10


def encrypt_blob(data: bytes, token: str) -> bytes:
    key = hashlib.sha256(token.encode()).digest()
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, data, None)


def api_url(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def auth_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Cache-Control": "no-cache, no-store",
        "Pragma": "no-cache",
    }


def checked_response(response: requests.Response, label: str) -> requests.Response:
    if response.status_code >= 400:
        body = response.text[:1000].replace("\n", " ")
        raise RuntimeError(f"{label}: HTTP {response.status_code} | {body}")
    return response


def preflight_hub(base_url: str, token: str, session: requests.Session) -> str:
    response = session.get(api_url(base_url, "/api/models"), params={"_ts": time.time_ns()}, timeout=20)
    checked_response(response, "GET /api/models")
    model_ids = {item.get("id") for item in response.json() if isinstance(item, dict)}
    if MODEL not in model_ids:
        raise RuntimeError(f"Hub does not advertise {MODEL}. Update/restart Hub first.")
    checked_response(
        session.get(api_url(base_url, "/api/failed"), params={"_ts": time.time_ns()}, timeout=20),
        "GET /api/failed (auth check)",
    )
    snapshot = session.get(api_url(base_url, "/api/status"), params={"_ts": time.time_ns()}, timeout=20)
    checked_response(snapshot, "GET /api/status")
    payload = snapshot.json()
    instance_id = str(payload.get("hub_instance_id") or "")
    if payload.get("storage") != "sqlite" or not instance_id:
        raise RuntimeError("Hub protocol is too old; SQLite state + hub_instance_id are required")
    print(
        f"[preflight] Hub OK | instance={instance_id[:12]} | "
        f"queued={payload.get('queued_by_model', {}).get(MODEL, 0)} "
        f"inflight={payload.get('inflight_by_model', {}).get(MODEL, 0)}",
        flush=True,
    )
    return instance_id


def heartbeat_loop(
    base_url: str, token: str, worker_id: str, stop: threading.Event, active_task: dict[str, int | None]
) -> None:
    session = requests.Session()
    session.headers.update(auth_headers(token))
    while not stop.wait(HEARTBEAT_SECONDS):
        try:
            session.post(
                api_url(base_url, "/worker/heartbeat"),
                json={
                    "worker_id": worker_id,
                    "local_queue": 0,
                    "upload_queue": 0,
                    "active_task_id": active_task["id"],
                    "meta": {"persistent": True, "dual_t4": True},
                },
                timeout=15,
            ).raise_for_status()
        except Exception as exc:
            print(f"[heartbeat] {type(exc).__name__}: {exc}", flush=True)


def report_failure(
    base_url: str, session: requests.Session, task_id: int, exc: BaseException
) -> None:
    message = f"{type(exc).__name__}: {exc}"
    print(f"[HY21] FAIL #{task_id}: {message}", flush=True)
    try:
        session.post(
            api_url(base_url, "/task/fail"),
            json={"id": task_id, "error": message[:1900], "requeue": True},
            timeout=20,
        ).raise_for_status()
    except Exception as report_exc:
        print(f"[HY21] fail-report error: {report_exc}", flush=True)


def run_hub_task(
    base_url: str, token: str, session: requests.Session, worker_id: str, task: dict[str, Any]
) -> None:
    task_id = int(task["id"])
    started = time.perf_counter()
    response = session.get(api_url(base_url, task["input_url"]), timeout=60)
    checked_response(response, "GET task input")
    suffix = Path(str(task.get("source_label") or "input.png")).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"

    with tempfile.TemporaryDirectory(prefix=f"hunyuan3d21-{task_id}-") as tmp:
        workdir = Path(tmp)
        image_path = workdir / f"input{suffix}"
        image_path.write_bytes(response.content)
        cfg = replace(
            CFG,
            image=image_path,
            workdir=workdir,
            shape_steps=int(task.get("shape_steps", CFG.shape_steps)),
            octree_resolution=int(task.get("octree_resolution", CFG.octree_resolution)),
            paint_views=int(task.get("paint_views", CFG.paint_views)),
            paint_resolution=int(task.get("paint_resolution", CFG.paint_resolution)),
            texture_size=int(task.get("texture_size", CFG.texture_size)),
        )
        reset_peaks()
        shape_seconds = run_shape(cfg)
        paint_seconds = run_paint(cfg)
        export_pbr_glb(cfg)
        artifact = cfg.pbr_glb.read_bytes()
        encrypted = encrypt_blob(artifact, token)
        elapsed = time.perf_counter() - started
        upload = session.post(
            api_url(base_url, "/upload/artifact"),
            data={
                "id": str(task_id),
                "model": MODEL,
                "worker_id": worker_id,
                "gpu": "0",
                "seconds": f"{elapsed:.3f}",
                "output_format": "glb",
            },
            files={"file": (f"{task_id}.glb.bin", encrypted, "application/octet-stream")},
            timeout=REQUEST_TIMEOUT,
        )
        checked_response(upload, "POST /upload/artifact")
        print(
            f"[HY21] ✓ #{task_id} total={elapsed:.2f}s shape={shape_seconds:.2f}s "
            f"paint={paint_seconds:.2f}s glb={len(artifact)}",
            flush=True,
        )


def hub_main() -> None:
    prepare_imports()
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"2 GPUs required, got {torch.cuda.device_count()}")
    base_url = os.environ["BASE_URL"].rstrip("/")
    token = os.environ["KAGGLE_HUB_TOKEN"]
    run_id = f"{socket.gethostname()[:8]}-{uuid.uuid4().hex[:6]}"
    worker_id = f"hunyuan3d21-{run_id}"
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    session = requests.Session()
    session.headers.update(auth_headers(token))
    hub_instance_id = preflight_hub(base_url, token, session)
    register = session.post(
        api_url(base_url, "/worker/register"),
        json={
            "worker_id": worker_id,
            "model": MODEL,
            "gpus": gpu_names,
            "runtime": "hunyuan3d21-persistent-py310",
            "concurrency": 1,
            "meta": {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "persistent": True},
        },
        timeout=30,
    )
    checked_response(register, "POST /worker/register")
    print(f"[HY21] registered {worker_id} | GPUs={gpu_names}", flush=True)

    stop = threading.Event()
    active_task: dict[str, int | None] = {"id": None}
    threading.Thread(
        target=heartbeat_loop,
        args=(base_url, token, worker_id, stop, active_task),
        daemon=True,
    ).start()
    try:
        while True:
            task: dict[str, Any] | None = None
            try:
                response = session.post(
                    api_url(base_url, "/task/claim"),
                    json={"model": MODEL, "worker_id": worker_id, "wait_seconds": 25},
                    timeout=POLL_TIMEOUT,
                )
                response_instance = response.headers.get("X-Hub-Instance", "")
                if response_instance and response_instance != hub_instance_id:
                    raise RuntimeError(
                        f"Hub instance changed: expected={hub_instance_id[:12]} got={response_instance[:12]}"
                    )
                if response.status_code == 204:
                    continue
                checked_response(response, "POST /task/claim")
                task = response.json()
                active_task["id"] = int(task["id"])
                print(f"[HY21] ↓ #{task['id']} {task.get('source_label', 'input')}", flush=True)
                run_hub_task(base_url, token, session, worker_id, task)
                active_task["id"] = None
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if task is not None and "id" in task:
                    report_failure(base_url, session, int(task["id"]), exc)
                    active_task["id"] = None
                else:
                    print(f"[HY21] poll error: {type(exc).__name__}: {exc}", flush=True)
                    time.sleep(2)
                traceback.print_exc()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


def benchmark_main() -> None:
    prepare_imports()
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"2 GPUs required, got {torch.cuda.device_count()}")
    reset_peaks()
    shape_seconds = run_shape()
    paint_seconds = run_paint()
    export_pbr_glb()
    summary = {
        "shape_seconds": round(shape_seconds, 3),
        "paint_seconds": round(paint_seconds, 3),
        "shape_glb_bytes": CFG.shape_glb.stat().st_size,
        "pbr_glb_bytes": CFG.pbr_glb.stat().st_size,
    }
    (CFG.workdir / "hunyuan21-benchmark.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[HY21_RESULT] {json.dumps(summary)}", flush=True)
    print("✅ Hunyuan3D 2.1 Shape + Paint 完整自检通过", flush=True)


def main() -> None:
    if os.getenv("HUNYUAN3D21_HUB_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
        hub_main()
    else:
        benchmark_main()


if __name__ == "__main__":
    main()
