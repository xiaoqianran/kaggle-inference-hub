from __future__ import annotations

import gc
import hashlib
import io
import multiprocessing as mp
import os
import signal
import socket
import sys
import tempfile
import threading
import time
import traceback
import uuid
from argparse import Namespace
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import requests
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image

BASE_URL = os.environ["BASE_URL"].rstrip("/") + "/"
TOKEN = os.environ["KAGGLE_HUB_TOKEN"]
ROOT = Path(os.getenv("FAST_SAM3D_ROOT", "/kaggle/working/Fast-SAM3D"))
NOTEBOOK_DIR = ROOT / "notebook"
CHECKPOINT_DIR = Path(os.getenv("FAST_SAM3D_CHECKPOINT_DIR", str(NOTEBOOK_DIR / "checkpoints/hf")))
TORCH_CACHE = Path(os.getenv("FAST_SAM3D_TORCH_CACHE", "/kaggle/working/torch-cache"))
MODEL = "fast-sam3d"
POLL_TIMEOUT = 35
REQUEST_TIMEOUT = 180
HEARTBEAT_SECONDS = 10
IDLE_DIAGNOSTIC_SECONDS = 60


def encrypt_blob(data: bytes) -> bytes:
    key = hashlib.sha256(TOKEN.encode()).digest()
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, data, None)


def api_url(path: str) -> str:
    return urljoin(BASE_URL, path.lstrip("/"))


def auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Cache-Control": "no-cache, no-store",
        "Pragma": "no-cache",
    }


def checked_response(response: requests.Response, label: str) -> requests.Response:
    if response.status_code >= 400:
        body = response.text[:1000].replace("\n", " ")
        raise RuntimeError(f"{label}: HTTP {response.status_code} | {body}")
    return response


def server_snapshot(session: requests.Session) -> dict[str, Any]:
    response = session.get(api_url("/api/status"), params={"_ts": time.time_ns()}, timeout=15)
    checked_response(response, "GET /api/status")
    return response.json()


def preflight_hub() -> str:
    session = requests.Session()
    session.headers.update(auth_headers())
    response = session.get(api_url("/api/models"), params={"_ts": time.time_ns()}, timeout=20)
    checked_response(response, "GET /api/models")
    model_ids = {item.get("id") for item in response.json() if isinstance(item, dict)}
    if MODEL not in model_ids:
        raise RuntimeError(f"Hub does not advertise {MODEL}. Update/restart Hub first.")
    checked_response(
        session.get(api_url("/api/failed"), params={"_ts": time.time_ns()}, timeout=20),
        "GET /api/failed (auth check)",
    )
    snapshot = server_snapshot(session)
    if snapshot.get("storage") != "sqlite":
        raise RuntimeError("Connected Hub is an old in-memory build. Update/restart Hub first.")
    instance_id = str(snapshot.get("hub_instance_id") or "")
    if not instance_id:
        raise RuntimeError("Hub did not return hub_instance_id; protocol versions differ")
    queued = int(snapshot.get("queued_by_model", {}).get(MODEL, 0) or 0)
    inflight = int(snapshot.get("inflight_by_model", {}).get(MODEL, 0) or 0)
    print(
        f"[preflight] Hub OK | instance={instance_id[:12]} | storage=sqlite | "
        f"{MODEL} queued={queued} inflight={inflight}",
        flush=True,
    )
    return instance_id


def validate_runtime() -> None:
    pipeline = CHECKPOINT_DIR / "pipeline.yaml"
    if not ROOT.is_dir():
        raise RuntimeError(f"Fast-SAM3D root not found: {ROOT}")
    if not pipeline.is_file():
        raise RuntimeError(f"Fast-SAM3D checkpoint config not found: {pipeline}")
    if not (NOTEBOOK_DIR / "inference.py").is_file():
        raise RuntimeError(f"Fast-SAM3D inference.py not found under {NOTEBOOK_DIR}")
    TORCH_CACHE.mkdir(parents=True, exist_ok=True)


def _rewrite_paths(value: Any) -> Any:
    if isinstance(value, str):
        replacements = {
            "/data3/wmq/Fast-sam3d-objects/checkpoints/torch-cache": str(TORCH_CACHE),
            "/data3/wmq/Fast-sam3d-objects/checkpoints/": str(NOTEBOOK_DIR / "checkpoints") + "/",
        }
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, dict):
        return {key: _rewrite_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_paths(item) for item in value]
    return value


def inference_args(enable_acceleration: bool = True) -> Namespace:
    return Namespace(
        ss_faster_stride=3,
        ss_warmup=2,
        ss_order=1,
        ss_momentum_beta=0.5,
        slat_thresh=0.5,
        slat_warmup=2,
        slat_token_ratio=0.15,
        mesh_spectral_threshold_low=0.5,
        mesh_spectral_threshold_high=0.7,
        enable_ss_faster=enable_acceleration,
        enable_slat_token=enable_acceleration,
        enable_mesh_aggregation=enable_acceleration,
        enable_acceleration=enable_acceleration,
        enable_taylor=False,
        enable_easy=False,
    )


def build_inference(enable_acceleration: bool = True):
    os.environ.setdefault("CONDA_PREFIX", "/opt/conda")
    os.environ["TORCH_HOME"] = str(TORCH_CACHE)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(NOTEBOOK_DIR))
    sys.path.insert(0, str(ROOT))
    os.chdir(NOTEBOOK_DIR)

    from omegaconf import OmegaConf
    from inference import Inference

    config = OmegaConf.load(CHECKPOINT_DIR / "pipeline.yaml")
    plain = OmegaConf.to_container(config, resolve=False)
    config = OmegaConf.create(_rewrite_paths(plain))
    config.workspace_dir = str(CHECKPOINT_DIR)
    if enable_acceleration:
        config["ss_generator_config_path"] = "ss_generator_faster.yaml"
        config["slat_generator_config_path"] = "slat_generator_faster.yaml"

    args = inference_args(enable_acceleration)
    inference = Inference(config, compile=False, args=args)
    if hasattr(inference, "get_params"):
        inference.get_params(args)
    return inference


def prepare_inputs(image_raw: bytes, mask_raw: bytes, directory: Path):
    from fft.fft2d import calculate_hfer_robust

    image = Image.open(io.BytesIO(image_raw)).convert("RGB")
    mask_image = Image.open(io.BytesIO(mask_raw)).convert("L")
    if mask_image.size != image.size:
        raise ValueError(f"Mask size {mask_image.size} must match image size {image.size}")
    mask_path = directory / "mask.png"
    mask_image.save(mask_path)
    image_array = np.asarray(image, dtype=np.uint8)
    mask_array = np.asarray(mask_image, dtype=np.uint8) > 0
    if not mask_array.any():
        raise ValueError("Mask is empty")
    hfer = calculate_hfer_robust(str(mask_path))
    return image_array, mask_array, hfer


def export_glb(glb: Any, path: Path) -> bytes:
    glb.export(str(path))
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"glTF":
        raise RuntimeError("Fast-SAM3D export did not produce a valid GLB")
    return data


def heartbeat_loop(
    worker_id: str,
    gpu: int,
    stop: threading.Event,
    active_task: dict[str, int | None],
) -> None:
    session = requests.Session()
    session.headers.update(auth_headers())
    while not stop.wait(HEARTBEAT_SECONDS):
        try:
            session.post(
                api_url("/worker/heartbeat"),
                json={
                    "worker_id": worker_id,
                    "local_queue": 0,
                    "upload_queue": 0,
                    "active_task_id": active_task["id"],
                    "meta": {"gpu_index": gpu, "persistent": True},
                },
                timeout=15,
            ).raise_for_status()
        except Exception as exc:
            print(f"[GPU{gpu}] heartbeat: {type(exc).__name__}: {exc}", flush=True)


def report_failure(session: requests.Session, task_id: int, exc: BaseException, gpu: int) -> None:
    message = f"{type(exc).__name__}: {exc}"
    print(f"[GPU{gpu}] FAIL #{task_id}: {message}", flush=True)
    try:
        session.post(
            api_url("/task/fail"),
            json={"id": task_id, "error": message[:1900], "requeue": True},
            timeout=20,
        ).raise_for_status()
    except Exception as report_exc:
        print(f"[GPU{gpu}] fail-report error: {report_exc}", flush=True)


def gpu_worker(gpu: int, run_id: str, hub_instance_id: str) -> None:
    torch.cuda.set_device(gpu)
    gpu_name = torch.cuda.get_device_name(gpu)
    worker_id = f"fast-sam3d-{run_id}-g{gpu}"

    print(f"[GPU{gpu}] loading Fast-SAM3D on {gpu_name} ...", flush=True)
    started = time.perf_counter()
    inference = build_inference(enable_acceleration=True)
    print(f"[GPU{gpu}] READY in {time.perf_counter()-started:.2f}s", flush=True)

    session = requests.Session()
    session.headers.update(auth_headers())
    register_response = session.post(
        api_url("/worker/register"),
        json={
            "worker_id": worker_id,
            "model": MODEL,
            "gpus": [gpu_name],
            "runtime": "fast-sam3d-persistent-py311",
            "concurrency": 1,
            "meta": {
                "gpu_index": gpu,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "persistent": True,
                "checkpoint_dir": str(CHECKPOINT_DIR),
            },
        },
        timeout=30,
    )
    checked_response(register_response, "POST /worker/register")
    print(f"[GPU{gpu}] registered as {worker_id}", flush=True)

    stop = threading.Event()
    active_task: dict[str, int | None] = {"id": None}
    heartbeat = threading.Thread(
        target=heartbeat_loop,
        args=(worker_id, gpu, stop, active_task),
        daemon=True,
    )
    heartbeat.start()

    def shutdown(*_args):
        stop.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    last_idle_diagnostic = 0.0

    try:
        while True:
            task: dict[str, Any] | None = None
            try:
                response = session.post(
                    api_url("/task/claim"),
                    json={"model": MODEL, "worker_id": worker_id, "wait_seconds": 25},
                    timeout=POLL_TIMEOUT,
                )
                response_instance = response.headers.get("X-Hub-Instance", "")
                if response_instance and response_instance != hub_instance_id:
                    raise RuntimeError(
                        f"Hub instance changed: expected={hub_instance_id[:12]} got={response_instance[:12]}"
                    )
                if response.status_code == 204:
                    now = time.monotonic()
                    if now - last_idle_diagnostic >= IDLE_DIAGNOSTIC_SECONDS:
                        last_idle_diagnostic = now
                        snapshot = server_snapshot(session)
                        queued = int(snapshot.get("queued_by_model", {}).get(MODEL, 0) or 0)
                        inflight = int(snapshot.get("inflight_by_model", {}).get(MODEL, 0) or 0)
                        print(
                            f"[GPU{gpu}] idle | Hub {MODEL} queued={queued} inflight={inflight} "
                            f"instance={str(snapshot.get('hub_instance_id', ''))[:12]}",
                            flush=True,
                        )
                    continue
                checked_response(response, "POST /task/claim")
                task = response.json()
                task_id = int(task["id"])
                active_task["id"] = task_id
                seed = int(task.get("seed", 42))
                print(
                    f"[GPU{gpu}] ↓ #{task_id} {task.get('source_label', 'input')} seed={seed}",
                    flush=True,
                )

                t0 = time.perf_counter()
                image_response = session.get(api_url(task["input_url"]), timeout=60)
                image_response.raise_for_status()
                mask_response = session.get(api_url(task["mask_url"]), timeout=60)
                mask_response.raise_for_status()
                t_download = time.perf_counter()

                with tempfile.TemporaryDirectory(prefix=f"fastsam3d-{task_id}-") as temp_dir:
                    temp = Path(temp_dir)
                    image, mask, hfer = prepare_inputs(image_response.content, mask_response.content, temp)
                    t_pre = time.perf_counter()

                    if hasattr(inference, "get_hfer"):
                        inference.get_hfer(hfer)
                    with torch.inference_mode():
                        output = inference(image, mask, seed=seed)
                    t_model = time.perf_counter()

                    artifact = export_glb(output["glb"], temp / "result.glb")
                    encrypted = encrypt_blob(artifact)
                    t_export = time.perf_counter()

                elapsed = t_export - t0
                upload = session.post(
                    api_url("/upload/artifact"),
                    data={
                        "id": str(task_id),
                        "model": MODEL,
                        "worker_id": worker_id,
                        "gpu": str(gpu),
                        "seconds": f"{elapsed:.3f}",
                        "output_format": "glb",
                    },
                    files={
                        "file": (
                            f"{task_id}.glb.bin",
                            encrypted,
                            "application/octet-stream",
                        )
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                checked_response(upload, "POST /upload/artifact")
                t_upload = time.perf_counter()
                active_task["id"] = None

                print(
                    f"[GPU{gpu}] ✓ #{task_id} total={elapsed:.2f}s "
                    f"download={t_download-t0:.2f}s preprocess={t_pre-t_download:.2f}s "
                    f"model={t_model-t_pre:.2f}s export={t_export-t_model:.2f}s "
                    f"upload={t_upload-t_export:.2f}s",
                    flush=True,
                )

                del image, mask, hfer, output, artifact, encrypted
                gc.collect()
                torch.cuda.empty_cache()

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if task is not None and "id" in task:
                    report_failure(session, int(task["id"]), exc, gpu)
                    active_task["id"] = None
                else:
                    print(f"[GPU{gpu}] poll error: {type(exc).__name__}: {exc}", flush=True)
                    time.sleep(2)
                traceback.print_exc()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        print(f"[GPU{gpu}] stopped", flush=True)


def main() -> None:
    validate_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No CUDA GPU found")
    wanted = int(os.getenv("FAST_SAM3D_GPU_COUNT", str(gpu_count)))
    gpu_count = min(gpu_count, max(1, wanted))
    run_id = f"{socket.gethostname()[:8]}-{uuid.uuid4().hex[:6]}"

    print(f"Fast-SAM3D persistent worker | GPUs={gpu_count} | base={BASE_URL}", flush=True)
    hub_instance_id = preflight_hub()

    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(target=gpu_worker, args=(gpu, run_id, hub_instance_id), name=f"fast-sam3d-gpu{gpu}")
        for gpu in range(gpu_count)
    ]
    for process in processes:
        process.start()

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("Stopping workers ...", flush=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)


if __name__ == "__main__":
    mp.freeze_support()
    main()
