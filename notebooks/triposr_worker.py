from __future__ import annotations

import hashlib
import io
import multiprocessing as mp
import os
import signal
import socket
import threading
import time
import traceback
import uuid
from typing import Any
from urllib.parse import urljoin

import numpy as np
import rembg
import requests
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

BASE_URL = os.environ["BASE_URL"].rstrip("/") + "/"
TOKEN = os.environ["KAGGLE_HUB_TOKEN"]
MODEL_ID = os.getenv("TRIPOSR_MODEL_ID", "stabilityai/TripoSR")
REMBG_MODEL = os.getenv("TRIPOSR_REMBG_MODEL", "u2net")
POLL_TIMEOUT = 35
REQUEST_TIMEOUT = 120
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
    """Verify URL, protocol, TripoSR route, token, and shared SQLite state."""
    session = requests.Session()
    session.headers.update(auth_headers())
    response = session.get(api_url("/api/models"), params={"_ts": time.time_ns()}, timeout=20)
    checked_response(response, "GET /api/models")
    model_ids = {item.get("id") for item in response.json() if isinstance(item, dict)}
    if "triposr" not in model_ids:
        raise RuntimeError(f"Hub does not advertise triposr. Models={sorted(x for x in model_ids if x)}")

    # Authenticated read verifies the Bearer token without claiming a task.
    checked_response(
        session.get(api_url("/api/failed"), params={"_ts": time.time_ns()}, timeout=20),
        "GET /api/failed (auth check)",
    )
    snapshot = server_snapshot(session)
    if snapshot.get("storage") != "sqlite":
        raise RuntimeError(
            "Connected Hub is an old in-memory build. Update/restart the local Hub before starting 003."
        )
    instance_id = str(snapshot.get("hub_instance_id") or "")
    if not instance_id:
        raise RuntimeError("Hub did not return hub_instance_id; local Hub and 003 protocol versions differ")
    queued = int(snapshot.get("queued_by_model", {}).get("triposr", 0) or 0)
    inflight = int(snapshot.get("inflight_by_model", {}).get("triposr", 0) or 0)
    print(
        f"[preflight] Hub OK | instance={instance_id[:12]} | storage=sqlite | "
        f"triposr queued={queued} inflight={inflight}",
        flush=True,
    )
    return instance_id


def prefetch_model() -> None:
    from huggingface_hub import snapshot_download

    print(f"[prefetch] {MODEL_ID}", flush=True)
    snapshot_download(repo_id=MODEL_ID, allow_patterns=["config.yaml", "model.ckpt"])
    print("[prefetch] TripoSR assets ready", flush=True)


def build_rembg_session(gpu: int):
    providers = [
        ("CUDAExecutionProvider", {"device_id": gpu}),
        "CPUExecutionProvider",
    ]
    session = rembg.new_session(REMBG_MODEL, providers=providers)
    active = session.inner_session.get_providers()
    options = session.inner_session.get_provider_options()
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(f"rembg CUDA provider unavailable on GPU{gpu}: {active}")
    print(
        f"[GPU{gpu}] rembg={REMBG_MODEL} providers={active} options={options.get('CUDAExecutionProvider', {})}",
        flush=True,
    )
    return session


def prepare_image(raw: bytes, session, remove_bg: bool, foreground_ratio: float) -> Image.Image:
    image = Image.open(io.BytesIO(raw))
    if not remove_bg:
        return image.convert("RGB")

    image = remove_background(image, session)
    image = resize_foreground(image, foreground_ratio)
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * 0.5
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def export_mesh(mesh, output_format: str) -> bytes:
    data = mesh.export(file_type=output_format)
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


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
    device = f"cuda:{gpu}"
    gpu_name = torch.cuda.get_device_name(gpu)
    worker_id = f"triposr-{run_id}-g{gpu}"

    print(f"[GPU{gpu}] loading TripoSR on {gpu_name} ...", flush=True)
    started = time.perf_counter()
    model = TSR.from_pretrained(MODEL_ID, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(device)
    model.eval()
    rembg_session = build_rembg_session(gpu)
    print(f"[GPU{gpu}] READY in {time.perf_counter()-started:.2f}s", flush=True)

    session = requests.Session()
    session.headers.update(auth_headers())
    register_response = session.post(
        api_url("/worker/register"),
        json={
            "worker_id": worker_id,
            "model": "triposr",
            "gpus": [gpu_name],
            "runtime": "triposr-persistent-py310",
            "concurrency": 1,
            "meta": {
                "gpu_index": gpu,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "rembg_model": REMBG_MODEL,
                "rembg_providers": rembg_session.inner_session.get_providers(),
                "persistent": True,
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
                # POST cannot be cached as a stale 204 by a tunnel/CDN.
                response = session.post(
                    api_url("/task/claim"),
                    json={"model": "triposr", "worker_id": worker_id, "wait_seconds": 25},
                    timeout=POLL_TIMEOUT,
                )
                response_instance = response.headers.get("X-Hub-Instance", "")
                if response_instance and response_instance != hub_instance_id:
                    raise RuntimeError(
                        f"Hub instance changed: expected={hub_instance_id[:12]} got={response_instance[:12]}. "
                        "Tunnel may point at multiple unrelated Hub databases."
                    )
                if response.status_code == 204:
                    now = time.monotonic()
                    if now - last_idle_diagnostic >= IDLE_DIAGNOSTIC_SECONDS:
                        last_idle_diagnostic = now
                        snapshot = server_snapshot(session)
                        queued = int(snapshot.get("queued_by_model", {}).get("triposr", 0) or 0)
                        inflight = int(snapshot.get("inflight_by_model", {}).get("triposr", 0) or 0)
                        print(
                            f"[GPU{gpu}] idle | Hub triposr queued={queued} inflight={inflight} "
                            f"instance={str(snapshot.get('hub_instance_id', ''))[:12]}",
                            flush=True,
                        )
                    continue
                checked_response(response, "POST /task/claim")
                task = response.json()
                task_id = int(task["id"])
                active_task["id"] = task_id
                print(
                    f"[GPU{gpu}] ↓ #{task_id} {task.get('source_label','input')} "
                    f"res={task.get('mc_resolution',256)} fmt={task.get('output_format','glb')}",
                    flush=True,
                )

                t0 = time.perf_counter()
                input_response = session.get(api_url(task["input_url"]), timeout=60)
                input_response.raise_for_status()
                t_download = time.perf_counter()

                image = prepare_image(
                    input_response.content,
                    rembg_session,
                    bool(task.get("remove_background", True)),
                    float(task.get("foreground_ratio", 0.85)),
                )
                t_pre = time.perf_counter()

                model.renderer.set_chunk_size(int(task.get("chunk_size", 8192)))
                with torch.inference_mode():
                    scene_codes = model([image], device=device)
                t_model = time.perf_counter()

                with torch.inference_mode():
                    meshes = model.extract_mesh(
                        scene_codes,
                        True,
                        resolution=int(task.get("mc_resolution", 256)),
                    )
                t_mesh = time.perf_counter()

                mesh = meshes[0]
                output_format = str(task.get("output_format", "glb")).lower()
                artifact = export_mesh(mesh, output_format)
                encrypted = encrypt_blob(artifact)
                t_export = time.perf_counter()

                elapsed = t_export - t0
                upload = session.post(
                    api_url("/upload/artifact"),
                    data={
                        "id": str(task_id),
                        "model": "triposr",
                        "worker_id": worker_id,
                        "gpu": str(gpu),
                        "seconds": f"{elapsed:.3f}",
                        "output_format": output_format,
                        "vertices": str(len(mesh.vertices)),
                        "faces": str(len(mesh.faces)),
                    },
                    files={
                        "file": (
                            f"{task_id}.{output_format}.bin",
                            encrypted,
                            "application/octet-stream",
                        )
                    },
                    timeout=REQUEST_TIMEOUT,
                )
                upload.raise_for_status()
                t_up = time.perf_counter()
                active_task["id"] = None

                print(
                    f"[GPU{gpu}] ✓ #{task_id} total={elapsed:.2f}s "
                    f"download={t_download-t0:.2f}s rembg={t_pre-t_download:.2f}s "
                    f"model={t_model-t_pre:.2f}s mesh={t_mesh-t_model:.2f}s "
                    f"export={t_export-t_mesh:.2f}s upload={t_up-t_export:.2f}s "
                    f"v={len(mesh.vertices)} f={len(mesh.faces)}",
                    flush=True,
                )

                del image, scene_codes, meshes, mesh, artifact, encrypted
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No CUDA GPU found")

    wanted = int(os.getenv("TRIPOSR_GPU_COUNT", str(gpu_count)))
    gpu_count = min(gpu_count, max(1, wanted))
    run_id = f"{socket.gethostname()[:8]}-{uuid.uuid4().hex[:6]}"

    print(
        f"TripoSR persistent worker | GPUs={gpu_count} | rembg={REMBG_MODEL} | base={BASE_URL}",
        flush=True,
    )
    hub_instance_id = preflight_hub()
    prefetch_model()

    ctx = mp.get_context("spawn")
    processes = [
        ctx.Process(target=gpu_worker, args=(gpu, run_id, hub_instance_id), name=f"triposr-gpu{gpu}")
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
