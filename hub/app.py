from __future__ import annotations

import asyncio
import json
import queue
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import (
    DEFAULT_MODEL,
    ARTIFACT_MAX_BYTES,
    INPUT_MAX_BYTES,
    MODEL_SPECS,
    OUTPUT_DIR,
    PORT,
    PROMPT_AI_API_KEY,
    PROMPT_AI_BASE_URL,
    PROMPT_AI_CONCURRENCY,
    PROMPT_AI_ENABLED,
    PROMPT_AI_MAX_TOKENS,
    PROMPT_AI_MODEL,
    PROMPT_AI_TEMPERATURE,
    PROMPT_AI_TIMEOUT_SECONDS,
    TOKEN,
    canonical_model,
)
from .crypto import decrypt_blob
from .prompt_pipeline import PromptPipeline, PromptPipelineError, PromptPipelineSettings
from .state import state

app = FastAPI(title="Kaggle Inference Hub", version="0.5.0")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR = Path(__file__).with_name("web")


@app.middleware("http")
async def hub_response_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(("/assets/", "/images/", "/outputs/")):
        # Fingerprinted frontend and completed output files are immutable.
        # Long-lived caching keeps repeated previews/downloads off the Hub.
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    else:
        # Worker polling and diagnostics must never be cached by a tunnel/CDN.
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    response.headers["X-Hub-Instance"] = state.instance_id
    return response

prompt_pipeline = PromptPipeline(
    PromptPipelineSettings(
        enabled=PROMPT_AI_ENABLED,
        base_url=PROMPT_AI_BASE_URL,
        api_key=PROMPT_AI_API_KEY,
        model=PROMPT_AI_MODEL,
        timeout_seconds=PROMPT_AI_TIMEOUT_SECONDS,
        concurrency=PROMPT_AI_CONCURRENCY,
        max_tokens=PROMPT_AI_MAX_TOKENS,
        temperature=PROMPT_AI_TEMPERATURE,
    )
)


class TaskIn(BaseModel):
    prompt: str
    source_prompt: str | None = None
    prompt_meta: dict = Field(default_factory=dict)
    model: str = DEFAULT_MODEL
    seed: int | None = None
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    steps: int | None = Field(default=None, ge=1, le=200)


class BatchIn(BaseModel):
    prompts: list[str]
    source_prompts: list[str] | None = None
    prompt_meta: dict = Field(default_factory=dict)
    prompt_metas: list[dict] | None = None
    model: str = DEFAULT_MODEL
    seed: int | None = None
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    steps: int | None = Field(default=None, ge=1, le=200)


class WorkerIn(BaseModel):
    worker_id: str
    model: str
    gpus: list[str] = Field(default_factory=list)
    runtime: str = "kaggle"
    concurrency: int = 2
    meta: dict = Field(default_factory=dict)


class HeartbeatIn(BaseModel):
    worker_id: str
    local_queue: int = 0
    upload_queue: int = 0
    active_task_id: int | None = None
    meta: dict = Field(default_factory=dict)


class FailIn(BaseModel):
    id: int
    error: str
    requeue: bool = True


class ClaimIn(BaseModel):
    model: str = DEFAULT_MODEL
    worker_id: str
    wait_seconds: float = Field(default=25, ge=0, le=30)


class PromptProcessIn(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL
    mode: str = "enhance"
    translate_to_english: bool = True


class PromptBatchProcessIn(BaseModel):
    prompts: list[str]
    model: str = DEFAULT_MODEL
    mode: str = "enhance"
    translate_to_english: bool = True


def auth(authorization: str | None) -> None:
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def model_or_400(value: str | None) -> str:
    try:
        return canonical_model(value)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown model: {value}")


def prompt_model_or_400(value: str | None) -> str:
    model = model_or_400(value)
    if MODEL_SPECS[model].input_kind != "prompt":
        raise HTTPException(status_code=400, detail=f"Model does not accept prompts: {model}")
    return model


def image_suffix(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


def local_output_path(url: str) -> Path:
    path = unquote(urlsplit(url).path)
    prefixes = ("/images/", "/outputs/")
    prefix = next((item for item in prefixes if path.startswith(item)), None)
    if prefix is None:
        raise HTTPException(status_code=400, detail="source_url must point to a Hub output")
    relative = path[len(prefix):]
    root = OUTPUT_DIR.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Source image does not exist")
    if candidate.stat().st_size > INPUT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Source image is too large")
    if image_suffix(candidate.read_bytes()[:16]) is None:
        raise HTTPException(status_code=400, detail="Source must be PNG, JPEG, or WebP")
    return candidate


def sanitize_prompt_meta(meta: dict | None) -> dict:
    if not meta:
        return {}
    allowed = {
        "mode",
        "provider_model",
        "elapsed_ms",
        "translate_to_english",
        "target_model",
        "edited_after_ai",
        "stale_model_adapter",
    }
    return {key: meta[key] for key in allowed if key in meta}


def make_task(
    prompt: str,
    model: str,
    width: int,
    height: int,
    steps: int | None,
    seed: int | None,
    *,
    source_prompt: str | None = None,
    prompt_meta: dict | None = None,
) -> dict:
    spec = MODEL_SPECS[model]
    task = {
        "id": state.next_id(),
        "model": model,
        "prompt": prompt,
        "seed": seed if seed is not None else int(time.time_ns() % 2_147_483_647),
        "width": width,
        "height": height,
        "steps": steps if steps is not None else spec.default_steps,
        "created_at": time.time(),
        "attempt": 0,
    }
    source = (source_prompt or "").strip()
    if source and source != prompt:
        task["source_prompt"] = source
    clean_meta = sanitize_prompt_meta(prompt_meta)
    if clean_meta:
        task["prompt_meta"] = clean_meta
    return task


async def broadcast(item: dict) -> None:
    payload = json.dumps(item, ensure_ascii=False)
    dead = []
    for ws in list(state.clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.clients.discard(ws)


@app.get("/api/models")
def models():
    return [
        {
            "id": x.id,
            "label": x.label,
            "default_steps": x.default_steps,
            "description": x.description,
            "input_kind": x.input_kind,
            "output_kind": x.output_kind,
        }
        for x in MODEL_SPECS.values()
    ]


@app.get("/api/prompt/pipeline")
def prompt_pipeline_config():
    return prompt_pipeline.public_config()


@app.post("/api/prompt/process")
async def process_prompt(x: PromptProcessIn, authorization: str | None = Header(None)):
    auth(authorization)
    model = prompt_model_or_400(x.model)
    try:
        return await prompt_pipeline.process(
            x.prompt,
            target_model=model,
            mode=x.mode,
            translate_to_english=x.translate_to_english,
        )
    except PromptPipelineError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/prompt/process/batch")
async def process_prompt_batch(x: PromptBatchProcessIn, authorization: str | None = Header(None)):
    auth(authorization)
    model = prompt_model_or_400(x.model)
    prompts = [p.strip() for p in x.prompts if p.strip()]
    if not prompts:
        raise HTTPException(status_code=400, detail="No prompts")
    if len(prompts) > 200:
        raise HTTPException(status_code=400, detail="AI batch is limited to 200 prompts")

    async def one(prompt: str) -> dict:
        try:
            result = await prompt_pipeline.process(
                prompt,
                target_model=model,
                mode=x.mode,
                translate_to_english=x.translate_to_english,
            )
            # Batch UI is intentionally one-line-per-task. Collapse model line breaks
            # so one AI result can never accidentally become multiple GPU tasks.
            result["processed"] = " ".join(
                part.strip() for part in result["processed"].splitlines() if part.strip()
            )
            return {"ok": True, **result}
        except PromptPipelineError as exc:
            return {"ok": False, "original": prompt, "processed": prompt, "error": str(exc)}

    items = await asyncio.gather(*(one(prompt) for prompt in prompts))
    return {
        "total": len(items),
        "succeeded": sum(1 for item in items if item["ok"]),
        "failed": sum(1 for item in items if not item["ok"]),
        "items": items,
    }


@app.post("/task", status_code=202)
def add_task(x: TaskIn, authorization: str | None = Header(None)):
    auth(authorization)
    prompt = x.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")
    model = prompt_model_or_400(x.model)
    task = make_task(
        prompt,
        model,
        x.width,
        x.height,
        x.steps,
        x.seed,
        source_prompt=x.source_prompt,
        prompt_meta=x.prompt_meta,
    )
    try:
        state.enqueue(task)
    except queue.Full:
        raise HTTPException(status_code=503, detail=f"Task queue is full: {model}")
    snap = state.snapshot()
    return {"queued": 1, "queue_size": snap["queued_by_model"][model], "task": task}


@app.post("/task/batch", status_code=202)
def add_batch(x: BatchIn, authorization: str | None = Header(None)):
    auth(authorization)
    model = prompt_model_or_400(x.model)
    if x.source_prompts is not None and len(x.source_prompts) != len(x.prompts):
        raise HTTPException(status_code=400, detail="source_prompts length must match prompts")
    if x.prompt_metas is not None and len(x.prompt_metas) != len(x.prompts):
        raise HTTPException(status_code=400, detail="prompt_metas length must match prompts")
    pairs = []
    for i, raw in enumerate(x.prompts):
        prompt = raw.strip()
        if not prompt:
            continue
        source = None
        if x.source_prompts is not None:
            source = x.source_prompts[i].strip()
        meta = x.prompt_metas[i] if x.prompt_metas is not None else x.prompt_meta
        pairs.append((prompt, source, meta))
    if not pairs:
        raise HTTPException(status_code=400, detail="No prompts")
    items = []
    for i, (prompt, source_prompt, prompt_meta) in enumerate(pairs):
        seed = x.seed + i if x.seed is not None else None
        task = make_task(
            prompt,
            model,
            x.width,
            x.height,
            x.steps,
            seed,
            source_prompt=source_prompt,
            prompt_meta=prompt_meta,
        )
        items.append(task)
    try:
        state.enqueue_many(items)
    except queue.Full:
        raise HTTPException(status_code=503, detail=f"Not enough queue capacity: {model}")
    return {"queued": len(items), "queue_size": state.queue_size(model), "tasks": items}


@app.post("/task/triposr", status_code=202)
async def add_triposr_task(
    file: UploadFile | None = File(None),
    source_url: str = Form(""),
    output_format: str = Form("glb"),
    mc_resolution: int = Form(256),
    chunk_size: int = Form(8192),
    foreground_ratio: float = Form(0.85),
    remove_background: bool = Form(True),
    authorization: str | None = Header(None),
):
    auth(authorization)
    source_url = source_url.strip()
    if (file is None) == (not source_url):
        raise HTTPException(status_code=400, detail="Provide exactly one of file or source_url")
    output_format = output_format.strip().lower()
    if output_format not in {"glb", "obj"}:
        raise HTTPException(status_code=400, detail="output_format must be glb or obj")
    if not 128 <= mc_resolution <= 512:
        raise HTTPException(status_code=400, detail="mc_resolution must be between 128 and 512")
    if not 1024 <= chunk_size <= 131072:
        raise HTTPException(status_code=400, detail="chunk_size must be between 1024 and 131072")
    if not 0.5 <= foreground_ratio <= 1.0:
        raise HTTPException(status_code=400, detail="foreground_ratio must be between 0.5 and 1.0")

    task_id = state.next_id()
    owned_input = False
    if file is not None:
        data = await file.read(INPUT_MAX_BYTES + 1)
        if len(data) > INPUT_MAX_BYTES:
            raise HTTPException(status_code=413, detail="Input image is too large")
        suffix = image_suffix(data)
        if suffix is None:
            raise HTTPException(status_code=400, detail="Input must be PNG, JPEG, or WebP")
        input_dir = OUTPUT_DIR / "_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / f"triposr_{int(time.time()*1000)}_{task_id:06d}{suffix}"
        await asyncio.to_thread(input_path.write_bytes, data)
        source_label = Path(file.filename or f"upload{suffix}").name
        public_source_url = f"/outputs/_inputs/{input_path.name}"
        owned_input = True
    else:
        input_path = local_output_path(source_url)
        source_label = input_path.name
        public_source_url = urlsplit(source_url).path

    task = {
        "id": task_id,
        "model": "triposr",
        "input_url": f"/task/input/{task_id}",
        "source_url": public_source_url,
        "source_label": source_label,
        "output_format": output_format,
        "mc_resolution": mc_resolution,
        "chunk_size": chunk_size,
        "foreground_ratio": foreground_ratio,
        "remove_background": remove_background,
        "created_at": time.time(),
        "attempt": 0,
        "_input_path": str(input_path.resolve()),
        "_owned_input": owned_input,
    }
    try:
        state.enqueue(task)
    except queue.Full:
        if owned_input:
            input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Task queue is full: triposr")
    return {
        "queued": 1,
        "queue_size": state.queue_size("triposr"),
        "task": {key: value for key, value in task.items() if not key.startswith("_")},
    }


@app.get("/task/next")
async def next_task(
    model: str = Query(DEFAULT_MODEL),
    worker_id: str = Query("legacy-worker"),
    authorization: str | None = Header(None),
):
    auth(authorization)
    model = model_or_400(model)
    task = await asyncio.to_thread(state.claim, model, worker_id, 25)
    if task is None:
        return Response(status_code=204)
    state.heartbeat(worker_id, {"model": model, "last_claimed_task": task["id"]})
    return {key: value for key, value in task.items() if not key.startswith("_")}


@app.post("/task/claim")
async def claim_task(x: ClaimIn, authorization: str | None = Header(None)):
    """Non-cacheable task claim endpoint used by current workers.

    GET /task/next remains available for 001/002 and older workers.
    """
    auth(authorization)
    model = model_or_400(x.model)
    task = await asyncio.to_thread(state.claim, model, x.worker_id, x.wait_seconds)
    if task is None:
        return Response(status_code=204)
    state.heartbeat(x.worker_id, {"model": model, "last_claimed_task": task["id"]})
    return {key: value for key, value in task.items() if not key.startswith("_")}


@app.get("/task/input/{task_id}")
def task_input(task_id: int, authorization: str | None = Header(None)):
    auth(authorization)
    task = state.lease_task(task_id)
    if task is None or task.get("model") != "triposr":
        raise HTTPException(status_code=404, detail="TripoSR task is not inflight")
    path = Path(task.get("_input_path", ""))
    root = OUTPUT_DIR.resolve()
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise HTTPException(status_code=404, detail="Input image does not exist")
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Input image does not exist")
    return FileResponse(resolved, filename=task.get("source_label") or resolved.name)


@app.post("/task/fail")
def task_fail(x: FailIn, authorization: str | None = Header(None)):
    auth(authorization)
    try:
        requeued, task = state.fail(x.id, x.error[:2000], x.requeue)
    except queue.Full:
        raise HTTPException(status_code=503, detail="Queue full while requeueing")
    if task is None:
        raise HTTPException(status_code=404, detail="Task is not inflight")
    return {"id": x.id, "requeued": requeued, "attempt": task.get("attempt", 0)}


@app.post("/worker/register")
def worker_register(x: WorkerIn, authorization: str | None = Header(None)):
    auth(authorization)
    model = model_or_400(x.model)
    return state.register_worker({
        "worker_id": x.worker_id,
        "model": model,
        "gpus": x.gpus,
        "runtime": x.runtime,
        "concurrency": x.concurrency,
        "meta": x.meta,
    })


@app.post("/worker/heartbeat")
def worker_heartbeat(x: HeartbeatIn, authorization: str | None = Header(None)):
    auth(authorization)
    lease_renewed = False
    if x.active_task_id is not None:
        lease_renewed = state.renew_lease(x.active_task_id, x.worker_id)
    item = state.heartbeat(x.worker_id, {
        "local_queue": x.local_queue,
        "upload_queue": x.upload_queue,
        "active_task_id": x.active_task_id,
        "meta": x.meta,
    })
    if item is None:
        raise HTTPException(status_code=404, detail="Worker not registered")
    return {**item, "lease_renewed": lease_renewed}


@app.get("/api/status")
def status():
    return state.snapshot()


@app.get("/api/history")
def get_history(model: str | None = None, after: int = Query(0, ge=0), limit: int = Query(300, ge=1, le=500)):
    canonical = model_or_400(model) if model else None
    return state.history_items(canonical, after=after, limit=limit)


@app.get("/api/failed")
def get_failed(authorization: str | None = Header(None)):
    auth(authorization)
    return state.failed_items(200)


@app.post("/upload/artifact")
async def upload_artifact(
    file: UploadFile = File(...),
    id: int = Form(...),
    gpu: int = Form(...),
    seconds: float = Form(0),
    model: str = Form("triposr"),
    worker_id: str = Form(...),
    output_format: str = Form("glb"),
    vertices: int | None = Form(None),
    faces: int | None = Form(None),
    authorization: str | None = Header(None),
):
    auth(authorization)
    model = model_or_400(model)
    if MODEL_SPECS[model].output_kind != "artifact":
        raise HTTPException(status_code=400, detail=f"Model does not upload artifacts: {model}")
    owner = state.lease_owner(id)
    leased_task = state.lease_task(id)
    if leased_task is None:
        raise HTTPException(status_code=409, detail="Task is no longer inflight")
    if owner != worker_id:
        raise HTTPException(status_code=409, detail=f"Task lease belongs to {owner}, not {worker_id}")
    output_format = output_format.strip().lower()
    if output_format not in {"glb", "obj"} or output_format != leased_task.get("output_format"):
        raise HTTPException(status_code=400, detail="Artifact format does not match the task")
    encrypted = await file.read(ARTIFACT_MAX_BYTES + 29)
    if len(encrypted) > ARTIFACT_MAX_BYTES + 28:
        raise HTTPException(status_code=413, detail="Encrypted artifact is too large")
    try:
        data = await asyncio.to_thread(decrypt_blob, encrypted, TOKEN)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Decrypt failed: {type(exc).__name__}")
    if len(data) > ARTIFACT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Artifact is too large")
    if output_format == "glb" and (len(data) < 12 or data[:4] != b"glTF"):
        raise HTTPException(status_code=400, detail="Artifact is not a valid GLB")
    if output_format == "obj":
        try:
            obj_head = data[:65536].decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Artifact is not a text OBJ")
        if not any(line.startswith("v ") for line in obj_head.splitlines()):
            raise HTTPException(status_code=400, detail="OBJ has no vertices")

    model_dir = OUTPUT_DIR / model
    model_dir.mkdir(parents=True, exist_ok=True)
    name = f"{int(time.time()*1000)}_{id:06d}_gpu{gpu}.{output_format}"
    path = model_dir / name
    await asyncio.to_thread(path.write_bytes, data)
    item = {
        "kind": "artifact",
        "id": id,
        "model": model,
        "worker_id": worker_id,
        "gpu": gpu,
        "seconds": seconds,
        "output_format": output_format,
        "vertices": vertices,
        "faces": faces,
        "source_url": leased_task.get("source_url", ""),
        "source_label": leased_task.get("source_label", "input image"),
        "download_url": f"/outputs/{model}/{name}",
        "time": time.time(),
    }
    try:
        item = state.finish(id, item)
    except KeyError:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail="Task is no longer inflight")
    state.heartbeat(worker_id, {"last_completed_task": id})
    await broadcast(item)
    return item


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    id: int = Form(...),
    gpu: int = Form(...),
    seed: int = Form(...),
    prompt: str = Form(""),
    seconds: float = Form(0),
    model: str = Form(DEFAULT_MODEL),
    worker_id: str = Form("legacy-worker"),
    steps: int | None = Form(None),
    authorization: str | None = Header(None),
):
    auth(authorization)
    model = model_or_400(model)
    if MODEL_SPECS[model].output_kind != "image":
        raise HTTPException(status_code=400, detail=f"Model does not upload images: {model}")
    owner = state.lease_owner(id)
    leased_task = state.lease_task(id)
    if owner is not None and worker_id != "legacy-worker" and owner != worker_id:
        raise HTTPException(status_code=409, detail=f"Task lease belongs to {owner}, not {worker_id}")
    try:
        encrypted = await file.read()
        data = await asyncio.to_thread(decrypt_blob, encrypted, TOKEN)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Decrypt failed: {type(exc).__name__}")

    model_dir = OUTPUT_DIR / model
    model_dir.mkdir(parents=True, exist_ok=True)
    name = f"{int(time.time()*1000)}_{id:06d}_gpu{gpu}_seed{seed}.webp"
    path = model_dir / name
    await asyncio.to_thread(path.write_bytes, data)

    item = {
        "kind": "image",
        "id": id,
        "model": model,
        "worker_id": worker_id,
        "gpu": gpu,
        "seed": seed,
        "steps": steps,
        "prompt": prompt,
        "seconds": seconds,
        "url": f"/images/{model}/{name}",
        "time": time.time(),
    }
    if leased_task:
        if leased_task.get("source_prompt"):
            item["source_prompt"] = leased_task["source_prompt"]
        if leased_task.get("prompt_meta"):
            item["prompt_meta"] = leased_task["prompt_meta"]
    try:
        item = state.finish(id, item)
    except KeyError:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail="Task is no longer inflight")
    state.heartbeat(worker_id, {"last_completed_task": id})
    await broadcast(item)
    return item


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        state.clients.discard(websocket)


@app.get("/", response_class=FileResponse)
def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Frontend is not built. Run `pnpm --dir frontend build`.",
        )
    return FileResponse(index_path)


app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets", check_dir=False), name="frontend-assets")
