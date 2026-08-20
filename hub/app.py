from __future__ import annotations

import asyncio
import json
import queue
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, Response
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


@app.middleware("http")
async def hub_response_headers(request: Request, call_next):
    response = await call_next(request)
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


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


INDEX_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kaggle Inference Hub</title>
<style>
*{box-sizing:border-box}:root{--bg:#11111b;--panel:#181825;--panel2:#1e1e2e;--border:#313244;--text:#cdd6f4;--muted:#7f849c;--blue:#89b4fa;--green:#a6e3a1;--red:#f38ba8;--purple:#cba6f7;--peach:#fab387}
body{margin:0;background:var(--bg);color:var(--text);font:14px Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}.shell{max-width:1760px;margin:auto;padding:20px}.top{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:18px}.title{font-size:24px;font-weight:850}.sub{color:var(--muted);font-size:12px;margin-top:4px}.status{display:flex;gap:8px;flex-wrap:wrap}.pill{background:var(--panel2);border:1px solid var(--border);border-radius:999px;padding:7px 11px;color:#bac2de}.dot{width:8px;height:8px;border-radius:50%;background:var(--red);display:inline-block;margin-right:6px}.online .dot{background:var(--green)}
.layout{display:grid;grid-template-columns:460px minmax(0,1fr);gap:18px;align-items:start}.left{display:flex;flex-direction:column;gap:14px}.panel{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:16px;box-shadow:0 15px 40px #0003}.panel-head{display:flex;justify-content:space-between;gap:12px;margin-bottom:12px}.panel h2{margin:0;font-size:15px}.label{font-size:10px;color:#9399b2;margin:10px 0 6px;font-weight:800;letter-spacing:.06em}textarea,input,select{width:100%;background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:10px;padding:10px;outline:none}textarea{resize:vertical;line-height:1.55}textarea:focus,input:focus,select:focus{border-color:var(--blue)}#singlePrompt{min-height:210px}#batchPrompts{min-height:150px}.row{display:grid;grid-template-columns:1fr 1fr;gap:9px}.actions{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}.btn{border:0;border-radius:10px;padding:10px 13px;font-weight:800;cursor:pointer}.btn:disabled{opacity:.45;cursor:not-allowed}.primary{background:var(--blue);color:var(--bg);flex:1}.batch-primary{background:var(--purple);color:var(--bg);flex:1}.ai-btn{background:var(--green);color:var(--bg)}.ghost{background:var(--border);color:var(--text)}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px}.stat{background:var(--bg);border-radius:10px;padding:10px}.stat b{display:block;font-size:16px}.stat span{color:var(--muted);font-size:9px}.workers{display:grid;gap:8px}.worker{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:10px}.worker-top{display:flex;justify-content:space-between;gap:8px}.worker-name{font-weight:800}.worker-meta{font-size:10px;color:var(--muted);margin-top:5px}.ok{color:var(--green)}.off{color:var(--red)}.ai-panel{border-color:#45475a;background:linear-gradient(180deg,#1e1e2e,#181825)}.ai-status{font-size:10px;font-weight:850;padding:5px 8px;border-radius:999px;border:1px solid var(--border);color:var(--muted)}.ai-status.ready{color:var(--green);border-color:#405b46}.ai-status.disabled{color:var(--red);border-color:#5d3a45}.check{display:flex;align-items:center;gap:8px;color:#bac2de;font-size:12px;margin-top:11px}.check input{width:auto}.ai-hint{margin-top:10px;padding:9px 10px;border-radius:9px;background:var(--bg);color:var(--muted);font-size:11px}.restore{display:none}
.gallery-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}.gallery{columns:4 240px;column-gap:12px}.card{break-inside:avoid;margin-bottom:12px;background:var(--panel2);border:1px solid var(--border);border-radius:13px;overflow:hidden;animation:in .2s ease}.card img{width:100%;display:block}.info{padding:10px}.meta{font-size:10px;color:var(--blue)}.model{font-size:10px;color:var(--peach);font-weight:800;margin-bottom:4px}.prompt{white-space:pre-wrap;font-size:12px;color:#bac2de;margin-top:5px;line-height:1.4}.source{margin-top:8px;color:var(--muted);font-size:10px}.source summary{cursor:pointer;color:var(--green)}.source div{white-space:pre-wrap;margin-top:5px;line-height:1.35}.card-actions{display:flex;gap:7px;margin-top:9px}.card-actions .btn,.download{font-size:11px;padding:8px 10px;text-decoration:none;text-align:center}.tripo-btn{background:var(--purple);color:var(--bg);flex:1}.download{display:block;background:var(--green);color:var(--bg);border-radius:9px;font-weight:850;flex:1}.artifact{border-color:#5b4770}.empty{color:#585b70;text-align:center;padding:80px 20px}.toast{position:fixed;right:20px;bottom:20px;background:var(--green);color:var(--bg);padding:11px 15px;border-radius:10px;font-weight:800;display:none;z-index:10;max-width:460px}@keyframes in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}@media(max-width:1050px){.layout{grid-template-columns:1fr}.gallery{columns:2 180px}}
</style></head><body><div class="shell">
<div class="top"><div><div class="title">Kaggle Inference Hub</div><div class="sub">Local Control Plane · model-routed Kaggle GPU workers</div></div><div class="status"><span class="pill" id="conn"><span class="dot"></span>Connecting</span><span class="pill">Queue <b id="q">0</b></span><span class="pill">Inflight <b id="inflight">0</b></span></div></div>
<div class="layout"><aside class="left">
<section class="panel"><div class="panel-head"><div><h2>目标模型</h2><div class="sub" id="modelDesc">选择对应 Kaggle Worker</div></div></div><select id="model"></select><div class="label">ACCESS TOKEN</div><input id="token" type="password" placeholder="KAGGLE_HUB_TOKEN"></section>
<section class="panel ai-panel"><div class="panel-head"><div><h2>AI Prompt Pipeline</h2><div class="sub" id="aiProvider">提交 GPU 前先优化，结果会回填供你确认</div></div><span class="ai-status" id="aiState">检查中</span></div><div class="label">处理模式</div><select id="aiMode"></select><label class="check"><input id="aiTranslate" type="checkbox" checked>输出英文 Prompt</label><div class="ai-hint">自动读取当前目标模型，并使用对应 Prompt 适配规则；AI 处理不会自动提交 GPU。</div></section>
<section class="panel"><div class="panel-head"><div><h2>单个 Prompt</h2><div class="sub">任意换行均作为一个任务</div></div></div><textarea id="singlePrompt" placeholder="A cinematic portrait...\n\nmultiline prompt is supported"></textarea><div class="actions"><button class="btn ai-btn" id="singleAi" onclick="optimizeSingle()">AI 优化</button><button class="btn primary" id="singleSubmit" onclick="submitSingle()">提交 1 个任务</button><button class="btn ghost restore" id="singleRestore" onclick="restoreSingle()">恢复原文</button><button class="btn ghost" onclick="clearSingle()">清空</button></div></section>
<section class="panel"><div class="panel-head"><div><h2>批量 Prompt</h2><div class="sub">仅这里按一行一个任务拆分</div></div></div><textarea id="batchPrompts" placeholder="A mountain lake at sunrise\nA futuristic Tokyo street\nA forest covered in mist"></textarea><div class="actions"><button class="btn ai-btn" id="batchAi" onclick="optimizeBatch()">AI 批量优化</button><button class="btn batch-primary" id="batchSubmit" onclick="submitBatch()">批量加入队列</button><button class="btn ghost restore" id="batchRestore" onclick="restoreBatch()">恢复原文</button><button class="btn ghost" onclick="clearBatch()">清空</button></div></section>
<section class="panel"><div class="panel-head"><div><h2>生成参数</h2><div class="sub">模型切换时自动更新默认 Steps</div></div></div><div class="row"><div><div class="label">WIDTH</div><input id="width" type="number" value="1024"></div><div><div class="label">HEIGHT</div><input id="height" type="number" value="1024"></div></div><div class="row"><div><div class="label">STEPS</div><input id="steps" type="number" value="2"></div><div><div class="label">BASE SEED · 可空</div><input id="seed" type="number" placeholder="随机"></div></div><div class="stats"><div class="stat"><b id="batchCount">0</b><span>BATCH</span></div><div class="stat"><b id="queueCount">0</b><span>MODEL QUEUE</span></div><div class="stat"><b id="imageCount">0</b><span>IMAGES</span></div><div class="stat"><b id="workerCount">0</b><span>WORKERS</span></div></div></section>
<section class="panel"><div class="panel-head"><div><h2>TripoSR · 图片转 3D</h2><div class="sub">上传本地图片，或在右侧生成图上点击“转 3D”</div></div></div><div class="label">PNG / JPEG / WEBP · 最大 20 MB</div><input id="tripoFile" type="file" accept="image/png,image/jpeg,image/webp"><div class="row"><div><div class="label">OUTPUT</div><select id="tripoFormat"><option value="glb">GLB（推荐）</option><option value="obj">OBJ</option></select></div><div><div class="label">MC RESOLUTION</div><select id="tripoResolution"><option value="128">128 · 快速</option><option value="256" selected>256 · 标准</option><option value="384">384 · 精细</option><option value="512">512 · 高显存</option></select></div></div><label class="check"><input id="tripoRemoveBg" type="checkbox" checked>自动移除背景并缩放主体</label><div class="actions"><button class="btn tripo-btn" id="tripoSubmit" onclick="submitTripoUpload()">上传并加入 TripoSR 队列</button></div><div class="ai-hint">推荐主体完整、单物体、无遮挡图片。当前队列：<b id="tripoQueueCount">0</b> · 已完成：<b id="artifactCount">0</b></div></section>
<section class="panel"><div class="panel-head"><div><h2>Workers</h2><div class="sub">45 秒内心跳视为在线</div></div></div><div id="workers" class="workers"><div class="sub">暂无 Worker</div></div></section>
</aside><section><div class="gallery-head"><div><b>Live Results</b><div class="sub">SANA / Z-Image 图片与 TripoSR 3D 文件</div></div><button class="btn ghost" onclick="loadHistory()">刷新</button></div><main id="grid" class="gallery"><div class="empty" id="empty">等待第一个结果...</div></main></section></div></div><div class="toast" id="toast"></div>
<script>
const $=id=>document.getElementById(id);
const grid=$("grid"), token=$("token"), singlePrompt=$("singlePrompt"), batchPrompts=$("batchPrompts"), model=$("model");
let MODELS={};
let PROMPT_AI={enabled:false,configured:false,modes:[]};
let singleAiState=null;
let batchAiState=null;
let lastEventId=0;
const renderedEvents=new Set();

token.value=localStorage.getItem("kaggle_hub_token")||"";
token.addEventListener("change",()=>localStorage.setItem("kaggle_hub_token",token.value));

function toast(text,bad=false){
  const el=$("toast");
  el.textContent=text;
  el.style.background=bad?"#f38ba8":"#a6e3a1";
  el.style.display="block";
  setTimeout(()=>el.style.display="none",3000);
}

function authHeaders(json=true){
  const h={"Authorization":"Bearer "+token.value};
  if(json)h["Content-Type"]="application/json";
  return h;
}

async function apiJson(url, options={}){
  const r=await fetch(url,options);
  let data=null;
  try{data=await r.json()}catch{data={detail:await r.text().catch(()=>"")}}
  if(!r.ok)throw new Error(data?.detail||data?.error||`HTTP ${r.status}`);
  return data;
}

async function initModels(){
  const xs=await apiJson('/api/models');
  for(const x of xs){
    MODELS[x.id]=x;
    if(x.input_kind!=='prompt')continue;
    const o=document.createElement('option');
    o.value=x.id;o.textContent=x.label;model.appendChild(o);
  }
  const saved=localStorage.getItem('kaggle_hub_model');
  if(saved&&MODELS[saved]?.input_kind==='prompt')model.value=saved;
  applyModel(true);
}

function applyModel(first=false){
  const x=MODELS[model.value];
  if(!x)return;
  $("modelDesc").textContent=x.description;
  const key='steps_'+x.id;
  if(first||!localStorage.getItem(key))$("steps").value=localStorage.getItem(key)||x.default_steps;
  else $("steps").value=localStorage.getItem(key);
  localStorage.setItem('kaggle_hub_model',x.id);
  status();
}

model.addEventListener('change',()=>{
  applyModel(false);
  let stale=false;
  if(singleAiState){singleAiState.stale=true;stale=true}
  if(batchAiState){batchAiState.stale=true;stale=true}
  if(stale)toast('目标模型已切换，建议重新执行 AI 优化');
});

$("steps").addEventListener('change',()=>{
  if(model.value)localStorage.setItem('steps_'+model.value,$("steps").value)
});

async function initPromptPipeline(){
  try{
    PROMPT_AI=await apiJson('/api/prompt/pipeline');
    const sel=$("aiMode");
    sel.innerHTML='';
    for(const x of PROMPT_AI.modes||[]){
      const o=document.createElement('option');o.value=x.id;o.textContent=x.label;sel.appendChild(o);
    }
    const savedMode=localStorage.getItem('prompt_ai_mode');
    if(savedMode&&(PROMPT_AI.modes||[]).some(x=>x.id===savedMode))sel.value=savedMode;
    const savedTranslate=localStorage.getItem('prompt_ai_translate');
    if(savedTranslate!==null)$("aiTranslate").checked=savedTranslate==='1';
    sel.addEventListener('change',()=>localStorage.setItem('prompt_ai_mode',sel.value));
    $("aiTranslate").addEventListener('change',()=>localStorage.setItem('prompt_ai_translate',$("aiTranslate").checked?'1':'0'));

    const stateEl=$("aiState");
    if(PROMPT_AI.configured){
      stateEl.textContent='READY';stateEl.className='ai-status ready';
      $("aiProvider").textContent=`${PROMPT_AI.provider_model} · 并发 ${PROMPT_AI.concurrency}`;
    }else{
      stateEl.textContent=PROMPT_AI.enabled?'未配置':'DISABLED';stateEl.className='ai-status disabled';
      $("aiProvider").textContent=PROMPT_AI.enabled?'请配置 PROMPT_AI_BASE_URL / MODEL':'设置 PROMPT_AI_ENABLED=true 后启用';
    }
    $("singleAi").disabled=!PROMPT_AI.configured;
    $("batchAi").disabled=!PROMPT_AI.configured;
  }catch(e){
    $("aiState").textContent='ERROR';$("aiState").className='ai-status disabled';
    $("singleAi").disabled=true;$("batchAi").disabled=true;
    $("aiProvider").textContent='Pipeline 状态读取失败';
  }
}

function commonParams(){
  const body={
    model:model.value,
    width:Number($("width").value||1024),
    height:Number($("height").value||1024),
    steps:Number($("steps").value||MODELS[model.value]?.default_steps||2)
  };
  const seed=$("seed").value.trim();
  if(seed!=="")body.seed=Number(seed);
  return body;
}

function pipelineParams(){
  return {
    model:model.value,
    mode:$("aiMode").value||'enhance',
    translate_to_english:$("aiTranslate").checked,
  };
}

function promptMeta(x, extra={}){
  return {
    mode:x.mode,
    provider_model:x.provider_model,
    elapsed_ms:x.elapsed_ms,
    translate_to_english:x.translate_to_english,
    target_model:x.target_model,
    ...extra,
  };
}

function showRestore(id,show){$(id).style.display=show?'inline-block':'none'}

async function optimizeSingle(){
  const current=singlePrompt.value.trim();
  if(!current)return toast('请输入 Prompt',true);
  if(!PROMPT_AI.configured)return toast('AI Prompt Pipeline 未配置',true);
  const btn=$("singleAi"), old=btn.textContent;
  btn.disabled=true;btn.textContent='处理中...';
  try{
    const x=await apiJson('/api/prompt/process',{
      method:'POST',headers:authHeaders(),
      body:JSON.stringify({...pipelineParams(),prompt:current})
    });
    const rootSource=singleAiState?.source||current;
    singlePrompt.value=x.processed;
    singleAiState={source:rootSource,processed:x.processed,meta:promptMeta(x),edited:false,stale:false};
    showRestore('singleRestore',true);
    toast(`AI 优化完成 · ${x.elapsed_ms} ms`);
  }catch(e){toast('AI 优化失败：'+e.message,true)}
  finally{btn.disabled=false;btn.textContent=old}
}

async function optimizeBatch(){
  const current=batchLines();
  if(!current.length)return toast('请输入批量 Prompt',true);
  if(!PROMPT_AI.configured)return toast('AI Prompt Pipeline 未配置',true);
  const btn=$("batchAi"), old=btn.textContent;
  btn.disabled=true;btn.textContent='批量处理中...';
  try{
    const x=await apiJson('/api/prompt/process/batch',{
      method:'POST',headers:authHeaders(),
      body:JSON.stringify({...pipelineParams(),prompts:current})
    });
    const rootSources=(batchAiState&&batchAiState.sourcePrompts.length===current.length)?batchAiState.sourcePrompts:current.slice();
    const processed=x.items.map(item=>item.processed);
    batchPrompts.value=processed.join('\n');
    batchAiState={
      sourcePrompts:rootSources,
      processedPrompts:processed,
      items:x.items,
      edited:false,
      stale:false,
    };
    showRestore('batchRestore',true);
    updateBatchCount();
    toast(`AI 批量完成 ${x.succeeded}/${x.total}${x.failed?` · ${x.failed} 条回退原文`:''}`,x.failed===x.total);
  }catch(e){toast('AI 批量优化失败：'+e.message,true)}
  finally{btn.disabled=false;btn.textContent=old}
}

function restoreSingle(){
  if(!singleAiState)return;
  singlePrompt.value=singleAiState.source;
  singleAiState=null;
  showRestore('singleRestore',false);
}

function restoreBatch(){
  if(!batchAiState)return;
  batchPrompts.value=batchAiState.sourcePrompts.join('\n');
  batchAiState=null;
  showRestore('batchRestore',false);
  updateBatchCount();
}

function clearSingle(){singlePrompt.value='';singleAiState=null;showRestore('singleRestore',false)}
function clearBatch(){batchPrompts.value='';batchAiState=null;showRestore('batchRestore',false);updateBatchCount()}

singlePrompt.addEventListener('input',()=>{
  if(singleAiState&&singlePrompt.value.trim()!==singleAiState.processed.trim())singleAiState.edited=true;
});

function batchLines(){return batchPrompts.value.split(/\r?\n/).map(x=>x.trim()).filter(Boolean)}
function updateBatchCount(){$("batchCount").textContent=batchLines().length}
batchPrompts.addEventListener('input',()=>{
  updateBatchCount();
  if(batchAiState&&batchLines().join('\n')!==batchAiState.processedPrompts.join('\n'))batchAiState.edited=true;
});

async function submitSingle(){
  const prompt=singlePrompt.value.trim();
  if(!prompt)return toast('请输入 Prompt',true);
  const btn=$("singleSubmit");btn.disabled=true;
  try{
    const body={...commonParams(),prompt};
    if(singleAiState){
      body.source_prompt=singleAiState.source;
      body.prompt_meta={...singleAiState.meta,edited_after_ai:singleAiState.edited,stale_model_adapter:singleAiState.stale};
    }
    const x=await apiJson('/task',{method:'POST',headers:authHeaders(),body:JSON.stringify(body)});
    toast(`#${x.task.id} → ${MODELS[x.task.model]?.label||x.task.model}`);
    clearSingle();status();
  }catch(e){toast('提交失败：'+e.message,true)}
  finally{btn.disabled=false}
}

async function submitBatch(){
  const prompts=batchLines();
  if(!prompts.length)return toast('请输入批量 Prompt',true);
  const btn=$("batchSubmit");btn.disabled=true;
  try{
    const body={...commonParams(),prompts};
    if(batchAiState&&batchAiState.sourcePrompts.length===prompts.length){
      body.source_prompts=batchAiState.sourcePrompts;
      body.prompt_metas=prompts.map((_,i)=>{
        const item=batchAiState.items[i];
        if(!item?.ok)return {};
        return promptMeta(item,{edited_after_ai:batchAiState.edited,stale_model_adapter:batchAiState.stale});
      });
    }
    const x=await apiJson('/task/batch',{method:'POST',headers:authHeaders(),body:JSON.stringify(body)});
    toast(`已加入 ${x.queued} 个任务`);
    clearBatch();status();
  }catch(e){toast('批量提交失败：'+e.message,true)}
  finally{btn.disabled=false}
}

function tripoForm(){
  const form=new FormData();
  form.append('output_format',$('tripoFormat').value);
  form.append('mc_resolution',$('tripoResolution').value);
  form.append('chunk_size','8192');
  form.append('foreground_ratio','0.85');
  form.append('remove_background',$('tripoRemoveBg').checked?'true':'false');
  return form;
}

async function submitTripoUpload(){
  const file=$('tripoFile').files[0];
  if(!file)return toast('请选择一张 PNG、JPEG 或 WebP 图片',true);
  const btn=$('tripoSubmit'),old=btn.textContent;btn.disabled=true;btn.textContent='上传中...';
  try{
    const form=tripoForm();form.append('file',file,file.name);
    const x=await apiJson('/task/triposr',{method:'POST',headers:authHeaders(false),body:form});
    $('tripoFile').value='';toast(`#${x.task.id} → TripoSR`);status();
  }catch(e){toast('TripoSR 提交失败：'+e.message,true)}
  finally{btn.disabled=false;btn.textContent=old}
}

async function submitTripoSource(sourceUrl,btn){
  const old=btn.textContent;btn.disabled=true;btn.textContent='提交中...';
  try{
    const form=tripoForm();form.append('source_url',sourceUrl);
    const x=await apiJson('/task/triposr',{method:'POST',headers:authHeaders(false),body:form});
    toast(`#${x.task.id} → TripoSR`);status();
  }catch(e){toast('TripoSR 提交失败：'+e.message,true)}
  finally{btn.disabled=false;btn.textContent=old}
}

function addArtifactCard(x,first){
  const card=document.createElement('article');card.className='card artifact';
  if(x.source_url){const img=document.createElement('img');img.loading='lazy';img.src=`${x.source_url}?t=${x.time}`;card.appendChild(img)}
  const info=document.createElement('div');info.className='info';
  const m=document.createElement('div');m.className='model';m.textContent='TripoSR · 3D READY';
  const meta=document.createElement('div');meta.className='meta';
  const mesh=x.vertices?` · ${x.vertices} vertices / ${x.faces||'?'} faces`:'';
  meta.textContent=`#${x.id} · GPU${x.gpu} · ${x.seconds}s · ${String(x.output_format||'glb').toUpperCase()}${mesh}`;
  const p=document.createElement('div');p.className='prompt';p.textContent=x.source_label||'input image';
  const actions=document.createElement('div');actions.className='card-actions';
  const link=document.createElement('a');link.className='download';link.href=x.download_url;link.download='';link.textContent=`下载 ${String(x.output_format||'glb').toUpperCase()}`;
  actions.appendChild(link);info.append(m,meta,p,actions);card.appendChild(info);
  first?grid.prepend(card):grid.append(card);
}

function addCard(x,first=true){
  const eventId=Number(x.event_id||0);
  if(eventId&&renderedEvents.has(eventId))return;
  if(eventId){renderedEvents.add(eventId);lastEventId=Math.max(lastEventId,eventId)}
  $("empty")?.remove();
  if(x.kind==='artifact'){addArtifactCard(x,first);return}
  const card=document.createElement('article');card.className='card';
  const img=document.createElement('img');img.loading='lazy';img.src=`${x.url}?t=${x.time}`;
  const info=document.createElement('div');info.className='info';
  const m=document.createElement('div');m.className='model';m.textContent=MODELS[x.model]?.label||x.model;
  const meta=document.createElement('div');meta.className='meta';
  const ai=x.prompt_meta?.mode?` · AI ${x.prompt_meta.mode}`:'';
  meta.textContent=`#${x.id} · GPU${x.gpu} · ${x.seconds}s · seed ${x.seed}${x.steps?' · '+x.steps+' steps':''}${ai}`;
  const p=document.createElement('div');p.className='prompt';p.textContent=x.prompt;
  info.append(m,meta,p);
  if(x.source_prompt){
    const d=document.createElement('details');d.className='source';
    const s=document.createElement('summary');s.textContent='查看 AI 前原始 Prompt';
    const v=document.createElement('div');v.textContent=x.source_prompt;
    d.append(s,v);info.appendChild(d);
  }
  const actions=document.createElement('div');actions.className='card-actions';
  const tripo=document.createElement('button');tripo.className='btn tripo-btn';tripo.textContent='转为 3D';
  tripo.addEventListener('click',()=>submitTripoSource(x.url,tripo));actions.appendChild(tripo);info.appendChild(actions);
  card.append(img,info);first?grid.prepend(card):grid.append(card);
}

async function loadHistory(){
  const xs=await apiJson('/api/history');
  grid.innerHTML='';renderedEvents.clear();lastEventId=0;
  if(!xs.length){grid.innerHTML='<div class="empty" id="empty">等待第一个结果...</div>';return}
  xs.forEach(x=>addCard(x,false));
}

async function syncHistory(){
  try{
    const xs=await apiJson(`/api/history?after=${lastEventId}&limit=100`);
    xs.forEach(x=>addCard(x,true));
  }catch{}
}

async function status(){
  try{
    const x=await apiJson('/api/status');
    $("q").textContent=x.queued;$("inflight").textContent=x.inflight;
    $("queueCount").textContent=x.queued_by_model?.[model.value]||0;
    $("imageCount").textContent=x.images;
    $("tripoQueueCount").textContent=x.queued_by_model?.triposr||0;
    $("artifactCount").textContent=x.artifacts||0;
    const online=(x.workers||[]).filter(w=>w.online);$("workerCount").textContent=online.length;
    const box=$("workers");box.innerHTML='';
    if(!x.workers?.length){box.innerHTML='<div class="sub">暂无 Worker</div>'}
    else{
      x.workers.sort((a,b)=>Number(b.online)-Number(a.online)).forEach(w=>{
        const d=document.createElement('div');d.className='worker';
        d.innerHTML=`<div class="worker-top"><span class="worker-name">${w.worker_id}</span><span class="${w.online?'ok':'off'}">${w.online?'● ONLINE':'● OFFLINE'}</span></div><div class="worker-meta">${MODELS[w.model]?.label||w.model} · ${w.gpus?.join(' + ')||'GPU'} · local q ${w.local_queue||0} · upload q ${w.upload_queue||0}</div>`;
        box.appendChild(d);
      });
    }
  }catch{}
}

setInterval(status,1500);
setInterval(syncHistory,2000);
updateBatchCount();
Promise.all([initModels(),initPromptPipeline()]).then(()=>{loadHistory();status()});
const proto=location.protocol==='https:'?'wss':'ws';
function connect(){
  const s=new WebSocket(`${proto}://${location.host}/ws`);
  s.onopen=()=>{$("conn").classList.add('online');$("conn").innerHTML='<span class="dot"></span>Online'};
  s.onmessage=e=>{addCard(JSON.parse(e.data));status()};
  s.onclose=()=>{$("conn").classList.remove('online');$("conn").innerHTML='<span class="dot"></span>Offline';setTimeout(connect,1500)};
}
connect();

document.addEventListener('keydown',e=>{
  if(!(e.ctrlKey||e.metaKey)||e.key!=='Enter')return;
  e.preventDefault();
  if(document.activeElement===batchPrompts)submitBatch();else submitSingle();
});
</script></body></html>'''
