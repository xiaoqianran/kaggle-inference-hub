import ast
import hashlib
import importlib
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi.testclient import TestClient

from hub.config import MODEL_SPECS, TOKEN, canonical_model
from hub.crypto import decrypt_blob
from hub.prompt_pipeline.pipeline import PromptPipeline
from hub.prompt_pipeline.prompts import build_system_prompt
from hub.state import HubState


def embedded_worker(notebook: dict) -> str:
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "WORKER_SOURCE = " not in source:
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "WORKER_SOURCE"
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError("Notebook does not embed WORKER_SOURCE")


def test_sqlite_state(db_path: Path) -> None:
    # Two independent HubState objects represent two Uvicorn processes.
    producer = HubState(db_path)
    consumer = HubState(db_path)
    assert producer.instance_id == consumer.instance_id

    task_id = producer.next_id()
    producer.enqueue(
        {
            "id": task_id,
            "model": "triposr",
            "source_label": "cube.webp",
            "attempt": 0,
            "created_at": 1,
        }
    )
    assert consumer.snapshot()["queued_by_model"]["triposr"] == 1
    task = consumer.claim("triposr", "tripo-test", 0.01)
    assert task and task["id"] == task_id and task["attempt"] == 1
    assert producer.lease_owner(task_id) == "tripo-test"
    assert producer.renew_lease(task_id, "tripo-test")

    producer.register_worker(
        {
            "worker_id": "tripo-test",
            "model": "triposr",
            "gpus": ["T4"],
            "runtime": "test",
            "concurrency": 1,
            "meta": {},
        }
    )
    assert consumer.heartbeat("tripo-test", {"active_task_id": task_id})
    item = producer.finish(
        task_id,
        {
            "kind": "artifact",
            "id": task_id,
            "model": "triposr",
            "download_url": "/outputs/test.glb",
            "time": 2,
        },
    )
    assert item["event_id"] == 1
    assert consumer.snapshot()["inflight"] == 0
    assert consumer.history_items(after=0)[0]["id"] == task_id
    assert consumer.history_items(after=item["event_id"]) == []

    # Atomic claim: two Hub instances can never receive the same queued task.
    second_id = producer.next_id()
    producer.enqueue(
        {
            "id": second_id,
            "model": "triposr",
            "source_label": "sphere.webp",
            "attempt": 0,
            "created_at": 3,
        }
    )
    claims = []

    def claim_once(hub: HubState, worker_id: str) -> None:
        claims.append(hub.claim("triposr", worker_id, 0.05))

    threads = [
        threading.Thread(target=claim_once, args=(producer, "gpu-0")),
        threading.Thread(target=claim_once, args=(consumer, "gpu-1")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(task is not None for task in claims) == 1
    assert next(task for task in claims if task is not None)["id"] == second_id

    # A new state object sees the same inflight task after a simulated restart.
    restarted = HubState(db_path)
    assert restarted.snapshot()["inflight_by_model"]["triposr"] == 1


def test_http_protocol(directory: Path) -> None:
    module = importlib.import_module("hub.app")
    module.state = HubState(directory / "http-state.sqlite3")
    module.OUTPUT_DIR = directory / "outputs"
    module.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    png = b"\x89PNG\r\n\x1a\n" + b"test-image"

    with TestClient(module.app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert "Kaggle Inference Hub" in index.text

        queued = client.post(
            "/task/triposr",
            headers=headers,
            data={"output_format": "glb", "mc_resolution": "256"},
            files={"file": ("cube.png", png, "image/png")},
        )
        assert queued.status_code == 202, queued.text
        task_id = queued.json()["task"]["id"]

        claimed = client.post(
            "/task/claim",
            headers=headers,
            json={"model": "triposr", "worker_id": "http-gpu-0", "wait_seconds": 0},
        )
        assert claimed.status_code == 200, claimed.text
        assert claimed.json()["id"] == task_id
        assert claimed.headers["x-hub-instance"] == module.state.instance_id
        assert "no-store" in claimed.headers["cache-control"]

        source = client.get(claimed.json()["input_url"], headers=headers)
        assert source.status_code == 200 and source.content == png

        glb = b"glTF" + b"\x02\x00\x00\x00" + b"\x0c\x00\x00\x00"
        nonce = os.urandom(12)
        key = hashlib.sha256(TOKEN.encode()).digest()
        encrypted = nonce + AESGCM(key).encrypt(nonce, glb, None)
        uploaded = client.post(
            "/upload/artifact",
            headers=headers,
            data={
                "id": str(task_id),
                "gpu": "0",
                "model": "triposr",
                "worker_id": "http-gpu-0",
                "output_format": "glb",
            },
            files={"file": ("mesh.glb.bin", encrypted, "application/octet-stream")},
        )
        assert uploaded.status_code == 200, uploaded.text
        assert uploaded.json()["event_id"] == 1
        status = client.get("/api/status").json()
        assert status["storage"] == "sqlite" and status["inflight"] == 0


        fast_queued = client.post(
            "/task/fast-sam3d",
            headers=headers,
            data={"seed": "42"},
            files={
                "file": ("chair.png", png, "image/png"),
                "mask": ("mask.png", png, "image/png"),
            },
        )
        assert fast_queued.status_code == 202, fast_queued.text
        fast_task_id = fast_queued.json()["task"]["id"]
        fast_claimed = client.post(
            "/task/claim",
            headers=headers,
            json={"model": "fast-sam3d", "worker_id": "fast-http-gpu-0", "wait_seconds": 0},
        )
        assert fast_claimed.status_code == 200, fast_claimed.text
        fast_task = fast_claimed.json()
        assert fast_task["id"] == fast_task_id
        assert fast_task["model"] == "fast-sam3d"
        assert fast_task["output_format"] == "glb"
        assert client.get(fast_task["input_url"], headers=headers).content == png
        assert client.get(fast_task["mask_url"], headers=headers).content == png

        fast_nonce = os.urandom(12)
        fast_encrypted = fast_nonce + AESGCM(key).encrypt(fast_nonce, glb, None)
        fast_uploaded = client.post(
            "/upload/artifact",
            headers=headers,
            data={
                "id": str(fast_task_id),
                "gpu": "0",
                "model": "fast-sam3d",
                "worker_id": "fast-http-gpu-0",
                "output_format": "glb",
            },
            files={"file": ("fast.glb.bin", fast_encrypted, "application/octet-stream")},
        )
        assert fast_uploaded.status_code == 200, fast_uploaded.text
        assert fast_uploaded.json()["event_id"] == 2
        assert fast_uploaded.json()["model"] == "fast-sam3d"
        status = client.get("/api/status").json()
        assert status["storage"] == "sqlite" and status["inflight"] == 0


def main():
    assert canonical_model("sana") == "sana-sprint-1.6b"
    assert canonical_model("zimage") == "z-image-turbo-gguf"
    assert canonical_model("tripo-sr") == "triposr"
    assert canonical_model("sam3d") == "fast-sam3d"
    assert MODEL_SPECS["triposr"].input_kind == "image"
    assert MODEL_SPECS["fast-sam3d"].input_kind == "image"
    assert MODEL_SPECS["fast-sam3d"].output_kind == "artifact"

    with tempfile.TemporaryDirectory(prefix="kaggle-hub-test-") as directory:
        test_root = Path(directory)
        test_sqlite_state(test_root / "state.sqlite3")
        test_http_protocol(test_root)

    password = "test-password"
    key = hashlib.sha256(password.encode()).digest()
    nonce = os.urandom(12)
    plain = b"webp-bytes"
    encrypted = nonce + AESGCM(key).encrypt(nonce, plain, None)
    assert decrypt_blob(encrypted, password) == plain

    system_prompt = build_system_prompt("sana-sprint-1.6b", "enhance", True)
    assert "SANA Sprint 1.6B" in system_prompt
    assert "Return the final prompt in English" in system_prompt
    assert PromptPipeline._clean_output('Prompt: "a red cube"') == "a red cube"

    root = Path(__file__).resolve().parents[1]
    tripo_nb = root / "notebooks" / "003-triposr-image-to-3d.ipynb"
    tripo_worker = root / "notebooks" / "triposr_worker.py"
    fast_nb = root / "notebooks" / "007-fast-sam3d.ipynb"
    fast_worker = root / "notebooks" / "fast_sam3d_worker.py"
    assert tripo_nb.is_file()
    assert tripo_worker.is_file()
    assert not (root / "notebooks" / "003-triposr-build.ipynb").exists()
    notebook = json.loads(tripo_nb.read_text(encoding="utf-8"))
    notebook_text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    worker_text = tripo_worker.read_text(encoding="utf-8")
    assert embedded_worker(notebook) == worker_text
    assert '"rembg[gpu]"' in notebook_text
    assert '"onnxruntime"' not in notebook_text
    assert '"device_id": gpu' in worker_text
    assert "POST /task/claim" in worker_text
    assert 'snapshot.get("storage") != "sqlite"' in worker_text
    assert '"active_task_id": active_task["id"]' in worker_text
    assert 'triposr-persistent-py310' in worker_text
    assert 'mp.get_context("spawn")' in worker_text
    assert 'print("Token: wangran")' not in notebook_text

    assert fast_nb.is_file() and fast_worker.is_file()
    fast_notebook = json.loads(fast_nb.read_text(encoding="utf-8"))
    fast_notebook_text = "\n".join("".join(cell.get("source", [])) for cell in fast_notebook["cells"])
    fast_worker_text = fast_worker.read_text(encoding="utf-8")
    assert embedded_worker(fast_notebook) == fast_worker_text
    assert '"model": MODEL' in fast_worker_text
    assert 'MODEL = "fast-sam3d"' in fast_worker_text
    assert 'mp.get_context("spawn")' in fast_worker_text
    assert '"active_task_id": active_task["id"]' in fast_worker_text
    assert 'output = inference(image, mask, seed=seed)' in fast_worker_text
    assert 'Fast-SAM3D · Kaggle 双 T4 常驻 Worker' in fast_notebook_text
    assert 'update-alternatives' not in fast_notebook_text
    assert 'KAGGLE_HUB_TOKEN' in fast_notebook_text

    print(
        "OK: SQLite cross-process queue + atomic claims + durable leases/history + "
        "AES-GCM + synchronized persistent dual-GPU TripoSR/Fast-SAM3D workers"
    )


if __name__ == "__main__":
    main()
