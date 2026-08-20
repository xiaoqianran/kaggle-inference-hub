import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from hub.config import MODEL_SPECS, canonical_model
from hub.crypto import decrypt_blob
from hub.prompt_pipeline.pipeline import PromptPipeline
from hub.prompt_pipeline.prompts import build_system_prompt
from hub.state import HubState


def main():
    assert canonical_model("sana") == "sana-sprint-1.6b"
    assert canonical_model("zimage") == "z-image-turbo-gguf"
    assert canonical_model("tripo-sr") == "triposr"
    assert MODEL_SPECS["triposr"].input_kind == "image"

    state = HubState()
    state.enqueue({"id": 1, "model": "sana-sprint-1.6b", "prompt": "a", "attempt": 0})
    state.enqueue({"id": 2, "model": "z-image-turbo-gguf", "prompt": "b", "attempt": 0})
    state.enqueue({"id": 3, "model": "triposr", "source_label": "cube.webp", "attempt": 0})
    assert state.claim("triposr", "tripo-test", 0.01)["id"] == 3
    assert state.claim("z-image-turbo-gguf", "z-test", 0.01)["id"] == 2
    assert state.claim("sana-sprint-1.6b", "s-test", 0.01)["id"] == 1

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
    assert tripo_nb.is_file()
    assert tripo_worker.is_file()
    assert not (root / "notebooks" / "003-triposr-build.ipynb").exists()
    notebook = json.loads(tripo_nb.read_text(encoding="utf-8"))
    notebook_text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    worker_text = tripo_worker.read_text(encoding="utf-8")
    assert '"rembg[gpu]"' in notebook_text
    assert '"onnxruntime"' not in notebook_text
    assert '"device_id": gpu' in worker_text
    assert 'triposr-persistent-py310' in worker_text
    assert 'mp.get_context("spawn")' in worker_text
    assert 'UserSecretsClient' in notebook_text
    assert 'os.getenv("KAGGLE_HUB_TOKEN"' in notebook_text
    assert 'print("Token: wangran")' not in notebook_text

    print("OK: prompt/image-to-3D routing + queue isolation + AES-GCM + persistent dual-GPU TripoSR worker")


if __name__ == "__main__":
    main()
