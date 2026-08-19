from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    default_steps: int
    description: str


MODEL_SPECS = {
    "sana-sprint-1.6b": ModelSpec(
        id="sana-sprint-1.6b",
        label="SANA Sprint 1.6B",
        default_steps=2,
        description="Diffusers · dual T4 worker",
    ),
    "z-image-turbo-gguf": ModelSpec(
        id="z-image-turbo-gguf",
        label="Z-Image-Turbo GGUF",
        default_steps=8,
        description="stable-diffusion.cpp · Q4_K · T4 sm_75",
    ),
}

MODEL_ALIASES = {
    "sana": "sana-sprint-1.6b",
    "sana-sprint": "sana-sprint-1.6b",
    "z-image": "z-image-turbo-gguf",
    "zimage": "z-image-turbo-gguf",
    "z-image-turbo": "z-image-turbo-gguf",
}

DEFAULT_MODEL = "sana-sprint-1.6b"
TOKEN = os.getenv("KAGGLE_HUB_TOKEN", os.getenv("PASSWORD", "wangran"))
PORT = int(os.getenv("KAGGLE_HUB_PORT", "30100"))
OUTPUT_DIR = Path(os.getenv("KAGGLE_HUB_OUTPUT_DIR", "outputs"))
LEASE_SECONDS = int(os.getenv("KAGGLE_HUB_LEASE_SECONDS", "1200"))
MAX_ATTEMPTS = int(os.getenv("KAGGLE_HUB_MAX_ATTEMPTS", "3"))
QUEUE_SIZE = int(os.getenv("KAGGLE_HUB_QUEUE_SIZE", "1000"))
WORKER_TTL_SECONDS = int(os.getenv("KAGGLE_HUB_WORKER_TTL_SECONDS", "45"))
HISTORY_LIMIT = int(os.getenv("KAGGLE_HUB_HISTORY_LIMIT", "500"))


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


PROMPT_AI_ENABLED = env_bool("PROMPT_AI_ENABLED", False)
PROMPT_AI_BASE_URL = os.getenv("PROMPT_AI_BASE_URL", "").strip()
PROMPT_AI_API_KEY = os.getenv("PROMPT_AI_API_KEY", "").strip()
PROMPT_AI_MODEL = os.getenv("PROMPT_AI_MODEL", "").strip()
PROMPT_AI_TIMEOUT_SECONDS = float(os.getenv("PROMPT_AI_TIMEOUT_SECONDS", "60"))
PROMPT_AI_CONCURRENCY = max(1, int(os.getenv("PROMPT_AI_CONCURRENCY", "4")))
PROMPT_AI_MAX_TOKENS = max(64, int(os.getenv("PROMPT_AI_MAX_TOKENS", "900")))
PROMPT_AI_TEMPERATURE = float(os.getenv("PROMPT_AI_TEMPERATURE", "0.35"))


def canonical_model(value: str | None) -> str:
    value = (value or DEFAULT_MODEL).strip().lower()
    value = MODEL_ALIASES.get(value, value)
    if value not in MODEL_SPECS:
        raise KeyError(value)
    return value
