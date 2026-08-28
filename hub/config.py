from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypeAlias
from pathlib import Path


ArtifactOptionValue: TypeAlias = str | int | float | bool


@dataclass(frozen=True)
class ArtifactOptionSpec:
    id: str
    label: str
    kind: str
    default: ArtifactOptionValue
    choices: tuple[ArtifactOptionValue, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    help: str = ""
    visible: bool = True


@dataclass(frozen=True)
class ArtifactInputSpec:
    id: str
    label: str
    help: str = ""
    required: bool = True


@dataclass(frozen=True)
class ArtifactSpec:
    options: tuple[ArtifactOptionSpec, ...] = ()
    auxiliary_inputs: tuple[ArtifactInputSpec, ...] = ()


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    default_steps: int
    description: str
    input_kind: str = "prompt"
    output_kind: str = "image"
    artifact: ArtifactSpec | None = None


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
    "triposr": ModelSpec(
        id="triposr",
        label="TripoSR",
        default_steps=0,
        description="single image to GLB/OBJ · dual T4 worker",
        input_kind="image",
        output_kind="artifact",
        artifact=ArtifactSpec(
            options=(
                ArtifactOptionSpec("output_format", "输出格式", "select", "glb", ("glb", "obj")),
                ArtifactOptionSpec("mc_resolution", "MC Resolution", "select", 256, (128, 256, 384, 512)),
                ArtifactOptionSpec(
                    "remove_background",
                    "自动移除背景",
                    "boolean",
                    True,
                    help="缩放并居中单个主体",
                ),
                ArtifactOptionSpec("chunk_size", "Chunk Size", "integer", 8192, minimum=1024, maximum=131072, visible=False),
                ArtifactOptionSpec(
                    "foreground_ratio",
                    "Foreground Ratio",
                    "number",
                    0.85,
                    minimum=0.5,
                    maximum=1.0,
                    visible=False,
                ),
            )
        ),
    ),
    "fast-sam3d": ModelSpec(
        id="fast-sam3d",
        label="Fast-SAM3D",
        default_steps=0,
        description="masked image to GLB · persistent dual T4 worker",
        input_kind="image",
        output_kind="artifact",
        artifact=ArtifactSpec(
            options=(
                ArtifactOptionSpec("seed", "Seed", "integer", 42, minimum=0, maximum=2_147_483_647),
                ArtifactOptionSpec("output_format", "输出格式", "select", "glb", ("glb",), visible=False),
            ),
            auxiliary_inputs=(
                ArtifactInputSpec("mask", "Mask", "与 RGB 同尺寸的非空 mask；建议 PNG 黑白图"),
            ),
        ),
    ),
    "hunyuan3d-2.1": ModelSpec(
        id="hunyuan3d-2.1",
        label="Hunyuan3D 2.1",
        default_steps=20,
        description="image to PBR GLB · shape + paint · dual T4 worker",
        input_kind="image",
        output_kind="artifact",
        artifact=ArtifactSpec(
            options=(
                ArtifactOptionSpec("shape_steps", "Shape Steps", "integer", 20, minimum=1, maximum=50),
                ArtifactOptionSpec("octree_resolution", "Octree", "select", 256, (128, 256, 384, 512)),
                ArtifactOptionSpec("paint_views", "Paint Views", "select", 4, (2, 4, 6, 8)),
                ArtifactOptionSpec("texture_size", "Texture", "select", 2048, (1024, 2048, 4096)),
                ArtifactOptionSpec(
                    "paint_resolution",
                    "Paint Resolution",
                    "select",
                    256,
                    (128, 256, 384, 512),
                    visible=False,
                ),
                ArtifactOptionSpec("output_format", "输出格式", "select", "glb", ("glb",), visible=False),
            )
        ),
    ),
}

MODEL_ALIASES = {
    "sana": "sana-sprint-1.6b",
    "sana-sprint": "sana-sprint-1.6b",
    "z-image": "z-image-turbo-gguf",
    "zimage": "z-image-turbo-gguf",
    "z-image-turbo": "z-image-turbo-gguf",
    "tripo": "triposr",
    "tripo-sr": "triposr",
    "fastsam3d": "fast-sam3d",
    "fast-sam-3d": "fast-sam3d",
    "sam3d": "fast-sam3d",
    "hunyuan2.1": "hunyuan3d-2.1",
    "hunyuan21": "hunyuan3d-2.1",
    "hunyuan3d21": "hunyuan3d-2.1",
    "hunyuan-2.1": "hunyuan3d-2.1",
}

DEFAULT_MODEL = "sana-sprint-1.6b"
TOKEN = os.getenv("KAGGLE_HUB_TOKEN", os.getenv("PASSWORD", "wangran"))
PORT = int(os.getenv("KAGGLE_HUB_PORT", "30100"))
OUTPUT_DIR = Path(os.getenv("KAGGLE_HUB_OUTPUT_DIR", "outputs"))
STATE_DB = Path(os.getenv("KAGGLE_HUB_STATE_DB", str(OUTPUT_DIR / "hub-state.sqlite3")))
LEASE_SECONDS = int(os.getenv("KAGGLE_HUB_LEASE_SECONDS", "1200"))
MAX_ATTEMPTS = int(os.getenv("KAGGLE_HUB_MAX_ATTEMPTS", "3"))
QUEUE_SIZE = int(os.getenv("KAGGLE_HUB_QUEUE_SIZE", "1000"))
WORKER_TTL_SECONDS = int(os.getenv("KAGGLE_HUB_WORKER_TTL_SECONDS", "45"))
HISTORY_LIMIT = int(os.getenv("KAGGLE_HUB_HISTORY_LIMIT", "500"))
INPUT_MAX_BYTES = int(os.getenv("KAGGLE_HUB_INPUT_MAX_MB", "20")) * 1024 * 1024
ARTIFACT_MAX_BYTES = int(os.getenv("KAGGLE_HUB_ARTIFACT_MAX_MB", "100")) * 1024 * 1024


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
