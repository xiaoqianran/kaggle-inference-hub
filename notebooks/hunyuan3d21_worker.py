from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


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


def run_shape() -> float:
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
            image=str(CFG.image),
            num_inference_steps=CFG.shape_steps,
            octree_resolution=CFG.octree_resolution,
            guidance_scale=5.0,
        )[0]
    seconds = time.perf_counter() - started
    mesh.export(CFG.shape_obj)
    mesh.export(CFG.shape_glb)
    print(
        f"[HY21_SHAPE] seconds={seconds:.3f} glb={CFG.shape_glb.stat().st_size}",
        flush=True,
    )

    del mesh, pipeline
    clear_cuda()
    memory_snapshot("shape-released")
    return seconds


def run_paint() -> float:
    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    config = Hunyuan3DPaintConfig(
        max_num_view=CFG.paint_views, resolution=CFG.paint_resolution
    )
    config.render_size = CFG.render_size
    config.texture_size = CFG.texture_size
    realesrgan = CFG.root / "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    if not realesrgan.is_file():
        raise FileNotFoundError(realesrgan)
    config.realesrgan_ckpt_path = str(realesrgan)

    started = time.perf_counter()
    pipeline = Hunyuan3DPaintPipeline(config)
    print(f"[HY21] paint_load={time.perf_counter() - started:.2f}s", flush=True)
    memory_snapshot("paint-loaded")

    started = time.perf_counter()
    pipeline(
        mesh_path=str(CFG.shape_obj),
        image_path=str(CFG.image),
        output_mesh_path=str(CFG.pbr_obj),
        use_remesh=True,
        save_glb=False,
    )
    seconds = time.perf_counter() - started
    if not CFG.pbr_obj.is_file():
        raise FileNotFoundError(CFG.pbr_obj)
    print(
        f"[HY21_PAINT] seconds={seconds:.3f} obj={CFG.pbr_obj.stat().st_size}",
        flush=True,
    )
    memory_snapshot("paint-finished")
    return seconds


def export_pbr_glb() -> None:
    from convert_utils import create_glb_with_pbr_materials

    base = CFG.pbr_obj.with_suffix("")
    assets = {
        "obj": CFG.pbr_obj,
        "mtl": base.with_suffix(".mtl"),
        "albedo": Path(f"{base}.jpg"),
        "metallic": Path(f"{base}_metallic.jpg"),
        "roughness": Path(f"{base}_roughness.jpg"),
    }
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing PBR assets: {missing}")

    textures = {name: str(assets[name]) for name in ("albedo", "metallic", "roughness")}
    create_glb_with_pbr_materials(str(CFG.pbr_obj), textures, str(CFG.pbr_glb))
    if not CFG.pbr_glb.is_file() or CFG.pbr_glb.stat().st_size < 1000:
        raise RuntimeError(
            "PBR GLB export failed or produced an unexpectedly small file"
        )
    print(f"[HY21_EXPORT] glb={CFG.pbr_glb.stat().st_size}", flush=True)


def main() -> None:
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


if __name__ == "__main__":
    main()
