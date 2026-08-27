from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    sha256: str


@dataclass(frozen=True)
class SetupConfig:
    root: Path = Path("/kaggle/working/Hunyuan3D-2.1")
    venv: Path = Path("/kaggle/working/hy21-venv")
    wheel_dir: Path = Path("/kaggle/working/hy21-wheels")
    commit: str = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
    repo: str = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"


CFG = SetupConfig()
PY = CFG.venv / "bin/python"
PIP = [str(PY), "-m", "pip"]

ARTIFACTS = (
    Artifact(
        name="custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl",
        url=(
            "https://github.com/xiaoqianran/kaggle-build/releases/download/"
            "hunyuan3d21-rasterizer-t4-v1/custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl"
        ),
        sha256="c3dea51fdbc970c2d0c8c8718f0ef53b423383c81142532d632634aae753da80",
    ),
    Artifact(
        name="mesh_inpaint_processor-0.0.1-cp310-cp310-linux_x86_64.whl",
        url=(
            "https://github.com/xiaoqianran/kaggle-build/releases/download/"
            "hunyuan3d21-cp310-v1/mesh_inpaint_processor-0.0.1-cp310-cp310-linux_x86_64.whl"
        ),
        sha256="3572c7983dced6b21a7b018ab7e0bc885e0a26ede113402ac1c1ea5ecc515610",
    ),
)

CORE_DEPS = [
    "ninja==1.11.1.1",
    "pybind11==2.13.4",
    "transformers==4.46.0",
    "diffusers==0.30.0",
    "accelerate==1.1.1",
    "pytorch-lightning==1.9.5",
    "huggingface-hub==0.30.2",
    "safetensors==0.4.4",
    "numpy==1.24.4",
    "scipy==1.14.1",
    "einops==0.8.0",
    "pandas==2.2.2",
    "opencv-python==4.10.0.84",
    "imageio==2.36.0",
    "scikit-image==0.24.0",
    "rembg==2.0.65",
    "realesrgan==0.3.0",
    "basicsr==1.4.2",
    "trimesh==4.4.7",
    "pymeshlab==2022.2.post3",
    "pygltflib==1.16.3",
    "xatlas==0.0.9",
    "open3d==0.18.0",
    "omegaconf==2.3.0",
    "pyyaml==6.0.2",
    "configargparse==1.7",
    "tqdm==4.66.5",
    "psutil==6.0.0",
    "cupy-cuda12x==13.4.1",
    "onnxruntime==1.16.3",
    "torchmetrics==1.6.0",
    "pydantic==2.10.6",
    "timm",
    "pythreejs",
    "torchdiffeq",
]


def run(args: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(artifact: Artifact) -> Path:
    CFG.wheel_dir.mkdir(parents=True, exist_ok=True)
    path = CFG.wheel_dir / artifact.name
    if not path.exists() or sha256(path) != artifact.sha256:
        path.unlink(missing_ok=True)
        urllib.request.urlretrieve(artifact.url, path)
    actual = sha256(path)
    if actual != artifact.sha256:
        raise RuntimeError(f"SHA256 mismatch for {artifact.name}: {actual}")
    return path


def patch_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Patch anchor not found: {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def ensure_venv() -> None:
    if PY.exists():
        return
    shutil.rmtree(CFG.venv, ignore_errors=True)
    run(["uv", "venv", "--python", "3.10", "--seed", str(CFG.venv)])


def install_dependencies() -> None:
    run(PIP + ["install", "-q", "--upgrade", "pip", "wheel"])
    run(PIP + ["install", "-q", "setuptools==80.9.0"])
    run(
        PIP
        + [
            "install",
            "-q",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "torchaudio==2.5.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ]
    )
    run(PIP + ["install", "-q"] + CORE_DEPS)
    run(PIP + ["install", "-q", "git+https://github.com/gromgull/py-vox-io.git"])
    wheels = [str(download_verified(artifact)) for artifact in ARTIFACTS]
    run(PIP + ["install", "-q"] + wheels)


def verify_runtime_imports() -> None:
    run(
        [
            str(PY),
            "-c",
            (
                "import pkg_resources, pytorch_lightning, torch, "
                "custom_rasterizer, mesh_inpaint_processor; "
                "print('runtime imports OK', torch.__version__, torch.version.cuda)"
            ),
        ]
    )


def ensure_source() -> None:
    if not CFG.root.exists():
        run(["git", "clone", "--filter=blob:none", CFG.repo, str(CFG.root)])
    run(["git", "fetch", "-q", "origin", CFG.commit], cwd=CFG.root)
    run(["git", "checkout", "-q", CFG.commit], cwd=CFG.root)


def patch_sources() -> None:
    site = subprocess.check_output(
        [str(PY), "-c", "import site; print(site.getsitepackages()[0])"], text=True
    ).strip()
    patch_once(
        Path(site) / "basicsr/data/degradations.py",
        "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
        "from torchvision.transforms.functional import rgb_to_grayscale",
    )

    multiview = CFG.root / "hy3dpaint/utils/multiview_utils.py"
    patch_once(
        multiview,
        "        self.pipeline = pipeline.to(self.device)\n",
        "        self.pipeline = pipeline\n"
        "        if torch.cuda.is_available():\n"
        "            self.pipeline.enable_model_cpu_offload(gpu_id=0)\n",
    )
    patch_once(
        multiview,
        "            self.dino_v2 = self.dino_v2.to(self.device)\n",
        '            dino_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"\n'
        "            self.dino_v2 = self.dino_v2.to(dino_device)\n"
        "            self.dino_device = dino_device\n"
        '            print(f"[hy21] DINO Giant resident on {dino_device}", flush=True)\n',
    )
    patch_once(
        multiview,
        "        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))\n",
        '        exec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")\n'
        "        kwargs = dict(generator=torch.Generator(device=exec_device).manual_seed(0))\n",
    )
    patch_once(
        multiview,
        '            kwargs["dino_hidden_states"] = dino_hidden_states\n',
        '            kwargs["dino_hidden_states"] = dino_hidden_states.to("cuda:0", non_blocking=True)\n',
    )
    paint_pipeline = CFG.root / "hy3dpaint/hunyuanpaintpbr/pipeline.py"
    patch_once(
        paint_pipeline,
        "        dtype = next(self.vae.parameters()).dtype\n"
        "        images = (images - 0.5) * 2.0\n"
        "        posterior = self.vae.encode(images.to(dtype)).latent_dist\n",
        "        dtype = next(self.vae.parameters()).dtype\n"
        "        device = self._execution_device\n"
        "        images = (images - 0.5) * 2.0\n"
        "        images = images.to(device=device, dtype=dtype, non_blocking=True)\n"
        "        posterior = self.vae.encode(images).latent_dist\n",
    )

    paint_model = CFG.root / "hy3dpaint/hunyuanpaintpbr/unet/model.py"
    patch_once(
        paint_model,
        "        dtype = next(self.pipeline.vae.parameters()).dtype\n\n"
        "        images = (images - 0.5) * 2.0\n"
        "        posterior = self.pipeline.vae.encode(images.to(dtype)).latent_dist\n",
        "        dtype = next(self.pipeline.vae.parameters()).dtype\n"
        "        device = self.pipeline._execution_device\n\n"
        "        images = (images - 0.5) * 2.0\n"
        "        images = images.to(device=device, dtype=dtype, non_blocking=True)\n"
        "        posterior = self.pipeline.vae.encode(images).latent_dist\n",
    )

    patch_once(
        CFG.root / "hy3dpaint/DifferentiableRenderer/mesh_utils.py",
        "import bpy\n",
        "try:\n    import bpy\nexcept ImportError:\n    bpy = None\n",
    )


def install_mesh_extension() -> None:
    module_path = subprocess.check_output(
        [
            str(PY),
            "-c",
            "import mesh_inpaint_processor; print(mesh_inpaint_processor.__file__)",
        ],
        text=True,
    ).strip()
    target = CFG.root / "hy3dpaint/DifferentiableRenderer" / Path(module_path).name
    shutil.copy2(module_path, target)


def ensure_realesrgan() -> None:
    path = CFG.root / "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "wget",
            "-q",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "-O",
            str(path),
        ]
    )


def main() -> None:
    ensure_venv()
    install_dependencies()
    verify_runtime_imports()
    ensure_source()
    patch_sources()
    install_mesh_extension()
    ensure_realesrgan()
    run(
        [
            str(PY),
            "-c",
            "import torch; print(torch.__version__, torch.cuda.device_count())",
        ]
    )
    print("✅ Hunyuan3D 2.1 environment ready", flush=True)


if __name__ == "__main__":
    main()
