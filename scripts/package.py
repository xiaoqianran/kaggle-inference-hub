"""Create a small source package without local environments and runtime data."""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / f"{ROOT.name}.zip"

# These are development environments, caches, build output, or local runtime data.
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "node_modules",
    "wheels",
    "outputs",
    "sana_received",
}


def is_excluded(path: Path, output: Path) -> bool:
    relative = path.relative_to(ROOT)
    parts = relative.parts

    if path.resolve() == output.resolve():
        return True
    if any(part in EXCLUDED_DIRS for part in parts[:-1] if path.is_file()):
        return True
    if any(part in EXCLUDED_DIRS for part in parts):
        return True

    name = path.name
    if name.endswith((".pyc", ".pyo")):
        return True
    if name in {".env"} or (name.startswith(".env.") and name != ".env.example"):
        return True
    if name in {".DS_Store", "Thumbs.db"}:
        return True
    return False


def create_package(output: Path) -> tuple[int, int]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    files = 0
    bytes_added = 0
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for current, directories, filenames in os.walk(ROOT):
            current_path = Path(current)
            directories[:] = sorted(
                directory
                for directory in directories
                if not is_excluded(current_path / directory, output)
            )
            for filename in sorted(filenames):
                path = current_path / filename
                if is_excluded(path, output):
                    continue
                archive.write(path, path.relative_to(ROOT).as_posix())
                files += 1
                bytes_added += path.stat().st_size

    return files, bytes_added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the Kaggle Inference Hub without local-only files."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"ZIP path (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    files, bytes_added = create_package(output)
    print(f"Created: {output}")
    print(f"Files: {files}; source bytes: {bytes_added:,}; ZIP bytes: {output.stat().st_size:,}")


if __name__ == "__main__":
    main()
