"""Embed the canonical TripoSR worker into notebook 003.

The worker is kept as a normal Python file for linting and tests. Run this script
after editing it so Kaggle receives exactly the same source.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "notebooks" / "triposr_worker.py"
NOTEBOOK = ROOT / "notebooks" / "003-triposr-image-to-3d.ipynb"


def main() -> None:
    worker_source = WORKER.read_text(encoding="utf-8").rstrip() + "\n"
    if "'''" in worker_source:
        raise RuntimeError("triposr_worker.py contains triple single quotes and cannot be embedded raw")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    matches = []
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "WORKER_SOURCE = " in source:
            matches.append(cell)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one embedded worker cell, found {len(matches)}")

    embedded = (
        "from pathlib import Path\n\n"
        "ROOT = Path(\"/kaggle/working/TripoSR\")\n"
        "WORKER = ROOT / \"kaggle_worker.py\"\n\n"
        "WORKER_SOURCE = r'''"
        + worker_source
        + "'''\n"
        "WORKER.write_text(WORKER_SOURCE)\n"
        "print(\"✅ Worker written:\", WORKER)\n"
        "print(\"Lines:\", len(WORKER_SOURCE.splitlines()))\n"
    )
    matches[0]["source"] = embedded.splitlines(keepends=True)
    NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Synced {WORKER.name} -> {NOTEBOOK.name} ({len(worker_source.splitlines())} lines)")


if __name__ == "__main__":
    main()
