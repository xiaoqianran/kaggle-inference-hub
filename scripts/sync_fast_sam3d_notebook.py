"""Embed the canonical Fast-SAM3D worker into notebook 007."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "notebooks" / "fast_sam3d_worker.py"
NOTEBOOK = ROOT / "notebooks" / "007-fast-sam3d.ipynb"


def main() -> None:
    worker_source = WORKER.read_text(encoding="utf-8").rstrip() + "\n"
    if "'''" in worker_source:
        raise RuntimeError("fast_sam3d_worker.py contains triple single quotes")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    matches = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "WORKER_SOURCE = " in source:
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one embedded worker cell, found {len(matches)}")
    source = (
        'from pathlib import Path\n\n'
        'ROOT = Path("/kaggle/working/Fast-SAM3D")\n'
        'WORKER = ROOT / "kaggle_worker.py"\n\n'
        "WORKER_SOURCE = r'''" + worker_source + "'''\n"
        'WORKER.write_text(WORKER_SOURCE, encoding="utf-8")\n'
        'print(f"✅ wrote {WORKER} ({len(WORKER_SOURCE.splitlines())} lines)")\n'
    )
    cell = notebook["cells"][matches[0]]
    cell["source"] = source.splitlines(keepends=True)
    cell["execution_count"] = None
    cell["outputs"] = []
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Synced {WORKER.name} -> {NOTEBOOK.name} ({len(worker_source.splitlines())} lines)")


if __name__ == "__main__":
    main()
