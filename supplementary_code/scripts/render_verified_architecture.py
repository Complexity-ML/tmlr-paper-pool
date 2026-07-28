"""Render the checkpoint-verified architecture diagram from Graphviz source."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent / "figures"
SOURCE = SCRIPT_DIR / "architecture_complexity_deep.dot"


def render() -> Path:
    dot = shutil.which("dot")
    if dot is None:
        raise RuntimeError("Graphviz is required; install it and ensure 'dot' is on PATH")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_DIR / "architecture_complexity_deep.pdf"
    png = FIGURE_DIR / "architecture_complexity_deep.png"
    subprocess.run([dot, "-Tpdf", str(SOURCE), "-o", str(pdf)], check=True)
    subprocess.run([dot, "-Tpng", "-Gdpi=200", str(SOURCE), "-o", str(png)], check=True)
    return pdf


if __name__ == "__main__":
    print(render())
