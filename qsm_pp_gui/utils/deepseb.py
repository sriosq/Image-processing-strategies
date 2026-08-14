"""Reusable Spinal Cord Toolbox quantitative-map visualization helper."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..masking import MaskingError, run_command


Runner = Callable[[list[str], bool], None]


def call_deepseb(
    inpath: Path,
    outpath: Path,
    cmin: float,
    cmax: float,
    cbar: str,
    maskpath: Path | None = None,
    runner: Runner = run_command,
    force: bool = False,
) -> Path:
    """Create an axial-slice PNG from a quantitative NIfTI using sct_deepseb."""
    if not inpath.is_file() or inpath.stat().st_size == 0:
        raise MaskingError(f"DeepSeg QC input is missing or empty: {inpath}")
    if cmin >= cmax:
        raise MaskingError("Color minimum must be smaller than color maximum.")
    if not cbar.strip():
        raise MaskingError("A Matplotlib colorbar name is required.")
    if maskpath is not None and not maskpath.is_file():
        raise MaskingError(f"QC outline mask does not exist: {maskpath}")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if force and outpath.is_file():
        outpath.unlink()
    if not outpath.is_file() or outpath.stat().st_size == 0:
        command = [
            "sct_deepseb", "-i", str(inpath), "-o", str(outpath),
            "-cmin", str(cmin), "-cmax", str(cmax), "-cbar", cbar.strip(),
        ]
        if maskpath is not None:
            command += ["-s", str(maskpath)]
        runner(command, False)
    if not outpath.is_file() or outpath.stat().st_size == 0:
        raise MaskingError(f"sct_deepseb did not create a non-empty PNG: {outpath}")
    return outpath
