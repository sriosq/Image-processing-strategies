"""SEPIA background-field-removal definitions and execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .utils.deepseb import call_deepseb


class BgfrError(RuntimeError):
    pass


GENERAL = {"isBET": "0", "isInvert": "0", "isRefineBrainMask": "0"}
METHODS = {
    "def_pdf": {"method": "PDF", "tol": .1, "iteration": 50, "padSize": 40, "refine_method": "None", "refine_order": 4, "erode_radius": 0, "erode_before_radius": 0},
    "opt_pdf": {"method": "PDF", "tol": .1, "iteration": 250, "padSize": 14, "refine_method": "None", "refine_order": 4, "erode_radius": 0, "erode_before_radius": 0},
    "def_lbv": {"method": "LBV", "tol": .0001, "depth": 5, "peel": 2, "refine_order": 4, "erode_radius": 0, "erode_before_radius": 0},
    "opt_lbv": {"method": "LBV", "tol": .0001, "depth": 2, "peel": 1, "refine_order": 4, "erode_radius": 0, "erode_before_radius": 0},
    "def_sharp": {"method": "SHARP", "radius": 4, "threshold": .03, "refine_order": 4, "erode_radius": 0, "erode_before_radius": 0},
    "opt_sharp": {"method": "SHARP", "radius": 1, "threshold": .075, "refine_order": 4, "erode_radius": 0, "erode_before_radius": 0},
    "def_resharp": {"method": "RESHARP", "radius": 4, "alpha": .01, "refine_order": 4, "erode_radius": 0, "erode_before_radius": 0},
    "opt_resharp": {"method": "RESHARP", "radius": 1, "alpha": .0022, "refine_order": 4, "erode_radius": 0, "erode_before_radius": 0},
    "def_vsharp": {"method": "VSHARP", "radius": [8, 7, 6, 5, 4, 3], "refine_order": 4, "erode_radius": 0, "erode_before_radius": 0},
    "opt_vsharp": {"method": "VSHARP", "radius": [2, 1], "refine_order": 4, "erode_radius": 0, "erode_before_radius": 0},
}
PIPELINES = {
    "comp_bgfr": list(METHODS),
    "default": [name for name in METHODS if name.startswith("def_")],
    "optimized": [name for name in METHODS if name.startswith("opt_")],
}


def _matlab_value(value, matlab):
    if isinstance(value, list):
        return matlab.double(value)
    if isinstance(value, (int, float)):
        return matlab.double(float(value))
    return value


def run_bgfr(fieldmap: Path, header: Path, mask: Path, noise_sd: Path, sepia_directory: Path, output_directory: Path, procedure: str, cmin: float, cmax: float, cbar: str, runner, force: bool = False, logger: Callable[[str], None] | None = None) -> dict[str, dict[str, Path]]:
    if procedure not in PIPELINES:
        raise BgfrError(f"Unknown BGFR procedure: {procedure}")
    for label, path in (("fieldmap", fieldmap), ("SEPIA header", header), ("mask", mask), ("noise SD", noise_sd), ("SEPIA directory", sepia_directory)):
        if not path.exists():
            raise BgfrError(f"{label} does not exist: {path}")
    try:
        import matlab  # type: ignore[import-not-found]
        import matlab.engine  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise BgfrError(f"MATLAB Engine is unavailable: {exc}") from exc
    engine = matlab.engine.start_matlab()
    results: dict[str, dict[str, Path]] = {}
    try:
        engine.addpath(engine.genpath(str(sepia_directory)), nargout=0)
        for name in PIPELINES[procedure]:
            folder = output_directory / procedure / name
            prefix = folder / "Sepia"
            localfield = folder / "Sepia_localfield.nii.gz"
            png = folder / f"{name}.png"
            params_path = folder / f"{name}_params.json"
            folder.mkdir(parents=True, exist_ok=True)
            # Match the output-prefix layout used by the checked automation.
            prefix.mkdir(parents=True, exist_ok=True)
            params = {"general": dict(GENERAL), "bfr": dict(METHODS[name])}
            if force or not localfield.is_file() or localfield.stat().st_size == 0:
                converted = {"general": dict(GENERAL), "bfr": {key: _matlab_value(value, matlab) for key, value in METHODS[name].items()}}
                call = f"python_wrapper({fieldmap}, '', {noise_sd}, {header}, {METHODS[name]['method']}, {prefix}, {mask}, params)"
                if logger:
                    logger(call)
                engine.python_wrapper(str(fieldmap), "", str(noise_sd), str(header), METHODS[name]["method"], str(prefix), str(mask), converted, nargout=0)
            if not localfield.is_file() or localfield.stat().st_size == 0:
                raise BgfrError(f"SEPIA did not create the expected local field: {localfield}")
            params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
            call_deepseb(localfield, png, cmin, cmax, cbar, maskpath=mask, runner=runner, force=force)
            results[name] = {"localfield": localfield, "qc_png": png, "parameters": params_path}
    finally:
        engine.quit()
    return results
