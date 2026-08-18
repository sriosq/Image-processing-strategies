"""SEPIA dipole-inversion definitions and execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .utils.deepseb import call_deepseb


class DipoleError(RuntimeError):
    pass


GENERAL = {"isBET": "0", "isInvert": "0", "isRefineBrainMask": "0"}
METHODS = {
    "def_tkd": {"reference_tissue": "Brain mask", "method": "TKD", "threshold": .15},
    "opt_tkd": {"reference_tissue": "Brain mask", "method": "TKD", "threshold": .319},
    "def_iLSQR": {"reference_tissue": "Brain mask", "method": "iLSQR", "tol": .001, "maxiter": 100, "lambda": .13, "optimise": 0},
    "opt_iLSQR": {"reference_tissue": "Brain mask", "method": "iLSQR", "tol": .001, "maxiter": 60, "lambda": .17629, "optimise": 0},
    "auto_iLSQR": {"reference_tissue": "Brain mask", "method": "iLSQR", "tol": .000001, "maxiter": 57, "lambda": .13, "optimise": 1},
    "def_closedForm": {"reference_tissue": "Brain mask", "method": "Closed-form solution", "lambda": .13, "optimise": 0},
    "opt_closedForm": {"reference_tissue": "Brain mask", "method": "Closed-form solution", "lambda": .17629, "optimise": 0},
    "auto_closedForm": {"reference_tissue": "Brain mask", "method": "Closed-form solution", "lambda": .13, "optimise": 1},
    "def_fansi": {"reference_tissue": "Brain mask", "method": "FANSI", "tol": .1, "maxiter": 150, "lambda": .0002, "mu1": .02, "mu2": 1, "solver": "Non-linear", "constraint": "TV", "gradient_mode": "L1 norm", "isGPU": 0, "isWeakHarmonic": 0},
    "opt_fansi": {"reference_tissue": "Brain mask", "method": "FANSI", "tol": .05, "maxiter": 300, "lambda": .01, "mu1": 1, "mu2": 1.3815, "solver": "Non-linear", "constraint": "TV", "gradient_mode": "L1 norm", "isGPU": 1, "isWeakHarmonic": 1, "beta": 5000, "muh": 100},
    "def_medi": {"reference_tissue": "Brain mask", "method": "MEDI", "wData": 1, "lambda": 1000, "percentage": 90, "zeropad": [0, 0, 0], "isSMV": 1, "radius": 1, "merit": 0, "isLambdaCSF": 0, "lambdaCSF": 100},
    "opt_medi": {"reference_tissue": "Brain mask", "method": "MEDI", "wData": 1, "lambda": 670, "percentage": 20, "zeropad": [0, 0, 0], "isSMV": 0, "radius": 1, "merit": 0, "isLambdaCSF": 0, "lambdaCSF": 100},
}
PIPELINES = {
    "comp_di": list(METHODS),
    "default": [name for name in METHODS if name.startswith("def_")],
    "optimized": [name for name in METHODS if name.startswith("opt_")],
    "automatic": [name for name in METHODS if name.startswith("auto_")],
}


def _matlab_value(value, matlab):
    if isinstance(value, list):
        return matlab.double(value)
    if isinstance(value, (int, float)):
        return matlab.double(float(value))
    return value


def _rename_generated(source: Path, destination: Path) -> Path:
    if source.is_file() and source.resolve() != destination.resolve():
        if destination.exists():
            destination.unlink()
        source.replace(destination)
    return destination


def run_dipole_inversion(localfield: Path, magnitude: Path, weights: Path, header: Path, mask: Path, sepia_directory: Path, output_directory: Path, participant_id: str, procedure: str, cmin: float, cmax: float, cbar: str, runner, force: bool = False, logger: Callable[[str], None] | None = None) -> dict[str, dict[str, Path]]:
    if procedure not in PIPELINES:
        raise DipoleError(f"Unknown DI procedure: {procedure}")
    for label, path in (("local field", localfield), ("magnitude", magnitude), ("weights", weights), ("SEPIA header", header), ("mask", mask), ("SEPIA directory", sepia_directory)):
        if not path.exists():
            raise DipoleError(f"{label} does not exist: {path}")
    try:
        import matlab  # type: ignore[import-not-found]
        import matlab.engine  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise DipoleError(f"MATLAB Engine is unavailable: {exc}") from exc
    engine = matlab.engine.start_matlab()
    results: dict[str, dict[str, Path]] = {}
    try:
        # python_wrapper calls sepia_addpath; only SEPIA itself belongs here.
        engine.addpath(engine.genpath(str(sepia_directory)), nargout=0)
        for name in PIPELINES[procedure]:
            folder = output_directory / procedure / name
            prefix = folder / "Sepia"
            generated_chimap = folder / "Sepia_Chimap.nii.gz"
            chimap = folder / f"{participant_id}_desc-chimap.nii.gz"
            png = folder / f"{participant_id}_desc-chimap.png"
            params_path = folder / f"{participant_id}_desc-chimap.json"
            folder.mkdir(parents=True, exist_ok=True)
            prefix.mkdir(parents=True, exist_ok=True)
            params = {"general": dict(GENERAL), "qsm": dict(METHODS[name])}
            if not force:
                _rename_generated(generated_chimap, chimap)
            if force:
                for path in (chimap, png, params_path, generated_chimap):
                    if path.is_file():
                        path.unlink()
            if force or not chimap.is_file() or chimap.stat().st_size == 0:
                converted = {"general": dict(GENERAL), "qsm": {key: _matlab_value(value, matlab) for key, value in METHODS[name].items()}}
                call = f"python_wrapper({localfield}, {magnitude}, {weights}, {header}, {METHODS[name]['method']}, {prefix}, {mask}, params)"
                if logger:
                    logger(call)
                engine.python_wrapper(str(localfield), str(magnitude), str(weights), str(header), METHODS[name]["method"], str(prefix), str(mask), converted, nargout=0)
                _rename_generated(generated_chimap, chimap)
            if not chimap.is_file() or chimap.stat().st_size == 0:
                raise DipoleError(f"SEPIA did not create the expected chi map: {chimap}")
            params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
            call_deepseb(chimap, png, cmin, cmax, cbar, maskpath=mask, runner=runner, force=force)
            results[name] = {"chimap": chimap, "qc_png": png, "parameters": params_path}
    finally:
        engine.quit()
    return results
