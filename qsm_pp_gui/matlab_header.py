"""Bridge between the Python GUI and the MATLAB SEPIA-header utility."""

from __future__ import annotations

import json
from pathlib import Path

from .project import mark_milestone


class MatlabHeaderError(RuntimeError):
    """Raised when MATLAB cannot create or verify the SEPIA MAT header."""


def create_matlab_header(json_header_path: Path, project_path: Path) -> Path:
    """Run create_sepia_header.m and add the resulting path to the project."""
    json_header_path = json_header_path.resolve()
    project_path = project_path.resolve()
    output_path = json_header_path.with_suffix(".mat")
    matlab_utils = Path(__file__).resolve().parent / "utils"

    try:
        import matlab.engine  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise MatlabHeaderError(
            "MATLAB Engine for Python is unavailable. Install/configure the "
            "engine for this Python environment before creating the MAT header."
        ) from exc

    engine = None
    try:
        engine = matlab.engine.start_matlab()
        engine.addpath(str(matlab_utils), nargout=0)
        returned_path = engine.create_sepia_header(
            str(json_header_path), str(output_path), nargout=1
        )
    except Exception as exc:
        raise MatlabHeaderError(f"MATLAB could not create the SEPIA header: {exc}") from exc
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass

    matlab_path = Path(str(returned_path)) if returned_path else output_path
    if not matlab_path.is_file() or matlab_path.stat().st_size == 0:
        raise MatlabHeaderError(
            f"MATLAB returned successfully, but no non-empty MAT file was found at: {matlab_path}"
        )

    try:
        project = json.loads(project_path.read_text(encoding="utf-8"))
        project["sepia_header_mat"] = str(matlab_path.resolve())
        mark_milestone(project, "inputs_header")
        project_path.write_text(json.dumps(project, indent=2), encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        raise MatlabHeaderError(
            f"The MAT header was created, but the project file could not be updated: {exc}"
        ) from exc

    return matlab_path.resolve()
