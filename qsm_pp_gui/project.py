"""Load, validate, and track resumable QSM project milestones."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from scipy.io import loadmat


MILESTONES = (
    ("inputs_header", "Inputs & header"),
    ("megre_masks", "meGRE masks"),
    ("t1_sc_mask", "T1 SC mask"),
    ("disc_labels", "Disc labels"),
    ("vertebral_levels", "T1-space vertebral levels"),
    ("mask_qc", "T1-space mask QC"),
    ("registration", "meGRE-to-T1w registration"),
    ("registration_qc", "meGRE-space registration QC"),
    ("field_map", "Field map"),
    ("fieldmap_visualization", "Field-map PNGs"),
    ("fieldmap_qc", "Field-map QC"),
    ("noise_weights", "Noise & weights"),
    ("bgfr", "BGFR"),
    ("dipole_inversion", "Dipole inversion"),
)


class ProjectError(RuntimeError):
    """Raised when a saved project cannot be loaded or validated."""


def _nonempty_file(value: str | None) -> bool:
    if not value:
        return False
    path = Path(value)
    return path.is_file() and path.stat().st_size > 0


def _fieldmap_variant_complete(project: dict[str, Any], variant: str) -> bool:
    value = project.get("fieldmap", {}).get(variant, {})
    if not isinstance(value, dict):
        return False
    return all(_nonempty_file(value.get(name)) for name in ("b0", "corrected_phase"))


def _fieldmap_pngs_complete(project: dict[str, Any]) -> bool:
    fieldmap = project.get("fieldmap", {})
    return all(
        isinstance(fieldmap.get(variant), dict) and _nonempty_file(fieldmap[variant].get("qc_png"))
        for variant in ("masked", "unmasked")
    )


def mark_milestone(project: dict[str, Any], name: str, complete: bool = True) -> None:
    if name not in {key for key, _ in MILESTONES}:
        raise ProjectError(f"Unknown project milestone: {name}")
    milestones = project.setdefault("milestones", {})
    entry = milestones.setdefault(name, {})
    entry["complete"] = complete
    if complete:
        entry["completed_at"] = datetime.now(timezone.utc).isoformat()
    else:
        entry.pop("completed_at", None)


def milestone_complete(project: dict[str, Any], name: str) -> bool:
    return bool(project.get("milestones", {}).get(name, {}).get("complete", False))


def validate_sepia_headers(project: dict[str, Any]) -> tuple[Path, Path]:
    json_path = Path(project.get("sepia_header", ""))
    mat_path = Path(project.get("sepia_header_mat", ""))
    if not _nonempty_file(str(json_path)):
        raise ProjectError(f"SEPIA JSON header is missing or empty: {json_path}")
    try:
        header = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectError(f"SEPIA JSON header is invalid: {exc}") from exc
    required_json = {"TE", "B0", "CF", "B0_dir", "matrix_size", "voxel_size"}
    missing_json = required_json - set(header)
    if missing_json:
        raise ProjectError(f"SEPIA JSON header is missing: {', '.join(sorted(missing_json))}")
    if not _nonempty_file(str(mat_path)):
        raise ProjectError(f"SEPIA MATLAB header is missing or empty: {mat_path}")
    try:
        matlab_header = loadmat(mat_path, variable_names=["TE", "B0", "CF", "B0_dir", "matrixSize", "voxelSize"])
    except (OSError, ValueError, NotImplementedError) as exc:
        raise ProjectError(f"SEPIA MATLAB header cannot be read: {exc}") from exc
    required_mat = {"TE", "B0", "CF", "B0_dir", "matrixSize", "voxelSize"}
    missing_mat = required_mat - set(matlab_header)
    if missing_mat:
        raise ProjectError(f"SEPIA MATLAB header is missing: {', '.join(sorted(missing_mat))}")
    return json_path, mat_path


def refresh_milestones(project: dict[str, Any]) -> None:
    try:
        validate_sepia_headers(project)
    except ProjectError:
        mark_milestone(project, "inputs_header", False)
    else:
        if not milestone_complete(project, "inputs_header"):
            mark_milestone(project, "inputs_header")

    masking = project.get("masking", {})
    output_checks = {
        "megre_masks": all(_nonempty_file(masking.get(key)) for key in ("sc_mask", "gm_mask", "wm_mask")),
        "t1_sc_mask": _nonempty_file(masking.get("t1_sc_mask")),
        "disc_labels": _nonempty_file(masking.get("disc_labels")),
        "vertebral_levels": _nonempty_file(masking.get("vertebral_levels")),
        "registration": all(_nonempty_file(masking.get(key)) for key in ("warp_megre_to_t1w", "warp_t1w_to_megre", "vertebral_levels_megre")),
        "field_map": all(_fieldmap_variant_complete(project, variant) for variant in ("masked", "unmasked")),
        "fieldmap_visualization": _fieldmap_pngs_complete(project),
    }
    for name, complete in output_checks.items():
        was_complete = milestone_complete(project, name)
        if complete and not was_complete:
            mark_milestone(project, name)
        elif not complete and was_complete:
            mark_milestone(project, name, False)


def load_project(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        project = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectError(f"Could not read project JSON: {exc}") from exc
    for key in ("participant_id", "magnitude_path", "phase_path", "output_directory", "sepia_header"):
        if not project.get(key):
            raise ProjectError(f"Project is missing required field: {key}")
    header_path = Path(project["sepia_header"])
    try:
        header = json.loads(header_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectError(f"Could not read the project's SEPIA JSON header: {exc}") from exc
    refresh_milestones(project)
    project["project_file"] = str(path.resolve())
    path.write_text(json.dumps(project, indent=2), encoding="utf-8")
    return project, header


def next_milestone(project: dict[str, Any]) -> tuple[str, str] | None:
    for name, label in MILESTONES:
        if not milestone_complete(project, name):
            return name, label
    return None


def milestone_summary(project: dict[str, Any]) -> str:
    complete = [label for name, label in MILESTONES if milestone_complete(project, name)]
    upcoming = next_milestone(project)
    completed_text = ", ".join(complete) if complete else "None yet"
    next_text = upcoming[1] if upcoming else "Pipeline complete"
    return f"Completed milestones: {completed_text}\nNext step: {next_text}"


def set_project_milestone(project_path: Path, name: str, complete: bool = True) -> dict[str, Any]:
    try:
        project = json.loads(project_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectError(f"Could not update project milestone: {exc}") from exc
    mark_milestone(project, name, complete)
    project_path.write_text(json.dumps(project, indent=2), encoding="utf-8")
    return project
