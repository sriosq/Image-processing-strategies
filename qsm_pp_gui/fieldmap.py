"""ROMEO field-map runs and participant project integration."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import shutil
from typing import Callable

import nibabel as nib

from .config import ToolConfig
from .masking import MaskingError, run_command
from .project import mark_milestone
from .romeo import build_romeo_command
from .utils.deepseb import call_deepseb


Runner = Callable[[list[str], bool], None]


@dataclass(frozen=True, slots=True)
class FieldmapInputs:
    participant_id: str
    magnitude_path: Path
    phase_path: Path
    mask_path: Path
    echo_times_ms: list[float]
    output_root: Path
    phase_offset_correction: str

    @property
    def participant_directory(self) -> Path:
        return self.output_root / self.participant_id

    @property
    def fieldmap_directory(self) -> Path:
        return self.participant_directory / "fieldmap"

    def variant_directory(self, masked: bool) -> Path:
        return self.fieldmap_directory / ("masked_fieldmap" if masked else "unmasked_fieldmap")

    def output_paths(self, masked: bool) -> dict[str, Path]:
        folder = self.variant_directory(masked)
        return {
            "b0": folder / f"{self.participant_id}_desc-b0_fieldmap.nii.gz",
            "corrected_phase": folder / f"{self.participant_id}_desc-corrected_phase.nii.gz",
        }

    def validate(self) -> None:
        for label, path in (("4D magnitude", self.magnitude_path), ("4D phase", self.phase_path), ("meGRE mask", self.mask_path)):
            if not path.is_file():
                raise MaskingError(f"{label} file does not exist: {path}")
        if not self.echo_times_ms or any(value <= 0 for value in self.echo_times_ms):
            raise MaskingError("Echo times in milliseconds must be positive.")
        if self.phase_offset_correction not in {"on", "off", "bipolar"}:
            raise MaskingError("Phase-offset correction must be on, off, or bipolar.")
        try:
            magnitude_shape = nib.load(str(self.magnitude_path)).shape
            phase_shape = nib.load(str(self.phase_path)).shape
            mask_shape = nib.load(str(self.mask_path)).shape
        except (OSError, nib.filebasedimages.ImageFileError) as exc:
            raise MaskingError(f"A ROMEO input is not a readable NIfTI: {exc}") from exc
        if len(magnitude_shape) != 4 or len(phase_shape) != 4:
            raise MaskingError(f"ROMEO magnitude and phase must be 4D; received {magnitude_shape} and {phase_shape}.")
        if magnitude_shape != phase_shape:
            raise MaskingError(f"Magnitude and phase shapes do not match: {magnitude_shape} vs {phase_shape}.")
        if magnitude_shape[3] != len(self.echo_times_ms):
            raise MaskingError(f"The 4D data contain {magnitude_shape[3]} echoes but {len(self.echo_times_ms)} echo times were supplied.")
        if len(mask_shape) != 3 or mask_shape != magnitude_shape[:3]:
            raise MaskingError(f"The meGRE mask shape {mask_shape} must match the 4D spatial shape {magnitude_shape[:3]}.")


def _gzip_copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise MaskingError(f"Expected ROMEO output is missing or empty: {source}")
    with source.open("rb") as source_file, gzip.open(destination, "wb") as destination_file:
        shutil.copyfileobj(source_file, destination_file)


def run_fieldmap(inputs: FieldmapInputs, config: ToolConfig, masked: bool, runner: Runner = run_command, force: bool = False) -> dict[str, Path]:
    inputs.validate()
    output_directory = inputs.variant_directory(masked)
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = inputs.output_paths(masked)
    raw_outputs = {
        "b0": output_directory / "B0.nii",
        "corrected_phase": output_directory / "corrected_phase.nii",
    }
    if force:
        for path in (*outputs.values(), *raw_outputs.values()):
            if path.is_file():
                path.unlink()
    if not all(path.is_file() and path.stat().st_size > 0 for path in outputs.values()):
        command = build_romeo_command(
            config,
            str(inputs.phase_path),
            str(inputs.magnitude_path),
            inputs.echo_times_ms,
            str(output_directory),
            mask_path=str(inputs.mask_path),
            phase_offset_correction=inputs.phase_offset_correction,
            unwrap=masked,
        )
        runner(command, False)
        for name, raw_path in raw_outputs.items():
            _gzip_copy(raw_path, outputs[name])
    for path in outputs.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise MaskingError(f"Processed ROMEO output is missing or empty: {path}")
    return outputs


def update_project_fieldmap(inputs: FieldmapInputs, masked_output: dict[str, Path] | None = None, unmasked_output: dict[str, Path] | None = None) -> None:
    project_path = inputs.participant_directory / f"{inputs.participant_id}_qsm_project.json"
    if not project_path.is_file():
        raise MaskingError(f"Participant project file not found: {project_path}")
    project = json.loads(project_path.read_text(encoding="utf-8"))
    fieldmap = project.setdefault("fieldmap", {})
    if masked_output:
        fieldmap["masked"] = {name: str(path.resolve()) for name, path in masked_output.items()}
    if unmasked_output:
        fieldmap["unmasked"] = {name: str(path.resolve()) for name, path in unmasked_output.items()}
    fieldmap["phase_offset_correction"] = inputs.phase_offset_correction
    if all(
        all(Path(fieldmap.get(variant, {}).get(name, "")).is_file() for name in ("b0", "corrected_phase"))
        for variant in ("masked", "unmasked")
    ):
        mark_milestone(project, "field_map")
    mark_milestone(project, "fieldmap_visualization", False)
    mark_milestone(project, "fieldmap_qc", False)
    project_path.write_text(json.dumps(project, indent=2), encoding="utf-8")


def create_fieldmap_pngs(
    inputs: FieldmapInputs,
    cmin: float,
    cmax: float,
    cbar: str,
    runner: Runner = run_command,
    force: bool = False,
) -> dict[str, Path]:
    project_path = inputs.participant_directory / f"{inputs.participant_id}_qsm_project.json"
    if not project_path.is_file():
        raise MaskingError(f"Participant project file not found: {project_path}")
    project = json.loads(project_path.read_text(encoding="utf-8"))
    fieldmap = project.get("fieldmap", {})
    requested_settings = {"cmin": float(cmin), "cmax": float(cmax), "cbar": cbar.strip()}
    previous_settings = project.get("fieldmap_qc_settings", {})
    settings_changed = (
        previous_settings.get("cmin") != requested_settings["cmin"]
        or previous_settings.get("cmax") != requested_settings["cmax"]
        or previous_settings.get("cbar") != requested_settings["cbar"]
    )
    regenerate = force or settings_changed
    pngs: dict[str, Path] = {}
    for variant in ("masked", "unmasked"):
        variant_data = fieldmap.get(variant, {})
        if not isinstance(variant_data, dict) or not variant_data.get("b0"):
            raise MaskingError(f"The {variant} B0 fieldmap must be created before its PNG.")
        b0_path = Path(variant_data["b0"])
        name = b0_path.name[:-7] + ".png" if b0_path.name.endswith(".nii.gz") else b0_path.with_suffix(".png").name
        png_path = b0_path.parent / name
        pngs[variant] = call_deepseb(
            b0_path, png_path, cmin, cmax, cbar,
            maskpath=inputs.mask_path, runner=runner, force=regenerate,
        )
        variant_data["qc_png"] = str(png_path.resolve())
    project["fieldmap_qc_settings"] = requested_settings
    mark_milestone(project, "fieldmap_visualization")
    mark_milestone(project, "fieldmap_qc", False)
    project_path.write_text(json.dumps(project, indent=2), encoding="utf-8")
    return pngs
