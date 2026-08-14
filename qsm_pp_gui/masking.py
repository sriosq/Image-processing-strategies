"""Participant-centred Spinal Cord Toolbox masking workflow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Callable

import nibabel as nib

from .project import mark_milestone


class MaskingError(RuntimeError):
    """Raised when a masking input or external SCT step is invalid."""


Runner = Callable[[list[str], bool], None]


def run_command(command: list[str], interactive: bool = False) -> None:
    try:
        result = subprocess.run(
            command,
            capture_output=not interactive,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MaskingError(
            f"Command not found: {command[0]}. Ensure Spinal Cord Toolbox is on PATH."
        ) from exc
    if result.returncode != 0:
        detail = "" if interactive else (result.stderr or result.stdout).strip()
        raise MaskingError(
            f"Command failed ({result.returncode}): {' '.join(command)}"
            + (f"\n{detail}" if detail else "")
        )


def _require_output(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise MaskingError(f"{label} was not created or is empty: {path}")
    return path


def _remove_for_rerun(path: Path, force: bool) -> None:
    if force and path.is_file():
        path.unlink()


def _require_3d_nifti(path: Path, label: str) -> tuple[int, int, int]:
    try:
        shape = nib.load(str(path)).shape
    except (OSError, nib.filebasedimages.ImageFileError) as exc:
        raise MaskingError(f"{label} is not a readable NIfTI: {path}") from exc
    if len(shape) != 3:
        raise MaskingError(f"{label} must be 3D for SCT; received shape {shape}: {path}")
    return shape


@dataclass(frozen=True, slots=True)
class MaskingInputs:
    participant_id: str
    magnitude_path: Path
    t1_path: Path | None
    output_root: Path

    @property
    def participant_directory(self) -> Path:
        return self.output_root / self.participant_id

    @property
    def masking_directory(self) -> Path:
        return self.participant_directory / "masking"

    @property
    def first_echo(self) -> Path:
        return self.masking_directory / f"{self.participant_id}_echo-1_magnitude.nii.gz"

    @property
    def sc_mask(self) -> Path:
        return self.masking_directory / f"{self.participant_id}_desc-SC_mask.nii.gz"

    @property
    def gm_mask(self) -> Path:
        return self.masking_directory / f"{self.participant_id}_desc-GM_mask.nii.gz"

    @property
    def wm_mask(self) -> Path:
        return self.masking_directory / f"{self.participant_id}_desc-WM_mask.nii.gz"

    @property
    def t1_sc_mask(self) -> Path:
        return self.masking_directory / f"{self.participant_id}_desc-T1w_SC_mask.nii.gz"

    @property
    def disc_labels(self) -> Path:
        return self.masking_directory / f"{self.participant_id}_desc-discs_label.nii.gz"

    @property
    def vertebral_levels(self) -> Path:
        return self.masking_directory / f"{self.participant_id}_space-T1w_desc-SC_vertlevels_dseg.nii.gz"

    @property
    def legacy_vertebral_levels(self) -> Path:
        return self.masking_directory / f"{self.participant_id}_desc-SC_vertlevels_dseg.nii.gz"

    @property
    def registration_directory(self) -> Path:
        return self.masking_directory / "register"

    @property
    def warp_megre_to_t1w(self) -> Path:
        return self.registration_directory / "warp_megre2t1w.nii.gz"

    @property
    def warp_t1w_to_megre(self) -> Path:
        return self.registration_directory / "warp_t1w2megre.nii.gz"

    @property
    def registered_megre(self) -> Path:
        return self.registration_directory / f"{self.participant_id}_space-T1w_desc-meGRE_registered.nii.gz"

    @property
    def vertebral_levels_megre(self) -> Path:
        return self.registration_directory / f"{self.participant_id}_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"

    def validate_megre(self) -> None:
        if not self.participant_id:
            raise MaskingError("Enter and save a participant ID on the Inputs & header tab first.")
        if not self.magnitude_path.is_file():
            raise MaskingError(f"Magnitude file does not exist: {self.magnitude_path}")

    def validate_t1(self) -> None:
        if self.t1_path is None or not self.t1_path.is_file():
            raise MaskingError("Select an existing T1-weighted NIfTI file first.")


def extract_first_echo(inputs: MaskingInputs, force: bool = False) -> Path:
    """Save a participant-prefixed 3D first echo, preserving spatial metadata."""
    inputs.validate_megre()
    inputs.masking_directory.mkdir(parents=True, exist_ok=True)
    try:
        image = nib.load(str(inputs.magnitude_path))
    except (OSError, nib.filebasedimages.ImageFileError) as exc:
        raise MaskingError(f"Magnitude is not a readable NIfTI: {inputs.magnitude_path}") from exc
    if len(image.shape) not in {3, 4}:
        raise MaskingError(f"Magnitude must be a 3D or 4D NIfTI; received shape {image.shape}.")
    if inputs.first_echo.is_file() and inputs.first_echo.stat().st_size > 0 and not force:
        try:
            cached_shape = _require_3d_nifti(inputs.first_echo, "Cached first-echo magnitude")
        except MaskingError:
            inputs.first_echo.unlink()
        else:
            if cached_shape == image.shape[:3]:
                return inputs.first_echo
            inputs.first_echo.unlink()
    _remove_for_rerun(inputs.first_echo, force)
    if len(image.shape) == 3:
        data = image.dataobj[:, :, :]
    elif len(image.shape) == 4 and image.shape[3] >= 1:
        data = image.dataobj[:, :, :, 0]
    header = image.header.copy()
    header.set_data_shape(data.shape)
    nib.save(nib.Nifti1Image(data, image.affine, header), str(inputs.first_echo))
    _require_output(inputs.first_echo, "First-echo magnitude")
    _require_3d_nifti(inputs.first_echo, "First-echo magnitude")
    return inputs.first_echo


def create_megre_masks(inputs: MaskingInputs, runner: Runner = run_command, force: bool = False) -> dict[str, Path]:
    first_echo = extract_first_echo(inputs, force=force)
    steps = [
        (["sct_deepseg", "spinalcord", "-i", str(first_echo), "-o", str(inputs.sc_mask)], inputs.sc_mask, "Spinal-cord mask"),
        (["sct_deepseg", "graymatter", "-i", str(first_echo), "-o", str(inputs.gm_mask)], inputs.gm_mask, "Gray-matter mask"),
        (["sct_maths", "-i", str(inputs.sc_mask), "-sub", str(inputs.gm_mask), "-o", str(inputs.wm_mask)], inputs.wm_mask, "White-matter mask"),
    ]
    for command, output, label in steps:
        _remove_for_rerun(output, force)
        if not output.is_file() or output.stat().st_size == 0:
            runner(command, False)
        _require_output(output, label)
    return {"first_echo": first_echo, "sc_mask": inputs.sc_mask, "gm_mask": inputs.gm_mask, "wm_mask": inputs.wm_mask}


def create_t1_sc_mask(inputs: MaskingInputs, runner: Runner = run_command, force: bool = False) -> Path:
    inputs.validate_t1()
    inputs.masking_directory.mkdir(parents=True, exist_ok=True)
    if force:
        _remove_for_rerun(inputs.t1_sc_mask, True)
        _remove_for_rerun(inputs.vertebral_levels, True)
    if not inputs.t1_sc_mask.is_file() or inputs.t1_sc_mask.stat().st_size == 0:
        runner(["sct_deepseg", "spinalcord", "-i", str(inputs.t1_path), "-o", str(inputs.t1_sc_mask)], False)
    return _require_output(inputs.t1_sc_mask, "T1 spinal-cord mask")


def create_disc_labels(inputs: MaskingInputs, first_label: int = 1, last_label: int = 10, runner: Runner = run_command, force: bool = False) -> Path:
    inputs.validate_t1()
    if first_label < 1 or last_label < first_label:
        raise MaskingError("Disc label range must use positive integers with the end greater than or equal to the start.")
    inputs.masking_directory.mkdir(parents=True, exist_ok=True)
    if force:
        _remove_for_rerun(inputs.disc_labels, True)
        _remove_for_rerun(inputs.vertebral_levels, True)
    if not inputs.disc_labels.is_file() or inputs.disc_labels.stat().st_size == 0:
        runner([
            "sct_label_utils", "-i", str(inputs.t1_path),
            "-create-viewer", f"{first_label}:{last_label}",
            "-o", str(inputs.disc_labels),
        ], True)
    return _require_output(inputs.disc_labels, "Manual disc-label file")


def create_vertebral_levels(inputs: MaskingInputs, runner: Runner = run_command, force: bool = False) -> Path:
    inputs.validate_t1()
    _require_output(inputs.t1_sc_mask, "T1 spinal-cord mask")
    _require_output(inputs.disc_labels, "Manual disc-label file")
    if inputs.legacy_vertebral_levels.is_file() and not inputs.vertebral_levels.exists():
        shutil.move(str(inputs.legacy_vertebral_levels), str(inputs.vertebral_levels))
    _remove_for_rerun(inputs.vertebral_levels, force)
    if inputs.vertebral_levels.is_file() and inputs.vertebral_levels.stat().st_size > 0:
        return inputs.vertebral_levels
    before = set(inputs.masking_directory.glob("*labeled.nii*"))
    runner([
        "sct_label_vertebrae", "-i", str(inputs.t1_path),
        "-s", str(inputs.t1_sc_mask), "-c", "t1",
        "-discfile", str(inputs.disc_labels),
        "-ofolder", str(inputs.masking_directory),
    ], False)
    candidates = list(set(inputs.masking_directory.glob("*labeled.nii*")) - before)
    if not candidates:
        candidates = list(inputs.masking_directory.glob("*labeled.nii*"))
    if not candidates:
        raise MaskingError("SCT completed but no labeled vertebral segmentation was found.")
    generated = max(candidates, key=lambda path: path.stat().st_mtime)
    shutil.move(str(generated), str(inputs.vertebral_levels))
    return _require_output(inputs.vertebral_levels, "Vertebral-level segmentation")


def register_megre_to_t1w(inputs: MaskingInputs, runner: Runner = run_command, force: bool = False) -> dict[str, Path]:
    """Register first-echo meGRE to T1w and warp T1w vertebral labels to meGRE."""
    inputs.validate_megre()
    inputs.validate_t1()
    first_echo = extract_first_echo(inputs)
    _require_output(inputs.sc_mask, "meGRE spinal-cord mask")
    _require_output(inputs.t1_sc_mask, "T1 spinal-cord mask")
    if inputs.legacy_vertebral_levels.is_file() and not inputs.vertebral_levels.exists():
        shutil.move(str(inputs.legacy_vertebral_levels), str(inputs.vertebral_levels))
    _require_output(inputs.vertebral_levels, "T1-space vertebral-level segmentation")
    inputs.registration_directory.mkdir(parents=True, exist_ok=True)
    outputs = (
        inputs.warp_megre_to_t1w,
        inputs.warp_t1w_to_megre,
        inputs.registered_megre,
        inputs.vertebral_levels_megre,
    )
    if force:
        for output in outputs:
            _remove_for_rerun(output, True)
    registration_ready = all(path.is_file() and path.stat().st_size > 0 for path in outputs[:3])
    if not registration_ready:
        runner([
            "sct_register_multimodal",
            "-i", str(first_echo), "-iseg", str(inputs.sc_mask),
            "-d", str(inputs.t1_path), "-dseg", str(inputs.t1_sc_mask),
            "-param", "step=1,type=seg,algo=centermass",
            "-o", str(inputs.registered_megre),
            "-owarp", str(inputs.warp_megre_to_t1w),
            "-owarpinv", str(inputs.warp_t1w_to_megre),
            "-ofolder", str(inputs.registration_directory),
        ], False)
    for path, label in (
        (inputs.warp_megre_to_t1w, "meGRE-to-T1w warp"),
        (inputs.warp_t1w_to_megre, "T1w-to-meGRE warp"),
        (inputs.registered_megre, "Registered meGRE"),
    ):
        _require_output(path, label)
    if not inputs.vertebral_levels_megre.is_file() or inputs.vertebral_levels_megre.stat().st_size == 0:
        runner([
            "sct_apply_transfo",
            "-i", str(inputs.vertebral_levels),
            "-d", str(first_echo),
            "-w", str(inputs.warp_t1w_to_megre),
            "-x", "nn",
            "-o", str(inputs.vertebral_levels_megre),
        ], False)
    _require_output(inputs.vertebral_levels_megre, "meGRE-space vertebral-level segmentation")
    return {
        "vertebral_levels": inputs.vertebral_levels,
        "warp_megre_to_t1w": inputs.warp_megre_to_t1w,
        "warp_t1w_to_megre": inputs.warp_t1w_to_megre,
        "registered_megre": inputs.registered_megre,
        "vertebral_levels_megre": inputs.vertebral_levels_megre,
    }


def update_project_masking(inputs: MaskingInputs, outputs: dict[str, Path], milestone: str | None = None) -> None:
    project_path = inputs.participant_directory / f"{inputs.participant_id}_qsm_project.json"
    if not project_path.is_file():
        raise MaskingError(f"Participant project file not found: {project_path}")
    project = json.loads(project_path.read_text(encoding="utf-8"))
    masking = project.setdefault("masking", {})
    masking.update({name: str(path.resolve()) for name, path in outputs.items()})
    if inputs.t1_path:
        masking["t1_path"] = str(inputs.t1_path.resolve())
    if milestone:
        mark_milestone(project, milestone)
        if milestone in {"megre_masks", "t1_sc_mask", "disc_labels", "vertebral_levels"}:
            mark_milestone(project, "mask_qc", False)
            mark_milestone(project, "registration", False)
            mark_milestone(project, "registration_qc", False)
        elif milestone == "registration":
            mark_milestone(project, "registration_qc", False)
    project_path.write_text(json.dumps(project, indent=2), encoding="utf-8")
