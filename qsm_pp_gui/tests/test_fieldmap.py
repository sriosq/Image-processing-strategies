import json
from pathlib import Path

import nibabel as nib
import numpy as np

from qsm_pp_gui.config import ToolConfig
from qsm_pp_gui.fieldmap import FieldmapInputs, create_fieldmap_pngs, run_fieldmap, update_project_fieldmap


def test_masked_and_unmasked_romeo_commands_and_milestone(tmp_path: Path) -> None:
    participant = tmp_path / "sub-001"
    participant.mkdir()
    magnitude = tmp_path / "mag.nii.gz"
    phase = tmp_path / "phase.nii.gz"
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6, 2)), np.eye(4)), magnitude)
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6, 2)), np.eye(4)), phase)
    nib.save(nib.Nifti1Image(np.ones((4, 5, 6)), np.eye(4)), mask)
    project_path = participant / "sub-001_qsm_project.json"
    project_path.write_text(json.dumps({"participant_id": "sub-001"}), encoding="utf-8")
    inputs = FieldmapInputs("sub-001", magnitude, phase, mask, [6.93, 11.85], tmp_path, "bipolar")
    config = ToolConfig(romeo_script="C:/romeo/romeo.jl")
    commands: list[list[str]] = []

    def runner(command: list[str], interactive: bool) -> None:
        commands.append(command)
        output_directory = Path(command[command.index("-o") + 1])
        (output_directory / "B0.nii").write_bytes(b"B0-NIFTI")
        (output_directory / "corrected_phase.nii").write_bytes(b"PHASE-NIFTI")

    masked = run_fieldmap(inputs, config, True, runner)
    update_project_fieldmap(inputs, masked_output=masked)
    unmasked = run_fieldmap(inputs, config, False, runner)
    update_project_fieldmap(inputs, unmasked_output=unmasked)

    assert masked["b0"].name == "sub-001_desc-b0_fieldmap.nii.gz"
    assert masked["corrected_phase"].name == "sub-001_desc-corrected_phase.nii.gz"
    assert (masked["b0"].parent / "B0.nii").is_file()
    assert "-k" in commands[0] and "-u" in commands[0]
    assert "-k" in commands[1] and "-u" not in commands[1]
    assert commands[0][commands[0].index("-t") + 1] == "[6.93, 11.85]"
    assert commands[0][commands[0].index("--phase-offset-correction") + 1] == "bipolar"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    assert project["milestones"]["field_map"]["complete"] is True

    def png_runner(command: list[str], interactive: bool) -> None:
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"PNG")

    pngs = create_fieldmap_pngs(inputs, -3, 3, "bwr", runner=png_runner)
    assert pngs["masked"].name == "sub-001_desc-b0_fieldmap.png"
    assert pngs["unmasked"].name == "sub-001_desc-b0_fieldmap.png"
    assert commands[-1][0] == "sct_deepseb"
    assert commands[-1][commands[-1].index("-cbar") + 1] == "bwr"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    assert project["milestones"]["fieldmap_visualization"]["complete"] is True
    assert project["milestones"]["fieldmap_qc"]["complete"] is False

    command_count = len(commands)
    updated_pngs = create_fieldmap_pngs(inputs, -0.5, 0.5, "viridis", runner=png_runner)
    assert len(commands) == command_count + 2
    updated_command = commands[-1]
    assert updated_command[updated_command.index("-cmin") + 1] == "-0.5"
    assert updated_command[updated_command.index("-cmax") + 1] == "0.5"
    assert updated_command[updated_command.index("-cbar") + 1] == "viridis"
    assert updated_pngs["masked"].is_file()
