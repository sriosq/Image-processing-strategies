import json
from pathlib import Path

import nibabel as nib
import numpy as np

from qsm_pp_gui.masking import MaskingInputs, create_disc_labels, create_megre_masks, extract_first_echo, register_megre_to_t1w, update_project_masking


def make_inputs(tmp_path: Path) -> MaskingInputs:
    magnitude = tmp_path / "magnitude.nii.gz"
    image = nib.Nifti1Image(np.zeros((4, 5, 6, 2), dtype=np.float32), np.eye(4))
    image.get_fdata()[:, :, :, 0].fill(1)
    nib.save(image, magnitude)
    participant = tmp_path / "output" / "sub-001"
    participant.mkdir(parents=True)
    (participant / "sub-001_qsm_project.json").write_text("{}", encoding="utf-8")
    return MaskingInputs("sub-001", magnitude, None, tmp_path / "output")


def test_extract_first_echo_is_3d_and_prefixed(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    output = extract_first_echo(inputs)
    assert output.name == "sub-001_echo-1_magnitude.nii.gz"
    assert nib.load(output).shape == (4, 5, 6)
    assert output.parent.name == "masking"


def test_megre_mask_commands_and_project_update(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    commands: list[list[str]] = []

    def runner(command: list[str], interactive: bool) -> None:
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"NIFTI")

    outputs = create_megre_masks(inputs, runner)
    update_project_masking(inputs, outputs)

    assert commands[0][:2] == ["sct_deepseg", "spinalcord"]
    assert commands[1][:2] == ["sct_deepseg", "graymatter"]
    assert commands[2][0] == "sct_maths"
    assert commands[2][commands[2].index("-sub") + 1].endswith("sub-001_desc-GM_mask.nii.gz")
    project = json.loads((inputs.participant_directory / "sub-001_qsm_project.json").read_text(encoding="utf-8"))
    assert project["masking"]["wm_mask"].endswith("sub-001_desc-WM_mask.nii.gz")


def test_cached_4d_first_echo_is_replaced_with_3d(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    inputs.masking_directory.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6, 2)), np.eye(4)), inputs.first_echo)
    output = extract_first_echo(inputs)
    assert nib.load(output).shape == (4, 5, 6)


def test_forced_disc_relabel_uses_range_and_invalidates_levels(tmp_path: Path) -> None:
    base = make_inputs(tmp_path)
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6)), np.eye(4)), t1)
    inputs = MaskingInputs(base.participant_id, base.magnitude_path, t1, base.output_root)
    inputs.masking_directory.mkdir(parents=True, exist_ok=True)
    inputs.disc_labels.write_bytes(b"OLD")
    inputs.vertebral_levels.write_bytes(b"STALE")
    commands: list[list[str]] = []

    def runner(command: list[str], interactive: bool) -> None:
        commands.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"NEW")

    create_disc_labels(inputs, 3, 7, runner=runner, force=True)

    assert commands[0][commands[0].index("-create-viewer") + 1] == "3:7"
    assert inputs.disc_labels.read_bytes() == b"NEW"
    assert not inputs.vertebral_levels.exists()


def test_registration_uses_named_warps_and_nn_labels(tmp_path: Path) -> None:
    base = make_inputs(tmp_path)
    t1 = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6)), np.eye(4)), t1)
    inputs = MaskingInputs(base.participant_id, base.magnitude_path, t1, base.output_root)
    extract_first_echo(inputs)
    for path in (inputs.sc_mask, inputs.t1_sc_mask, inputs.vertebral_levels):
        path.write_bytes(b"NIFTI")
    commands: list[list[str]] = []

    def runner(command: list[str], interactive: bool) -> None:
        commands.append(command)
        for option in ("-o", "-owarp", "-owarpinv"):
            if option in command:
                Path(command[command.index(option) + 1]).write_bytes(b"NIFTI")

    outputs = register_megre_to_t1w(inputs, runner=runner)

    assert commands[0][0] == "sct_register_multimodal"
    assert commands[0][commands[0].index("-param") + 1] == "step=1,type=seg,algo=centermass"
    assert outputs["warp_megre_to_t1w"].name == "warp_megre2t1w.nii.gz"
    assert outputs["warp_t1w_to_megre"].name == "warp_t1w2megre.nii.gz"
    assert commands[1][0] == "sct_apply_transfo"
    assert commands[1][commands[1].index("-x") + 1] == "nn"
    assert outputs["vertebral_levels_megre"].name == "sub-001_space-meGRE_desc-SC_vertlevels_dseg.nii.gz"
