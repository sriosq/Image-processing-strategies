import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from qsm_pp_gui.project import load_project, milestone_complete, milestone_summary


def test_load_project_validates_headers_and_reconstructs_milestones(tmp_path: Path) -> None:
    participant = tmp_path / "sub-001"
    participant.mkdir()
    header_json = participant / "sub-001_sepia_header.json"
    header_mat = participant / "sub-001_sepia_header.mat"
    project_path = participant / "sub-001_qsm_project.json"
    magnitude = tmp_path / "magnitude.nii.gz"
    phase = tmp_path / "phase.nii.gz"
    magnitude.touch()
    phase.touch()
    header = {"TE": [0.004, 0.008], "B0": 3, "CF": 123000000, "B0_dir": [0, 0, 1], "matrix_size": [4, 5, 6], "voxel_size": [1, 1, 2]}
    header_json.write_text(json.dumps(header), encoding="utf-8")
    savemat(header_mat, {"TE": np.array(header["TE"]), "B0": 3, "CF": 123000000, "B0_dir": np.array([[0], [0], [1]]), "matrixSize": np.array([4, 5, 6]), "voxelSize": np.array([1, 1, 2])})
    project_path.write_text(json.dumps({"participant_id": "sub-001", "magnitude_path": str(magnitude), "phase_path": str(phase), "output_directory": str(participant), "sepia_header": str(header_json), "sepia_header_mat": str(header_mat)}), encoding="utf-8")

    project, loaded_header = load_project(project_path)

    assert loaded_header["TE"] == [0.004, 0.008]
    assert milestone_complete(project, "inputs_header")
    assert "Next step: meGRE masks" in milestone_summary(project)
