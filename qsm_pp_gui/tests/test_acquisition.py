import json
from pathlib import Path

from qsm_pp_gui.acquisition import Acquisition, parse_numbers


def test_parse_numbers_accepts_common_separators() -> None:
    assert parse_numbers("4.5, 8.5 12.5") == [4.5, 8.5, 12.5]


def test_save_header_uses_participant_prefix_and_sepia_units(tmp_path: Path) -> None:
    magnitude = tmp_path / "magnitude.nii.gz"
    phase = tmp_path / "phase.nii.gz"
    magnitude.touch()
    phase.touch()
    acquisition = Acquisition("sub-001", str(magnitude), str(phase), str(tmp_path / "output"), [4.0, 8.0], 3.0, [0.0, 0.0, 1.0], 123_000_000.0, [128, 128, 64], [1.0, 1.0, 2.0])
    header_path, project_path = acquisition.save()
    assert header_path.name == "sub-001_sepia_header.json"
    assert project_path.name == "sub-001_qsm_project.json"
    header = json.loads(header_path.read_text(encoding="utf-8"))
    assert header["TE"] == [0.004, 0.008]
    assert header["B0_dir"] == [0.0, 0.0, 1.0]
