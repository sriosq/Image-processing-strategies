from pathlib import Path

import pytest

from qsm_pp_gui.config import ToolConfig
from qsm_pp_gui.romeo import build_romeo_command


def test_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    expected = ToolConfig(romeo_script="C:/ROMEO/romeo.jl")
    expected.save(path)
    assert ToolConfig.load(path) == expected


def test_romeo_command_is_argument_list() -> None:
    config = ToolConfig(romeo_script="C:/Program Files/ROMEO/romeo.jl")
    command = build_romeo_command(config, "phase file.nii.gz", "mag.nii.gz", [0.004, 0.008], "out dir", mask_path="mask.nii.gz")
    assert command[:2] == ["julia", "C:\\Program Files\\ROMEO\\romeo.jl"]
    assert command[command.index("-t") + 1] == "[0.004, 0.008]"
    assert command[command.index("-k") + 1] == "mask.nii.gz"


def test_romeo_requires_echo_times() -> None:
    with pytest.raises(ValueError, match="echo time"):
        build_romeo_command(ToolConfig(), "phase", "mag", [], "out")
