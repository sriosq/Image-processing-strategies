import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from qsm_pp_gui.matlab_header import create_matlab_header


class FakeEngine:
    def __init__(self) -> None:
        self.added_path = None
        self.quit_called = False

    def addpath(self, path: str, nargout: int) -> None:
        self.added_path = path

    def create_sepia_header(self, json_path: str, output_path: str, nargout: int) -> str:
        Path(output_path).write_bytes(b"MAT")
        return output_path

    def quit(self) -> None:
        self.quit_called = True


def test_matlab_header_updates_project(tmp_path: Path, monkeypatch) -> None:
    header = tmp_path / "sub-001_sepia_header.json"
    project = tmp_path / "sub-001_qsm_project.json"
    header.write_text("{}", encoding="utf-8")
    project.write_text(json.dumps({"sepia_header": str(header)}), encoding="utf-8")
    fake_engine = FakeEngine()
    matlab_module = ModuleType("matlab")
    engine_module = ModuleType("matlab.engine")
    engine_module.start_matlab = lambda: fake_engine
    matlab_module.engine = engine_module
    monkeypatch.setitem(sys.modules, "matlab", matlab_module)
    monkeypatch.setitem(sys.modules, "matlab.engine", engine_module)

    result = create_matlab_header(header, project)

    assert result == header.with_suffix(".mat").resolve()
    assert json.loads(project.read_text(encoding="utf-8"))["sepia_header_mat"] == str(result)
    assert fake_engine.added_path.endswith("utils")
    assert fake_engine.quit_called
