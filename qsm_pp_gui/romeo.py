"""ROMEO command construction kept independent of the GUI."""

from __future__ import annotations

from pathlib import Path

from .config import ToolConfig


def build_romeo_command(
    config: ToolConfig,
    phase_path: str,
    magnitude_path: str,
    echo_times: list[float],
    output_path: str,
    mask_path: str | None = None,
    phase_offset_correction: str | None = None,
    unwrap: bool = False,
) -> list[str]:
    if not echo_times:
        raise ValueError("At least one echo time is required")
    if phase_offset_correction not in {None, "bipolar", "on", "off"}:
        raise ValueError("Phase offset correction must be bipolar, on, or off")
    command = [
        "julia",
        str(Path(config.romeo_script).expanduser()),
        "-p", phase_path,
        "-m", magnitude_path,
    ]
    if mask_path:
        command += ["-k", mask_path, "-B", "-Q"]
    else:
        command += ["-B", "-Q"]
    command += ["-t", "[" + ", ".join(map(str, echo_times)) + "]", "-o", output_path]
    if unwrap:
        command.insert(command.index("-o"), "-u")
    if phase_offset_correction:
        command += ["--phase-offset-correction", phase_offset_correction]
    return command
