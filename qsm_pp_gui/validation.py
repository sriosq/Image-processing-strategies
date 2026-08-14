"""Read-only checks for external programs and toolbox directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

from .config import ToolConfig


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _resolve_program(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate)
    return shutil.which(value)


def _program_check(name: str, value: str, version_args: list[str]) -> CheckResult:
    resolved = _resolve_program(value)
    if not resolved:
        return CheckResult(name, False, f"Not found: {value or '(not configured)'}")
    try:
        result = subprocess.run(
            [resolved, *version_args], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult(name, False, f"Could not run {resolved}: {exc}")
    output = (result.stdout or result.stderr).strip().splitlines()
    detail = output[0] if output else resolved
    return CheckResult(name, result.returncode == 0, detail)


def _directory_check(name: str, value: str) -> CheckResult:
    if not value:
        return CheckResult(name, False, "Not configured")
    path = Path(value).expanduser()
    return CheckResult(name, path.is_dir(), str(path) if path.is_dir() else f"Directory not found: {path}")


def validate_tools(config: ToolConfig) -> list[CheckResult]:
    results = [
        _program_check("Julia (PATH)", "julia", ["--version"]),
        _program_check("Spinal Cord Toolbox (PATH)", "sct_deepseg", ["-h"]),
        _program_check("SCT DeepSeg visualization (PATH)", "sct_deepseb", ["-h"]),
    ]
    romeo = Path(config.romeo_script).expanduser() if config.romeo_script else None
    results.append(CheckResult("ROMEO script", bool(romeo and romeo.is_file()), str(romeo) if romeo and romeo.is_file() else "File not found or not configured"))
    results.extend(
        [
            _directory_check("SEPIA", config.sepia_directory),
        ]
    )
    try:
        import matlab.engine  # type: ignore[import-not-found]  # noqa: F401
    except (ImportError, OSError) as exc:
        results.append(CheckResult("MATLAB Engine for Python", False, str(exc)))
    else:
        results.append(CheckResult("MATLAB Engine for Python", True, "Python package is importable"))
    return results
