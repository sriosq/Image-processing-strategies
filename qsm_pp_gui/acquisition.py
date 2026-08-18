"""Validated acquisition metadata and SEPIA header creation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_numbers(value: str, expected: int | None = None, integer: bool = False) -> list[float] | list[int]:
    parts = [part for part in re.split(r"[,;\s]+", value.strip()) if part]
    if expected is not None and len(parts) != expected:
        raise ValueError(f"Expected {expected} values, received {len(parts)}")
    if not parts:
        raise ValueError("At least one value is required")
    try:
        values = [int(part) if integer else float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("Values must be numbers separated by spaces or commas") from exc
    return values


@dataclass(frozen=True, slots=True)
class Acquisition:
    participant_id: str
    magnitude_path: str
    phase_path: str
    output_directory: str
    echo_times_ms: list[float]
    b0_tesla: float
    b0_direction: list[float]
    central_frequency_hz: float
    matrix_size: list[int]
    voxel_size_mm: list[float]
    echo_time_input_unit: str = "ms"
    central_frequency_input_unit: str = "Hz"
    echo_times_entered: list[float] | None = None
    central_frequency_entered: float | None = None

    def validate(self) -> None:
        if not _ID_PATTERN.fullmatch(self.participant_id):
            raise ValueError("Participant ID may contain only letters, numbers, dots, underscores, and hyphens")
        for label, value in (("Magnitude", self.magnitude_path), ("Phase", self.phase_path)):
            if not Path(value).is_file():
                raise ValueError(f"{label} file does not exist: {value}")
        if any(value <= 0 for value in self.echo_times_ms):
            raise ValueError("Echo times must be positive")
        if sorted(self.echo_times_ms) != self.echo_times_ms:
            raise ValueError("Echo times must be in increasing order")
        if self.b0_tesla <= 0 or self.central_frequency_hz <= 0:
            raise ValueError("B0 and central frequency must be positive")
        if len(self.b0_direction) != 3 or not any(self.b0_direction):
            raise ValueError("B0 direction must contain three values and cannot be the zero vector")
        if len(self.matrix_size) != 3 or any(value <= 0 for value in self.matrix_size):
            raise ValueError("Matrix size must contain three positive integers")
        if len(self.voxel_size_mm) != 3 or any(value <= 0 for value in self.voxel_size_mm):
            raise ValueError("Voxel size must contain three positive values")

    def sepia_header(self) -> dict[str, object]:
        """Return fields and units used by SEPIA's JSON header."""
        self.validate()
        return {
            "TE": [value / 1000.0 for value in self.echo_times_ms],
            "B0": self.b0_tesla,
            "CF": self.central_frequency_hz,
            "B0_dir": self.b0_direction,
            "matrix_size": self.matrix_size,
            "voxel_size": self.voxel_size_mm,
        }

    def save(self) -> tuple[Path, Path]:
        self.validate()
        participant_dir = Path(self.output_directory) / self.participant_id
        participant_dir.mkdir(parents=True, exist_ok=True)
        header_path = participant_dir / f"{self.participant_id}_sepia_header.json"
        project_path = participant_dir / f"{self.participant_id}_qsm_project.json"
        header_path.write_text(json.dumps(self.sepia_header(), indent=2), encoding="utf-8")
        project = {
            "participant_id": self.participant_id,
            "magnitude_path": str(Path(self.magnitude_path).resolve()),
            "phase_path": str(Path(self.phase_path).resolve()),
            "output_directory": str(participant_dir.resolve()),
            "sepia_header": str(header_path.resolve()),
            "acquisition_input": {
                "echo_times": self.echo_times_entered if self.echo_times_entered is not None else self.echo_times_ms,
                "echo_time_unit": self.echo_time_input_unit,
                "central_frequency": self.central_frequency_entered if self.central_frequency_entered is not None else self.central_frequency_hz,
                "central_frequency_unit": self.central_frequency_input_unit,
            },
        }
        project_path.write_text(json.dumps(project, indent=2), encoding="utf-8")
        return header_path, project_path
