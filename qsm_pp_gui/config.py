"""Persistent, machine-specific configuration for external QSM tools."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path


def default_config_path() -> Path:
    """Return the visible, project-local configuration path."""
    return Path(__file__).resolve().parent / "config.json"


@dataclass(slots=True)
class ToolConfig:
    romeo_script: str = ""
    sepia_directory: str = ""
    developer_mode: bool = False

    @classmethod
    def load(cls, path: Path | None = None) -> "ToolConfig":
        path = path or default_config_path()
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in allowed})

    def save(self, path: Path | None = None) -> Path:
        path = path or default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path
