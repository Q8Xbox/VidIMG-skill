"""Adapters that drive the installed engines through their official interfaces.

Every adapter: never writes next to or over its inputs, logs the exact command (proc.run),
and returns an AdapterResult whose `ok` is judged by the produced files, not exit codes alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..proc import RunResult


class PolicyError(ValueError):
    """Raised when a request violates the approved fidelity policy (e.g. generative without opt-in)."""


@dataclass
class AdapterResult:
    ok: bool
    outputs: list[Path] = field(default_factory=list)
    error: str | None = None
    run: RunResult | None = None
    details: dict = field(default_factory=dict)

    @property
    def output(self) -> Path | None:
        return self.outputs[0] if self.outputs else None


def safe_output_dir(output_dir: Path, *inputs: Path) -> Path:
    """Refuse to write into a folder that holds an input (protects originals); create it otherwise."""
    out = Path(output_dir).resolve()
    for i in inputs:
        if Path(i).resolve().parent == out:
            raise PolicyError(f"output folder {out} contains the input {i}; choose a separate folder")
    out.mkdir(parents=True, exist_ok=True)
    return out


OUTPUT_MARKER = ".vidimg-output"


def mark_output(folder: Path) -> None:
    """Leave a small marker in a results folder, so a later scan knows its work folders are not the person's media."""
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / OUTPUT_MARKER).touch()
    except OSError:
        pass
