# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""The run directory of the fitting lab, the source of truth of a run.

Layout::

    run_dir/
        config.json        settings of the run (human-readable)
        inputs.pkl         model, data and further Python inputs (dill)
        decisions.json     gate decisions by step
        events.jsonl       progress events, one JSON object per line
        runtime.json       state of the runtime model
        steps/<step>/
            result.json    summary, proposal and input hash of the step
            result.pkl     the full `StepResult` (dill)
        report.html, report.json

Writes of JSON files are atomic (write and rename), so readers (e.g. the
web UI) never see half-written files.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dill
import numpy as np
import pandas as pd

try:
    import fcntl
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["RunStore", "to_json"]


def to_json(obj: Any) -> Any:
    """JSON-compatible form: non-finite floats become None, pandas become dicts."""
    if isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_json(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, pd.Series):
        return {str(k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return {
            "index": [to_json(i) for i in obj.index],
            "columns": [str(c) for c in obj.columns],
            "data": [[to_json(v) for v in row] for row in obj.to_numpy().tolist()],
        }
    if isinstance(obj, np.ndarray):
        return to_json(obj.tolist())
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


class RunStore:
    """Files of one run."""

    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        self.root = Path(run_dir)

    # Paths ###################################################################

    @property
    def config_file(self) -> Path:
        """Settings of the run."""
        return self.root / "config.json"

    @property
    def inputs_file(self) -> Path:
        """Pickled model, data and further inputs."""
        return self.root / "inputs.pkl"

    @property
    def decisions_file(self) -> Path:
        """Gate decisions."""
        return self.root / "decisions.json"

    @property
    def events_file(self) -> Path:
        """Progress events."""
        return self.root / "events.jsonl"

    @property
    def runtime_file(self) -> Path:
        """State of the runtime model."""
        return self.root / "runtime.json"

    def step_dir(self, step: str) -> Path:
        """Directory of the artifacts of a step."""
        return self.root / "steps" / step

    # JSON ####################################################################

    def write_json(self, path: Path, data: Any) -> None:
        """Write JSON atomically (unique temporary file, then rename)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(to_json(data), indent=1))
        Path(tmp).replace(path)

    @staticmethod
    def read_json(path: Path, default: Any = None) -> Any:
        """Read JSON, `default` if the file doesn't exist."""
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    @contextlib.contextmanager
    def _locked(self, path: Path) -> Iterator[None]:
        """Exclusive lock of `path` across processes (where `fcntl` exists)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_name(path.name + ".lock").open("w") as f:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def write_pickle(self, path: Path, data: Any) -> None:
        """Write a dill pickle atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        with os.fdopen(fd, "wb") as f:
            dill.dump(data, f)
        Path(tmp).replace(path)

    @staticmethod
    def read_pickle(path: Path) -> Any:
        """Read a dill pickle (only of runs you trust, like any pickle)."""
        with path.open("rb") as f:
            return dill.load(f)  # nosec

    # Decisions ###############################################################

    def decisions(self) -> dict[str, dict[str, Any]]:
        """Gate decisions by step."""
        return dict(self.read_json(self.decisions_file, {}))

    def write_decision(self, step: str, decision: dict[str, Any]) -> None:
        """Record the decision of a step's gate (safe against concurrent writers)."""
        with self._locked(self.decisions_file):
            decisions = self.decisions()
            decisions[step] = decision
            self.write_json(self.decisions_file, decisions)

    def drop_decision(self, step: str) -> None:
        """Forget the decision of a step's gate."""
        with self._locked(self.decisions_file):
            decisions = self.decisions()
            if decisions.pop(step, None) is not None:
                self.write_json(self.decisions_file, decisions)

    # Events ##################################################################

    def append_event(self, event: dict[str, Any]) -> None:
        """Append one event (a single write, safe across processes)."""
        self.root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(to_json({"time": time.time(), **event})) + "\n"
        fd = os.open(self.events_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)

    def read_events(self, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """Complete events from byte `offset` on, and the offset after them."""
        if not self.events_file.exists():
            return [], offset
        with self.events_file.open("rb") as f:
            f.seek(offset)
            chunk = f.read()
        end = chunk.rfind(b"\n") + 1
        events = []
        for line in chunk[:end].splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events, offset + end
