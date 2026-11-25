# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Serve the real lab UI over a replay of recorded fits (see record_runs.py).

The server only watches (it never computes). A replay thread plays the part
of the driver: when the decisions ask for a step, it looks up the recorded
result with the same input hash, streams its progress events squeezed into
DURATION seconds and then stores the result, so the UI sees a live run.

usage: python replay_lab.py <recording dir> [port]
"""

import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

from mxlpy import lab
from mxlpy.lab._core import STEPS, StepName

# Seconds a step seems to take in the video
DURATION = {
    "setup": 0.6,
    "parameters": 0.8,
    "fit": 3.0,
    "diagnose": 3.0,
    "identifiability": 3.0,
    "uncertainty": 3.0,
}
TOKEN = "demo"  # noqa: S105


class Replay:
    """Plays recorded step results into a run directory, as a driver would."""

    def __init__(self, recording: Path, run_dir: Path) -> None:
        shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True)
        for name in ("config.json", "inputs.pkl", "runtime.json"):
            shutil.copy(recording / "run" / name, run_dir / name)
        self.run = lab.FittingLab.resume(run_dir)
        self.snapshots = {d.name: d for d in (recording / "snapshots").iterdir()}

    def play(self, step: StepName) -> bool:
        """Stream the recorded events of a step, then store its result."""
        run = self.run
        snap = self.snapshots.get(run._hash(step))  # noqa: SLF001
        if snap is None:
            print(f"No recording of {step} for these decisions", flush=True)
            return False
        target = DURATION[step]
        events = [
            e
            for e in json.loads((snap / "events.json").read_text())
            if e.get("step") == step
            and (e.get("source") != "lab" or e.get("kind") == "step_started")
        ]
        t0, t1 = events[0]["time"], events[-1]["time"]
        scale = target / max(t1 - t0, 1e-9)
        start = time.time()
        for event in events:
            ev = dict(event)
            due = start + (ev.pop("time") - t0) * scale
            time.sleep(max(due - time.time(), 0))
            if ev.get("kind") == "step_started" and ev.get("expected"):
                ev["expected"] = [target * 0.8, target, target * 1.4]
            if "duration" in ev and ev.get("source") != "lab":
                ev["duration"] *= scale
            run.store.append_event(ev)
        time.sleep(max(start + target - time.time(), 0))
        result = run.store.read_pickle(snap / "result" / "result.pkl")
        result.duration = target
        result.messages = [
            re.sub(
                r"Estimated runtime of steps 2-5: .*",
                "Estimated runtime of steps 2-5: 9 s to 20 s (typ. 12 s)",
                m,
            )
            for m in result.messages
        ]
        run._save(result)  # noqa: SLF001
        run.store.append_event(
            {
                "source": "lab",
                "kind": "step_finished",
                "step": step,
                "passed": result.passed,
                "duration": target,
            }
        )
        return True

    def next_step(self) -> tuple[StepName | None, str]:
        """The next step to run or the open gate, like `FittingLab._advance`."""
        run = self.run
        run._refresh()  # noqa: SLF001
        for step in STEPS:
            if step == "diagnose" and not run._needs_diagnosis():  # noqa: SLF001
                if not run.is_current(step):
                    run._skip(step)  # noqa: SLF001
                if run.decision(step) is None:
                    run._record(step, {}, "implicit")  # noqa: SLF001
                continue
            if not run.is_current(step):
                return step, "run"
            if run.decision(step) is None:
                return step, "gate"
        return None, "end"

    def loop(self) -> None:
        """Run forever, holding the run lock so the UI shows a driven run."""
        announced = None
        with self.run.store.driving():
            while True:
                step, what = self.next_step()
                if what == "run":
                    if self.run.paused:
                        time.sleep(0.2)
                        continue
                    announced = None
                    time.sleep(0.4)  # a moment between steps
                    if not self.play(step):  # type: ignore[arg-type]
                        time.sleep(1)
                    continue
                decisions = json.dumps(self.run.store.decisions(), sort_keys=True)
                if (mark := (what, step, decisions)) != announced:
                    announced = mark
                    kind = "gate_open" if what == "gate" else "run_finished"
                    event = {"source": "lab", "kind": kind}
                    self.run.store.append_event(
                        event if step is None else {**event, "step": step}
                    )
                time.sleep(0.2)


def main() -> None:
    """Serve the replay until killed."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    recording = Path(sys.argv[1])
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8770  # noqa: PLR2004
    replay = Replay(recording, recording.parent / "replay-run")
    server = replay.run.serve(port=port, token=TOKEN, open_browser=False, watch=True)
    threading.Thread(target=replay.loop, daemon=True).start()
    server.wait()


if __name__ == "__main__":
    main()
