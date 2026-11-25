# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Run the demo's real fits once and keep a snapshot of every step result.

The decisions are the ones record.spec.ts makes in the browser. Snapshots are
keyed by the step's input hash, so replay_lab.py finds the result that
matches the decisions made there. However long the fits take doesn't matter:
the replay sets the pace of the video.

usage: python record_runs.py <output dir>
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from demo_lab import make


def main() -> None:
    """Record run A (k_out fixed: the fit fails) and run B (all estimated)."""
    out = Path(sys.argv[1])
    shutil.rmtree(out, ignore_errors=True)
    snapshots = out / "snapshots"
    snapshots.mkdir(parents=True)
    run = make(out / "run")
    offset = 0

    def snapshot(step: str) -> None:
        """Keep the step's result and the events it wrote."""
        nonlocal offset
        res = run.results[step]
        events, offset = run.store.read_events(offset)
        dest = snapshots / res.input_hash
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(run.store.step_dir(step), dest / "result")
        (dest / "events.json").write_text(json.dumps(events))
        print(
            f"{step:16s} passed={res.passed} {res.duration:.1f} s, {len(events)} events"
        )

    run.step0_setup()
    snapshot("setup")
    run.decide("setup")
    run.step1_select_parameters()
    snapshot("parameters")
    proposed = run.results["parameters"].proposal["parameters"]
    run.decide("parameters", parameters=[p for p in proposed if p != "k_out"])
    run.step2_fit()
    snapshot("fit")
    # Reopened in the browser, then the proposal accepted
    run.reopen("parameters")
    run.decide("parameters")
    run.step2_fit()
    snapshot("fit")
    run.decide("fit", next="identifiability")
    run.run()  # skips diagnose, stops at the identifiability gate
    snapshot("identifiability")
    run.decide("identifiability")
    run.run()
    snapshot("uncertainty")
    run.decide("uncertainty")


if __name__ == "__main__":
    main()
