# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Serve a small fitting lab run for the Playwright smoke test.

Run from the MxlPy root: `uv run python web/lab/tests/e2e/serve.py [port]`.
"""

import sys
import tempfile

import numpy as np

from mxlpy import KineticModelBuilder, Simulator, fns, lab

TOKEN = "e2e-token"  # noqa: S105


def main() -> None:
    """Serve a fresh run of a two-step decay model."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    model = (
        KineticModelBuilder()
        .add_parameters({"k1": 0.5, "k2": 2.0})
        .add_variables({"x": 2.0, "y": 0.0})
        .add_reaction(
            "v1", fns.mass_action_1s, args=["x", "k1"], stoichiometry={"x": -1, "y": 1}
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["y", "k2"], stoichiometry={"y": -1}
        )
    )
    t = np.linspace(0.25, 8, 20)
    res = Simulator(model).simulate_time_course(t).get_result().unwrap_or_err()
    data = res.variables.loc[t, ["x", "y"]]
    data += np.random.default_rng(1).normal(0, 0.01, data.shape)
    run = lab.FittingLab(
        model,
        data,
        tempfile.mkdtemp(prefix="mxlpy-lab-e2e-"),
        n_samples=4,
        n_starts=3,
        n_pull_fits=2,
        prediction_times=3,
        parallel=False,
        max_hidden=1,
        structure_search=False,
    )
    server = run.serve(port=port, token=TOKEN, open_browser=False)
    server.wait()


if __name__ == "__main__":
    main()
