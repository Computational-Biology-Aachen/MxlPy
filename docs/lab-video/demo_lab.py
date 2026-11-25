# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Synthetic fitting lab run of the demo video: a linear chain x -> y -> z.

Two experiments start from different amounts of x. The run starts from wrong
parameter values, so the fit has something to do.
"""

from pathlib import Path

import numpy as np

from mxlpy import KineticModelBuilder, Simulator, fit, fns, lab

TRUE = {"k_in": 1.0, "k1": 0.5, "k2": 2.0, "k_out": 0.3}
START = {"k_in": 0.4, "k1": 1.5, "k2": 0.8, "k_out": 1.0}
SETTINGS = {
    "n_samples": 8,
    "n_starts": 24,
    "n_pull_fits": 4,
    "prediction_times": 6,
    "parallel": False,  # progress events in order, one start after the other
    "max_hidden": 1,
    "structure_search": False,
}


def model(parameters: dict[str, float]) -> KineticModelBuilder:
    """Inflow of x, x -> y -> z, outflow of z."""
    return (
        KineticModelBuilder()
        .add_parameters(parameters)
        .add_variables({"x": 2.0, "y": 0.0, "z": 0.0})
        .add_reaction("v_in", fns.constant, args=["k_in"], stoichiometry={"x": 1})
        .add_reaction(
            "v1", fns.mass_action_1s, args=["x", "k1"], stoichiometry={"x": -1, "y": 1}
        )
        .add_reaction(
            "v2", fns.mass_action_1s, args=["y", "k2"], stoichiometry={"y": -1, "z": 1}
        )
        .add_reaction(
            "v_out", fns.mass_action_1s, args=["z", "k_out"], stoichiometry={"z": -1}
        )
    )


def data() -> dict[str, fit.Condition]:
    """Time courses of the true model with additive noise (sd 0.03)."""
    rng = np.random.default_rng(3)
    t = np.linspace(0.25, 12, 30)
    out = {}
    for name, x0 in [("low", 0.5), ("high", 3.0)]:
        m = model(TRUE)
        m.update_variable("x", x0)
        res = Simulator(m).simulate_time_course(t).get_result().unwrap_or_err()
        df = res.variables.loc[t, ["x", "y", "z"]]
        out[name] = fit.Condition(df + rng.normal(0, 0.03, df.shape), y0={"x": x0})
    return out


def make(run_dir: Path) -> lab.FittingLab:
    """A fresh run of the demo in `run_dir`."""
    return lab.FittingLab(model(START), data(), run_dir, **SETTINGS)
