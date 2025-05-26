"""Compartmental models of infectious disease spread.

These models are based on the SIR (Susceptible-Infectious-Recovered) framework,
which is commonly used to model the spread of infectious diseases.

"""

from __future__ import annotations

from mxlpy import Model, fns

__all__ = ["get_sir", "get_sird"]


def get_sir() -> Model:
    """Create a simple SIR model."""
    return (
        Model()
        .add_variables({"s": 0.9, "i": 0.1, "r": 0.0})
        .add_parameters({"beta": 0.2, "gamma": 0.1})
        .add_reaction(
            "infection",
            fns.mass_action_2s,
            args=["s", "i", "beta"],
            stoichiometry={"s": -1, "i": 1},
        )
        .add_reaction(
            "recovery",
            fns.mass_action_1s,
            args=["i", "gamma"],
            stoichiometry={"i": -1, "r": 1},
        )
    )


def get_sird() -> Model:
    """Create a simple SIR model with death."""
    return (
        get_sir()
        .add_variable("d", 0.0)
        .add_parameter("mu", 0.01)
        .add_reaction(
            "death",
            fns.mass_action_1s,
            args=["i", "mu"],
            stoichiometry={"i": -1, "d": 1},
        )
    )
