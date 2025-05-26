"""Example models for phase plane analysis."""

from __future__ import annotations

from mxlpy import Model, fns

__all__ = ["get_phase_plane", "v1"]


def v1(s2: float, k: float, k1: float, n: float) -> float:
    """First reaction."""
    return k1 / (1 + (s2 / k) ** n)


def get_phase_plane() -> Model:
    """Get phase plane model."""
    return (
        Model()
        .add_variables({"s1": 1.0, "s2": 1.0})
        .add_parameters(
            {
                "k1": 20,
                "k2": 5,
                "k3": 5,
                "k4": 5,
                "k5": 2,
                "K": 1,
                "n": 4,
            }
        )
        .add_reaction(
            "v1",
            v1,
            args=["s2", "K", "k1", "n"],
            stoichiometry={"s1": 1},
        )
        .add_reaction(
            "v2",
            fns.constant,
            args=["k2"],
            stoichiometry={"s2": 1},
        )
        .add_reaction(
            "v3",
            fns.mass_action_1s,
            args=["s1", "k3"],
            stoichiometry={"s1": -1},
        )
        .add_reaction(
            "v4",
            fns.mass_action_1s,
            args=["s2", "k4"],
            stoichiometry={"s2": -1},
        )
        .add_reaction(
            "v5",
            fns.mass_action_1s,
            args=["s1", "k5"],
            stoichiometry={"s1": -1, "s2": 1},
        )
    )
