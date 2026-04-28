"""Example models of linear chains of reactions."""

from __future__ import annotations

from mxlpy import Model
from mxlpy.fns import constant, mass_action_1s, michaelis_menten_1s

__all__ = ["get_lin_chain_two_circles", "get_linear_chain_1v", "get_linear_chain_2v"]


def get_linear_chain_1v() -> Model:
    """Linear chain of reactions with one variable."""
    return (
        Model()
        .add_variables({"x": 1.0})
        .add_parameters({"k1": 1.0, "k_out": 1.0})
        .add_reaction(
            "v_in",
            constant,
            stoichiometry={"x": 1},
            args=["k1"],
        )
        .add_reaction(
            "v_out",
            mass_action_1s,
            stoichiometry={"x": -1},
            args=["k_out", "x"],
        )
    )


def get_linear_chain_2v() -> Model:
    """Linear chain of reactions with two variables."""
    return (
        Model()
        .add_variables({"x": 1.0, "y": 1.0})
        .add_parameters({"k1": 1.0, "k2": 2.0, "k_out": 1.0})
        .add_reaction(
            "v1",
            constant,
            stoichiometry={"x": 1},
            args=["k1"],
        )
        .add_reaction(
            "v2",
            mass_action_1s,
            stoichiometry={"x": -1, "y": 1},
            args=["k2", "x"],
        )
        .add_reaction(
            "v3",
            mass_action_1s,
            stoichiometry={"y": -1},
            args=["k_out", "y"],
        )
    )


def get_lin_chain_two_circles() -> Model:
    """Linear chain of reactions with two circles."""
    return (
        Model()
        .add_variables({"x1": 1.0, "x2": 1.0, "x3": 1.0, "x4": 1.0})
        .add_parameters(
            {
                "k0": 1.0,
                "vmax_1": 1.0,
                "km_1": 0.1,
                "vmax_2": 0.5,
                "km_2": 0.2,
                "vmax_3": 2.0,
                "km_3": 0.2,
                "k4": 0.5,
                "vmax_5": 1.0,
                "km_5": 0.3,
                "vmax_6": 1.0,
                "km_6": 0.4,
            }
        )
        .add_reaction("v0", constant, args=["k0"], stoichiometry={"x1": 1})
        .add_reaction(
            "v1",
            michaelis_menten_1s,
            args=["x1", "vmax_1", "km_1"],
            stoichiometry={"x1": -1, "x2": 1},
        )
        .add_reaction(
            "v2",
            michaelis_menten_1s,
            args=["x1", "vmax_2", "km_2"],
            stoichiometry={"x1": -1, "x3": 1},
        )
        .add_reaction(
            "v3",
            michaelis_menten_1s,
            args=["x1", "vmax_3", "km_3"],
            stoichiometry={"x1": -1, "x4": 1},
        )
        .add_reaction(
            "v4",
            mass_action_1s,
            args=["x4", "k4"],
            stoichiometry={"x4": -1},
        )
        .add_reaction(
            "v5",
            michaelis_menten_1s,
            args=["x2", "vmax_5", "km_5"],
            stoichiometry={"x2": -1, "x1": 1},
        )
        .add_reaction(
            "v6",
            michaelis_menten_1s,
            args=["x3", "vmax_6", "km_6"],
            stoichiometry={"x3": -1, "x1": 1},
        )
    )
