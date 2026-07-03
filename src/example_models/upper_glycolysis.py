"""Upper glycolysis model developed by Klipp et al. (2005)."""

from __future__ import annotations

from mxlpy import KineticModelBuilder
from mxlpy.fns import constant, mass_action_1s, mass_action_1s_1p, mass_action_2s

__all__ = ["get_model"]


def get_model() -> KineticModelBuilder:
    """Upper glycolysis model developed by Klipp et al. (2005)."""
    m = KineticModelBuilder()
    m.add_parameters(
        {
            "k1": 0.25,
            "k2": 1,
            "k3": 1,
            "k3m": 1,
            "k4": 1,
            "k5": 1,
            "k6": 1,
            "k7": 2.5,
        }
    )
    m.add_variables(
        {
            "GLC": 0,
            "G6P": 0,
            "F6P": 0,
            "FBP": 0,
            "ATP": 0.5,
            "ADP": 0.5,
        }
    )

    m.add_reaction(
        "v1",
        fn=constant,
        stoichiometry={"GLC": 1},
        args=["k1"],
    )
    m.add_reaction(
        "v2",
        fn=mass_action_2s,
        stoichiometry={"GLC": -1, "ATP": -1, "G6P": 1, "ADP": 1},
        args=["GLC", "ATP", "k2"],
    )
    m.add_reaction(
        "v3",
        fn=mass_action_1s_1p,
        stoichiometry={"G6P": -1, "F6P": 1},
        args=["G6P", "F6P", "k3", "k3m"],
    )
    m.add_reaction(
        "v4",
        fn=mass_action_2s,
        stoichiometry={"F6P": -1, "ATP": -1, "ADP": 1, "FBP": 1},
        args=["F6P", "ATP", "k4"],
    )
    m.add_reaction(
        "v5",
        fn=mass_action_1s,
        stoichiometry={"FBP": -1, "F6P": 1},
        args=["FBP", "k5"],
    )
    m.add_reaction(
        "v6",
        fn=mass_action_1s,
        stoichiometry={"FBP": -1},
        args=["FBP", "k6"],
    )
    m.add_reaction(
        "v7",
        fn=mass_action_1s,
        stoichiometry={"ADP": -1, "ATP": 1},
        args=["ADP", "k7"],
    )

    return m
