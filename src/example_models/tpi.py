"""Example label models."""

from __future__ import annotations

from mxlpy import Model, fns

__all__ = ["get_tpi_ald_model"]


def get_tpi_ald_model() -> Model:
    """Create model of triose phosphate isomerase and aldolase reactions.

    This is mostly used for showing a simple label propagation example.
    """
    p = {
        "kf_TPI": 1.0,
        "Keq_TPI": 21.0,
        "kf_Ald": 2000.0,
        "Keq_Ald": 7000.0,
    }
    p["kr_TPI"] = p["kf_TPI"] / p["Keq_TPI"]
    p["kr_Ald"] = p["kf_Ald"] / p["Keq_Ald"]

    GAP0 = 2.5e-5
    DHAP0 = GAP0 * p["Keq_TPI"]
    FBP0 = GAP0 * DHAP0 * p["Keq_Ald"]

    y0 = {"GAP": GAP0, "DHAP": DHAP0, "FBP": FBP0}

    return (
        Model()
        .add_variables(y0)
        .add_parameters(p)
        .add_reaction(
            "TPIf",
            fns.mass_action_1s,
            args=["GAP", "kf_TPI"],
            stoichiometry={"GAP": -1, "DHAP": 1},
        )
        .add_reaction(
            "TPIr",
            fns.mass_action_1s,
            args=["DHAP", "kr_TPI"],
            stoichiometry={"DHAP": -1, "GAP": 1},
        )
        .add_reaction(
            "ALDf",
            fns.mass_action_2s,
            args=["DHAP", "GAP", "kf_Ald"],
            stoichiometry={"DHAP": -1, "GAP": -1, "FBP": 1},
        )
        .add_reaction(
            "ALDr",
            fns.mass_action_1s,
            args=["FBP", "kr_Ald"],
            stoichiometry={
                "FBP": -1,
                "DHAP": 1,
                "GAP": 1,
            },
        )
    )
