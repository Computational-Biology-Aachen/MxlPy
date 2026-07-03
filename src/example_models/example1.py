"""Example model 1."""

from __future__ import annotations

from example_models.utils import filter_stoichiometry
from mxlpy import KineticModelBuilder
from mxlpy.fns import constant, michaelis_menten_2s

__all__ = ["get_example1"]


def get_example1() -> KineticModelBuilder:
    """Example model 1."""
    model = KineticModelBuilder()
    model.add_variables({"x2": 0.0, "x3": 0.0})
    model.add_parameters(
        {
            # These need to be static in order to train the model later
            "x1": 1.0,
            "ATP": 1.0,
            "NADPH": 1.0,
            # v2
            "vmax_v2": 2.0,
            "km_v2_1": 0.1,
            "km_v2_2": 0.1,
            # v3
            "vmax_v3": 2.0,
            "km_v3_1": 0.2,
            "km_v3_2": 0.2,
        }
    )

    model.add_reaction(
        "v2",
        michaelis_menten_2s,
        args=["x1", "ATP", "vmax_v2", "km_v2_1", "km_v2_2"],
        stoichiometry=filter_stoichiometry(model, {"x1": -1, "ATP": -1, "x2": 1}),
    )
    model.add_reaction(
        "v3",
        michaelis_menten_2s,
        args=["x1", "NADPH", "vmax_v3", "km_v3_1", "km_v3_2"],
        stoichiometry=filter_stoichiometry(model, {"x1": -1, "NADPH": -1, "x3": 1}),
    )
    model.add_reaction(
        "x2_out",
        constant,
        args=["x2"],
        stoichiometry={"x2": -1},
    )
    model.add_reaction(
        "x3_out",
        constant,
        args=["x3"],
        stoichiometry={"x3": -1},
    )

    return model
