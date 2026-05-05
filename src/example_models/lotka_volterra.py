"""Lotka-Volterra predator-prey model (v2): predation uses derived stoichiometry."""

from mxlpy import Model, fns
from mxlpy.types import Derived

__all__ = ["create_model"]


def _prey_growth(
    alpha: float,
    prey: float,
) -> float:
    """Prey intrinsic growth: Alpha * Prey."""
    return alpha * prey


def _predation(
    predator: float,
    prey: float,
) -> float:
    """Predation encounter rate: Predator * Prey."""
    return predator * prey


def _predator_death(
    predator: float,
    gamma: float,
) -> float:
    """Predator natural death: Gamma * Predator."""
    return gamma * predator


def create_model() -> Model:
    """Build the Lotka-Volterra predator-prey model (v2) with derived stoichiometry."""
    return (
        Model()
        .add_variable(
            "Prey",
            initial_value=10.0,
        )
        .add_variable(
            "Predator",
            initial_value=10.0,
        )
        .add_parameter(
            "Alpha",
            value=0.1,
        )
        .add_parameter(
            "Beta",
            value=0.02,
        )
        .add_parameter(
            "Gamma",
            value=0.4,
        )
        .add_parameter(
            "Delta",
            value=0.02,
        )
        .add_reaction(
            "prey_growth",
            fn=_prey_growth,
            args=["Alpha", "Prey"],
            stoichiometry={"Prey": 1.0},
        )
        .add_reaction(
            "predation",
            fn=_predation,
            args=["Predator", "Prey"],
            stoichiometry={
                "Prey": Derived(
                    fn=fns.neg,
                    args=["Beta"],
                ),
                "Predator": "Delta",
            },
        )
        .add_reaction(
            "predator_death",
            fn=_predator_death,
            args=["Predator", "Gamma"],
            stoichiometry={"Predator": -1.0},
        )
    )
