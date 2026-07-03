"""Test nested function calls in MXLpy."""

from __future__ import annotations

from mxlpy import KineticModelBuilder
from mxlpy.fns import constant

__all__ = ["get_model", "wrapper"]


def wrapper(x: float) -> float:
    """Wrapper function to call a constant function."""
    return constant(x)


def get_model() -> KineticModelBuilder:
    """Create a model with a nested function call."""
    return (
        KineticModelBuilder()
        .add_variables({"x": 0})
        .add_reaction("v1", wrapper, args=["x"], stoichiometry={"x": -1})
    )
