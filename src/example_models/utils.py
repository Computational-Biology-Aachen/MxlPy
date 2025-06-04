"""General utility functions for example models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mxlpy import Model

__all__ = ["filter_stoichiometry"]


def filter_stoichiometry(
    model: Model,
    stoichiometry: dict[str, float],
) -> dict[str, float]:
    """Only use components that are actually compounds in the model.

    Args:
        model: Metabolic model instance
        stoichiometry: Stoichiometry dictionary {component: value}

    """
    new: dict[str, float] = {}
    ids = model.ids
    variables = set(model.get_variable_names())
    for k, v in stoichiometry.items():
        if k in variables:
            new[k] = v
        elif k not in ids:
            msg = f"Missing component {k}"
            raise KeyError(msg)
    return new
