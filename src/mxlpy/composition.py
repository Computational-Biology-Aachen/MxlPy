"""Hierarchical model composition.

Merge multiple :class:`~mxlpy.model.Model` instances into a single model,
identifying shared components by name. This enables building large models from
smaller, independently tested submodules (e.g. a metabolic core plus a
regulatory layer) rather than as one monolithic model.
"""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mxlpy.model import Model

__all__ = ["compose"]


def _remove_by_ctx(model: Model, name: str, ctx: str) -> None:
    """Remove *name* from *model*, dispatching on its namespace context.

    Variables are removed without cascading into reaction stoichiometries, so a
    later model can re-supply the variable while existing reactions keep
    referencing it by name.
    """
    if ctx == "parameter":
        model.remove_parameter(name)
    elif ctx == "variable":
        model.remove_variable(name, remove_stoichiometries=False)
    elif ctx == "derived":
        model.remove_derived(name)
    elif ctx == "reaction":
        model.remove_reaction(name)
    elif ctx == "readout":
        model.remove_readout(name)
    elif ctx == "data":
        model.remove_data(name)
    elif ctx == "surrogate" and name in model.get_raw_surrogates(as_copy=False):
        # Surrogate output names also live in the namespace; only whole
        # surrogates can be removed by name (output collisions are best-effort).
        model.remove_surrogate(name)


def _merge_into(result: Model, model: Model, *, raise_on_conflict: bool) -> None:
    """Fold all components of *model* into *result* in place."""
    if duplicates := sorted(set(result.ids) & set(model.ids)):
        if raise_on_conflict:
            msg = (
                f"Cannot compose models: duplicate names {duplicates}. "
                "Rename the conflicting components in one of the models, or pass "
                "raise_on_conflict=False to let the later model override them."
            )
            raise ValueError(msg)
        for name in duplicates:
            warnings.warn(
                f"Component {name!r} is overridden by a later model during "
                "composition.",
                UserWarning,
                stacklevel=3,
            )
            _remove_by_ctx(result, name, result.ids[name])

    result.add_parameters(model.get_raw_parameters())
    result.add_variables(model.get_raw_variables())
    for name, derived in model.get_raw_derived().items():
        result.add_derived(name, derived.fn, args=derived.args, unit=derived.unit)
    for name, reaction in model.get_raw_reactions().items():
        result.add_reaction(
            name,
            reaction.fn,
            args=reaction.args,
            stoichiometry=dict(reaction.stoichiometry),
            unit=reaction.unit,
        )
    for name, readout in model.get_raw_readouts().items():
        result.add_readout(name, readout.fn, args=readout.args, unit=readout.unit)
    for name, surrogate in model.get_raw_surrogates().items():
        result.add_surrogate(name, copy.deepcopy(surrogate))
    for name, data in model._data.items():  # noqa: SLF001
        result.add_data(name, copy.deepcopy(data))


def compose(*models: Model, raise_on_conflict: bool = True) -> Model:
    """Merge multiple models into a single new model.

    Components are identified by name across the global namespace (variables,
    parameters, derived quantities, reactions, readouts, surrogates and data).
    Models are folded left-to-right into a copy of the first model; the inputs
    are never mutated.

    Examples
    --------
        >>> full_model = compose(signalling, metabolic)

    Parameters
    ----------
    models
        Two or more models to merge. A single model returns a copy.
    raise_on_conflict
        If ``True`` (default), any name defined in more than one model raises a
        ``ValueError``. If ``False``, a warning is emitted and the component
        from the later model overrides the earlier one.

    Returns
    -------
    Model
        A new model containing the union of all components.

    Raises
    ------
    ValueError
        If no models are given, or if names collide and
        ``raise_on_conflict`` is ``True``.

    """
    if not models:
        msg = "compose() requires at least one model"
        raise ValueError(msg)

    result = copy.deepcopy(models[0])
    for model in models[1:]:
        _merge_into(result, model, raise_on_conflict=raise_on_conflict)
    return result
