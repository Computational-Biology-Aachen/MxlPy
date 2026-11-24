# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Native JSON serialisation for MxlPy models.

This module reads and writes the ``.mxl.json`` format: a MxlPy-native, version
controllable representation that captures the full model structure (variables,
parameters, reactions, derived quantities and readouts).

Rate expressions are stored as trees of math nodes using the node set shared
with MxlWeb (:mod:`mxlpy.meta._mathml`), so the same files can be consumed by
both tools. Models round-trip through sympy, which means a loaded model is
behaviourally identical to the original and ``save -> load -> save`` reaches a
stable fixed point.

Examples
--------
>>> import mxlpy
>>> mxlpy.save(model, "my_model.mxl.json")
>>> model = mxlpy.load("my_model.mxl.json")

"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import pandas as pd
import sympy

from mxlpy._hidden_states import ACTIVATIONS, HiddenStates, NNLayer, init_hidden_states
from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.meta import _mathml as mml
from mxlpy.meta.array_fns import ARRAY_NODE_TYPES, array_lengths
from mxlpy.meta.source_tools import fn_to_sympy_expr
from mxlpy.meta.sympy_tools import (
    compile_fn,
    list_of_symbols,
    mathml_to_sympy,
    sympy_to_mathml,
)
from mxlpy.types import (
    Derived,
    InitialAssignment,
    Mechanism,
    SerializationError,
    Slider,
)
from mxlpy.units import unit_from_json, unit_to_json

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    from mxlpy.types import (
        DiffEq,
        Parameter,
        RateFn,
        Reaction,
        Readout,
        Term,
        Variable,
    )

__all__ = [
    "ODE_SCHEMA_URL",
    "SCHEMA_URL",
    "SPEC_VERSION",
    "STEADY_STATE_SCHEMA_URL",
    "load",
    "model_from_dict",
    "model_to_dict",
    "mxl_json_weights_files",
    "ode_model_from_dict",
    "ode_model_to_dict",
    "mxl_json_data_file",
    "ode_mxl_json_weights_files",
    "save",
    "steady_state_model_from_dict",
    "steady_state_model_to_dict",
]

SPEC_VERSION = "0.2"
# Written instead of SPEC_VERSION by a model with data (`data_ref`).
DATA_SPEC_VERSION = "0.3"
# Written by a model whose arrays depend on more than data (ADR 0015).
ARRAY_SPEC_VERSION = "0.4"
# Written by a model with terms or hidden-state blocks (ADR 0008 §6).
TERMS_SPEC_VERSION = "0.5"
# The schema is maintained in the shared, language-agnostic `mxl-schemas` repo
# so it can be consumed by both mxlpy and mxlweb. It is intentionally not
# vendored into this package. The URL is pinned to the major spec version, so a
# file's $schema always matches the spec_version it was written with.
_SCHEMA_MAJOR = SPEC_VERSION.split(".", 1)[0]
SCHEMA_URL = (
    "https://raw.githubusercontent.com/Computational-Biology-Aachen/"
    f"mxl-schemas/main/v{_SCHEMA_MAJOR}/kinetic-model.schema.json"
)
ODE_SCHEMA_URL = (
    "https://raw.githubusercontent.com/Computational-Biology-Aachen/"
    f"mxl-schemas/main/v{_SCHEMA_MAJOR}/ode-model.schema.json"
)
STEADY_STATE_SCHEMA_URL = (
    "https://raw.githubusercontent.com/Computational-Biology-Aachen/"
    f"mxl-schemas/main/v{_SCHEMA_MAJOR}/steady-state-model.schema.json"
)

_FN_NAME = "_mxl_fn"


###############################################################################
# Save
###############################################################################


def _names_to_dict(
    entity: Variable | Parameter | Derived | Readout | Reaction | DiffEq | Term,
) -> dict[str, Any]:
    names: dict[str, Any] = {
        "displayName": entity.display_name,
        "texName": entity.tex_name,
    }
    if (slider := getattr(entity, "slider", None)) is not None:
        names["slider"] = {
            k: v for k, v in dataclasses.asdict(slider).items() if v is not None
        }
    return {k: v for k, v in names.items() if v is not None}


def _fn_to_node_dict(
    fn: RateFn,
    *,
    origin: str,
    args: list[str],
) -> dict[str, Any]:
    expr = fn_to_sympy_expr(fn, origin=origin, model_args=list_of_symbols(args))
    if expr is None:
        msg = f"could not convert function of {origin!r} to an expression"
        raise SerializationError(msg)
    return sympy_to_mathml(expr).to_dict()


def _value_to_node_dict(
    value: float | InitialAssignment,
    *,
    origin: str,
) -> dict[str, Any]:
    if isinstance(value, InitialAssignment):
        return _fn_to_node_dict(value.fn, origin=origin, args=value.args)
    return mml.Num(value=float(value)).to_dict()


def _stoich_to_node_dict(
    value: float | Derived,
    *,
    origin: str,
) -> dict[str, Any]:
    if isinstance(value, Derived):
        return _fn_to_node_dict(value.fn, origin=origin, args=value.args)
    return mml.Num(value=float(value)).to_dict()


def _with_unit(
    entry: dict[str, Any],
    unit: sympy.Expr | None,
    customs: dict[str, dict[str, str]],
    *,
    origin: str,
) -> dict[str, Any]:
    """Attach `unit` to a serialised entity, collecting custom kinds into `customs`."""
    if unit is not None:
        try:
            entry["unit"] = unit_to_json(unit, customs)
        except ValueError as exc:
            msg = f"unit of {origin!r}: {exc}"
            raise SerializationError(msg) from exc
    return entry


def _unexportable_surrogates(model: KineticModelBuilder) -> list[str]:
    """Every attached surrogate that can't be exported as a nn_blocks entry, with why.

    A surrogate whose `to_mxl_json()` returns `None` (the base-class
    default — e.g. `_qss.py`/`_poly.py`) declines silently, by design; an
    NN-backend surrogate instead raises `SerializationError` once it's
    committed to trying (see
    `mxlpy.surrogates.abstract.AbstractSurrogate.to_mxl_json`'s docstring)
    — both outcomes are collected here, so `model_to_dict` can report one
    aggregate error covering either kind of unexportable surrogate.
    """
    unexportable: list[str] = []
    for name, surrogate in model._surrogates.items():  # noqa: SLF001
        try:
            export = surrogate.to_mxl_json()
        except SerializationError as exc:
            unexportable.append(f"{name} ({exc})")
            continue
        if export is None:
            unexportable.append(name)
    return sorted(unexportable)


def _check_surrogate_order(
    model: KineticModelBuilder | OdeModelBuilder, name: str, targets: list[str]
) -> None:
    """Refuse a surrogate whose target has non-additive terms.

    A loaded surrogate is a block, which is applied after the terms, while
    the surrogate is part of the terms' base, so the dynamics would change.
    """
    for term_name, term in model.get_raw_terms(as_copy=False).items():
        if term.target in targets and term.mechanism.name != "additive":
            msg = (
                f"surrogate {name!r} and the non-additive term {term_name!r} act on "
                f"{term.target!r}: a file applies terms before nn_blocks, which "
                "would change the dynamics"
            )
            raise SerializationError(msg)


def _mxl_json_blocks_and_weights(
    model: KineticModelBuilder,
) -> tuple[dict[str, Any], dict[str, dict[str, list[Any]]]]:
    """The `nn_blocks` section and, separately, every trained block's weights sidecar content.

    Weight/bias values are never inlined into `nn_blocks` itself (mirrors
    mxl-schemas' own split from `parameters`) — `spec["weights_ref"]` names
    a `<block>.weights.json` sidecar file whose content is this function's
    second return value, keyed by the same relative filename `save()`
    writes it under.

    Callers are expected to have already checked
    :func:`_unexportable_surrogates` — a surrogate whose `to_mxl_json()`
    raises isn't caught here, and propagates.
    """
    nn_blocks: dict[str, Any] = {}
    weights_files: dict[str, dict[str, list[Any]]] = {}
    for name, surrogate in model._surrogates.items():  # noqa: SLF001
        export = surrogate.to_mxl_json()
        if export is None:
            continue
        spec = dict(export.spec)
        spec.setdefault("target_kind", "variable")
        _check_surrogate_order(model, name, cast(list[str], spec["targets"]))
        if spec["trained"]:
            weights_ref = f"{name}.weights.json"
            spec["weights_ref"] = weights_ref
            weights_files[weights_ref] = export.weights
        nn_blocks[name] = spec
    blocks, block_weights = _blocks_to_dict(model)
    return nn_blocks | blocks, weights_files | block_weights


def mxl_json_weights_files(
    model: KineticModelBuilder,
) -> dict[str, dict[str, list[Any]]]:
    """Weights files of the exportable surrogates and the hidden-state blocks, keyed by the relative filename their `nn_blocks[id].weights_ref` names.

    Pairs with :func:`model_to_dict`'s output (which references these same
    filenames but never inlines their content) to produce the full
    multi-file export; :func:`save` calls this and writes each file
    alongside the main one. A model with no NN blocks (the common case)
    returns an empty dict.
    """
    _, weights_files = _mxl_json_blocks_and_weights(model)
    return weights_files


def mxl_json_data_file(
    model: KineticModelBuilder | OdeModelBuilder,
) -> dict[str, Any] | None:
    """The model's data sidecar (mxl-schemas `model-data.schema.json`), or None without data.

    Raises
    ------
    SerializationError
        If a data set isn't 1-D (a `pd.Series`).

    """
    out: dict[str, Any] = {}
    for name, values in model.get_raw_data().items():
        if not isinstance(values, pd.Series):
            msg = (
                f"data {name!r} is a {type(values).__name__}; only 1-D data "
                "(a pd.Series) can be saved, split a table into one series per column"
            )
            raise SerializationError(msg)
        out[name] = {"values": [float(x) for x in values.to_numpy()]}
    return out or None


def _dynamic_arrays(section: Mapping[str, Any], data: Mapping[str, Any]) -> bool:
    """Whether an array in the model section depends on more than data and numbers."""
    data = {k: np.asarray(v, dtype=float) for k, v in data.items()}
    exprs = {
        f"{kind}:{k}": mathml_to_sympy(mml.node_from_dict(v["fn"]))
        for kind in ("derived", "reactions", "readouts", "variables")
        for k, v in section.get(kind, {}).items()
        if "fn" in v
    }
    derived = [
        (k.removeprefix("derived:"), e)
        for k, e in exprs.items()
        if k.startswith("derived:")
    ]
    if array_lengths(derived, data):
        return True
    return any(
        not {str(s) for s in node.free_symbols} <= set(data)
        for e in exprs.values()
        for node in sympy.preorder_traversal(e)
        if type(node) in ARRAY_NODE_TYPES
    )


def _envelope(
    model: KineticModelBuilder | OdeModelBuilder,
    data_ref: str,
    section: Mapping[str, Any],
) -> dict[str, Any]:
    """`spec_version` (the lowest one with all features used) and `data_ref`."""
    uses_terms = bool(section.get("terms")) or any(
        "hidden_states" in b or "input_scales" in b or "output_scales" in b
        for b in section.get("nn_blocks", {}).values()
    )
    base = TERMS_SPEC_VERSION if uses_terms else SPEC_VERSION
    if not (data := model.get_raw_data()):
        return {"spec_version": base}
    try:
        dynamic = _dynamic_arrays(section, data)
    except ValueError as e:
        raise SerializationError(str(e)) from e
    version = ARRAY_SPEC_VERSION if dynamic else DATA_SPEC_VERSION
    return {"spec_version": max(base, version), "data_ref": data_ref}


def model_to_dict(
    model: KineticModelBuilder,
    *,
    model_id: str,
    description: str = "",
    data_ref: str = "model.data.json",
) -> dict[str, Any]:
    """Convert a model into its ``.mxl.json`` dict representation.

    Parameters
    ----------
    model
        Model to serialise
    model_id
        Identifier stored in the file
    description
        Human-readable description stored in the file

    Returns
    -------
    dict
        JSON-compatible mapping following the ``mxl-model`` schema

    Raises
    ------
    SerializationError
        If the model contains a surrogate that isn't representable as a
        mxl-schemas ``nn_blocks`` entry (see
        :meth:`mxlpy.surrogates.abstract.AbstractSurrogate.to_mxl_json`),
        or a rate function that cannot be converted into an expression.

    """
    if unexportable := _unexportable_surrogates(model):
        names = ", ".join(unexportable)
        msg = f"surrogates are not representable as nn_blocks: {names}"
        raise SerializationError(msg)
    nn_blocks, _ = _mxl_json_blocks_and_weights(model)

    customs: dict[str, dict[str, str]] = {}
    variables = {
        name: _with_unit(
            {
                "value": _value_to_node_dict(var.initial_value, origin=name),
                **_names_to_dict(var),
            },
            var.unit,
            customs,
            origin=name,
        )
        for name, var in model.get_raw_variables().items()
    }
    parameters = {
        name: _with_unit(
            {
                "value": _value_to_node_dict(par.value, origin=name),
                **_names_to_dict(par),
            },
            par.unit,
            customs,
            origin=name,
        )
        for name, par in model.get_raw_parameters().items()
    }
    reactions = {
        name: _with_unit(
            {
                "fn": _fn_to_node_dict(rxn.fn, origin=name, args=rxn.args),
                "stoichiometry": {
                    var: _stoich_to_node_dict(stoich, origin=f"{name}:{var}")
                    for var, stoich in rxn.stoichiometry.items()
                },
                **_names_to_dict(rxn),
            },
            rxn.unit,
            customs,
            origin=name,
        )
        for name, rxn in model.get_raw_reactions().items()
    }
    derived = {
        name: _with_unit(
            {
                "fn": _fn_to_node_dict(der.fn, origin=name, args=der.args),
                **_names_to_dict(der),
            },
            der.unit,
            customs,
            origin=name,
        )
        for name, der in model.get_raw_derived().items()
    }
    readouts = {
        name: _with_unit(
            {
                "fn": _fn_to_node_dict(rdt.fn, origin=name, args=rdt.args),
                **_names_to_dict(rdt),
            },
            rdt.unit,
            customs,
            origin=name,
        )
        for name, rdt in model.get_raw_readouts().items()
    }

    model_section: dict[str, Any] = {
        "variables": variables,
        "parameters": parameters,
        "reactions": reactions,
        "derived": derived,
        "readouts": readouts,
    }
    if nn_blocks:
        model_section["nn_blocks"] = nn_blocks
    if terms := _terms_to_dict(model, customs):
        model_section["terms"] = terms
    if customs:
        model_section["units"] = customs

    return {
        "$schema": SCHEMA_URL,
        **_envelope(model, data_ref, model_section),
        "kind": "kinetic",
        "model_id": model_id,
        "description": description,
        "model": model_section,
    }


###############################################################################
# Save — OdeModelBuilder
###############################################################################


def _unexportable_ode_surrogates(model: OdeModelBuilder) -> list[str]:
    """Every attached ode surrogate that can't be exported as a nn_blocks entry, with why.

    Ode-model counterpart of :func:`_unexportable_surrogates`.
    """
    unexportable: list[str] = []
    for name, surrogate in model._surrogates.items():  # noqa: SLF001
        try:
            export = surrogate.to_mxl_json()
        except SerializationError as exc:
            unexportable.append(f"{name} ({exc})")
            continue
        if export is None:
            unexportable.append(name)
    return sorted(unexportable)


def _ode_mxl_json_blocks_and_weights(
    model: OdeModelBuilder,
) -> tuple[dict[str, Any], dict[str, dict[str, list[Any]]]]:
    """The `nn_blocks` section and weights sidecars for an `OdeModelBuilder`'s surrogates.

    Mirrors :func:`_mxl_json_blocks_and_weights` exactly, but reads
    `OdeModelBuilder._surrogates` (`OdeSurrogateProtocol`) instead of
    `KineticModelBuilder._surrogates` (`SurrogateProtocol`) — the two are
    unrelated types (see `mxlpy.surrogates.abstract_ode`'s module
    docstring), but both `to_mxl_json()` methods return the same
    schema-shaped ``spec``/``weights`` split, so the assembly logic here is
    identical.
    """
    nn_blocks: dict[str, Any] = {}
    weights_files: dict[str, dict[str, list[Any]]] = {}
    for name, surrogate in model._surrogates.items():  # noqa: SLF001
        export = surrogate.to_mxl_json()
        if export is None:
            continue
        spec = dict(export.spec)
        spec.setdefault("target_kind", "variable")
        _check_surrogate_order(model, name, cast(list[str], spec["targets"]))
        if spec["trained"]:
            weights_ref = f"{name}.weights.json"
            spec["weights_ref"] = weights_ref
            weights_files[weights_ref] = export.weights
        nn_blocks[name] = spec
    blocks, block_weights = _blocks_to_dict(model)
    return nn_blocks | blocks, weights_files | block_weights


def ode_mxl_json_weights_files(
    model: OdeModelBuilder,
) -> dict[str, dict[str, list[Any]]]:
    """Weights files of the exportable ode surrogates and the hidden-state blocks, keyed by their `weights_ref`.

    Ode-model counterpart of :func:`mxl_json_weights_files`.
    """
    _, weights_files = _ode_mxl_json_blocks_and_weights(model)
    return weights_files


def ode_model_to_dict(
    model: OdeModelBuilder,
    *,
    model_id: str,
    description: str = "",
    data_ref: str = "model.data.json",
) -> dict[str, Any]:
    """Convert an `OdeModelBuilder` model into its ``.mxl.json`` dict representation.

    Ode-model counterpart of :func:`model_to_dict` — same overall shape,
    but each variable carries its own dx/dt directly (`fn`) rather than
    being derived from reactions and stoichiometry, so there is no
    ``reactions`` section at all (`ode-model.schema.json`, discriminated
    from `kinetic-model.schema.json` by `kind`).

    Parameters
    ----------
    model
        Model to serialise
    model_id
        Identifier stored in the file
    description
        Human-readable description stored in the file

    Returns
    -------
    dict
        JSON-compatible mapping following the ``mxl-ode-model`` schema

    Raises
    ------
    SerializationError
        If the model contains a surrogate that isn't representable as a
        mxl-schemas ``nn_blocks`` entry, or a rate function that cannot be
        converted into an expression.

    """
    if unexportable := _unexportable_ode_surrogates(model):
        names = ", ".join(unexportable)
        msg = f"surrogates are not representable as nn_blocks: {names}"
        raise SerializationError(msg)
    nn_blocks, _ = _ode_mxl_json_blocks_and_weights(model)

    customs: dict[str, dict[str, str]] = {}
    variables = {
        name: _with_unit(
            {
                "value": _value_to_node_dict(diff_eq.initial_value, origin=name),
                "fn": _fn_to_node_dict(diff_eq.fn, origin=name, args=diff_eq.args),
                **_names_to_dict(diff_eq),
            },
            diff_eq.unit,
            customs,
            origin=name,
        )
        for name, diff_eq in model.get_raw_diff_eqs().items()
    }
    parameters = {
        name: _with_unit(
            {
                "value": _value_to_node_dict(par.value, origin=name),
                **_names_to_dict(par),
            },
            par.unit,
            customs,
            origin=name,
        )
        for name, par in model.get_raw_parameters().items()
    }
    derived = {
        name: _with_unit(
            {
                "fn": _fn_to_node_dict(der.fn, origin=name, args=der.args),
                **_names_to_dict(der),
            },
            der.unit,
            customs,
            origin=name,
        )
        for name, der in model.get_raw_derived().items()
    }
    readouts = {
        name: _with_unit(
            {
                "fn": _fn_to_node_dict(rdt.fn, origin=name, args=rdt.args),
                **_names_to_dict(rdt),
            },
            rdt.unit,
            customs,
            origin=name,
        )
        for name, rdt in model.get_raw_readouts().items()
    }

    model_section: dict[str, Any] = {
        "variables": variables,
        "parameters": parameters,
        "derived": derived,
        "readouts": readouts,
    }
    if nn_blocks:
        model_section["nn_blocks"] = nn_blocks
    if terms := _terms_to_dict(model, customs):
        model_section["terms"] = terms
    if customs:
        model_section["units"] = customs

    return {
        "$schema": ODE_SCHEMA_URL,
        **_envelope(model, data_ref, model_section),
        "kind": "ode",
        "model_id": model_id,
        "description": description,
        "model": model_section,
    }


###############################################################################
# Save — SteadyStateModelBuilder
###############################################################################


def steady_state_model_to_dict(
    model: SteadyStateModelBuilder,
    *,
    model_id: str,
    description: str = "",
) -> dict[str, Any]:
    """Convert a `SteadyStateModelBuilder` model into its ``.mxl.json`` dict representation.

    Steady-state counterpart of :func:`model_to_dict` / :func:`ode_model_to_dict`
    — there are no state variables and no time integration, so the ``model``
    section carries only ``parameters`` and ``derived`` (the algebraic
    outputs); no ``variables``, ``reactions``, ``readouts`` or ``nn_blocks``
    section exists at all, matching `steady-state-model.schema.json`
    (`additionalProperties: false`) and `SteadyStateModelBuilder` itself,
    which has no such concepts (no surrogates, no readouts) to export.

    Parameters
    ----------
    model
        Model to serialise
    model_id
        Identifier stored in the file
    description
        Human-readable description stored in the file

    Returns
    -------
    dict
        JSON-compatible mapping following the ``mxl-steady-state-model`` schema

    Raises
    ------
    SerializationError
        If a parameter or derived rate function cannot be converted into an
        expression.

    """
    customs: dict[str, dict[str, str]] = {}
    parameters = {
        name: _with_unit(
            {
                "value": _value_to_node_dict(par.value, origin=name),
                **_names_to_dict(par),
            },
            par.unit,
            customs,
            origin=name,
        )
        for name, par in model.get_raw_parameters().items()
    }
    derived = {
        name: _with_unit(
            {
                "fn": _fn_to_node_dict(der.fn, origin=name, args=der.args),
                **_names_to_dict(der),
            },
            der.unit,
            customs,
            origin=name,
        )
        for name, der in model.get_raw_derived().items()
    }

    model_section: dict[str, Any] = {
        "parameters": parameters,
        "derived": derived,
    }
    if customs:
        model_section["units"] = customs

    return {
        "$schema": STEADY_STATE_SCHEMA_URL,
        "spec_version": SPEC_VERSION,
        "kind": "steady-state",
        "model_id": model_id,
        "description": description,
        "model": model_section,
    }


def save(
    model: KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder,
    path: str | Path,
    *,
    model_id: str | None = None,
    description: str = "",
) -> None:
    """Save a model to the native ``.mxl.json`` format.

    Dispatches on `model`'s type: a `KineticModelBuilder` is written via
    :func:`model_to_dict` (`kinetic-model.schema.json`), an
    `OdeModelBuilder` via :func:`ode_model_to_dict`
    (`ode-model.schema.json`), a `SteadyStateModelBuilder` via
    :func:`steady_state_model_to_dict` (`steady-state-model.schema.json`).

    Every trained, exportable NN block's weights are written alongside the
    main file as `<block>.weights.json`, in the same directory `path`
    itself lands in (`nn_blocks[id].weights_ref` names it as a path
    relative to the main file, per mxl-schemas). A model's data, likewise,
    as `<model_id>.data.json` (`data_ref`, mxl-schemas ADR 0014).

    Parameters
    ----------
    model
        Model to save
    path
        Destination file path
    model_id
        Identifier stored in the file. Defaults to the file name stem.
    description
        Human-readable description stored in the file

    Raises
    ------
    SerializationError
        If the model contains a surrogate that isn't representable as a
        mxl-schemas ``nn_blocks`` entry, or a rate function that cannot be
        converted into an expression.

    """
    path = Path(path)
    if model_id is None:
        model_id = path.name.removesuffix(".json").removesuffix(".mxl")
    data_ref = f"{model_id}.data.json"
    data_file = None
    if isinstance(model, OdeModelBuilder):
        data = ode_model_to_dict(
            model, model_id=model_id, description=description, data_ref=data_ref
        )
        weights_files = ode_mxl_json_weights_files(model)
        data_file = mxl_json_data_file(model)
    elif isinstance(model, SteadyStateModelBuilder):
        data = steady_state_model_to_dict(
            model, model_id=model_id, description=description
        )
        weights_files = {}
    else:
        data = model_to_dict(
            model, model_id=model_id, description=description, data_ref=data_ref
        )
        weights_files = mxl_json_weights_files(model)
        data_file = mxl_json_data_file(model)
    path.write_text(json.dumps(data, indent=2) + "\n")
    for weights_ref, weights in weights_files.items():
        (path.parent / weights_ref).write_text(json.dumps(weights, indent=2) + "\n")
    if data_file is not None:
        (path.parent / data_ref).write_text(json.dumps(data_file, indent=2) + "\n")


###############################################################################
# Terms and hidden-state blocks (spec_version 0.5, mxl-schemas ADR 0008 §6)
###############################################################################

_ODE = sympy.Symbol("ode")
_NDE = sympy.Symbol("nde")
_X = sympy.Symbol("x")


def _mechanism_to_node_dict(mechanism: Mechanism, *, origin: str) -> dict[str, Any]:
    """A mechanism as an expression over the placeholders `ode` and `nde`."""
    if mechanism.name == "additive":
        expr: sympy.Expr = _ODE + _NDE
    elif mechanism.name == "relative":
        expr = _ODE * (1 + _NDE)
    else:
        parsed = fn_to_sympy_expr(mechanism.fn, origin=origin, model_args=[_ODE, _NDE])
        if parsed is None:
            msg = f"could not convert the mechanism of {origin!r} to an expression"
            raise SerializationError(msg)
        expr = parsed
    return sympy_to_mathml(expr).to_dict()


def _mechanism_from_node_dict(node: dict[str, Any], *, origin: str) -> Mechanism:
    expr = mathml_to_sympy(mml.node_from_dict(node))
    if unknown := sorted(str(s) for s in expr.free_symbols - {_ODE, _NDE}):
        msg = f"{origin!r}: a mechanism can only read `ode` and `nde`, not {unknown}"
        raise SerializationError(msg)
    if sympy.expand(expr - (_ODE + _NDE)) == 0:
        return Mechanism.of("additive")
    if sympy.expand(expr - _ODE * (1 + _NDE)) == 0:
        return Mechanism.of("relative")
    base, term = sympy.Symbol("base"), sympy.Symbol("term")
    fn = compile_fn(
        cast(sympy.Expr, expr.xreplace({_ODE: base, _NDE: term})),
        ["base", "term"],
        name="_mechanism",
    )
    return Mechanism(fn=fn)


def _target_kind(
    model: KineticModelBuilder | OdeModelBuilder, targets: list[str], *, origin: str
) -> str:
    reactions = set(getattr(model, "get_reaction_names", list)())
    kinds = {"reaction" if t in reactions else "variable" for t in targets}
    if len(kinds) > 1:
        msg = f"{origin!r} mixes variable and reaction targets, which a file can't hold"
        raise SerializationError(msg)
    return kinds.pop()


def _terms_to_dict(
    model: KineticModelBuilder | OdeModelBuilder,
    customs: dict[str, dict[str, str]],
) -> dict[str, Any]:
    return {
        name: _with_unit(
            {
                "fn": _fn_to_node_dict(term.fn, origin=name, args=term.args),
                "target": term.target,
                "target_kind": _target_kind(model, [term.target], origin=name),
                "mechanism": _mechanism_to_node_dict(term.mechanism, origin=name),
                **_names_to_dict(term),
            },
            term.unit,
            customs,
            origin=name,
        )
        for name, term in model.get_raw_terms(as_copy=False).items()
    }


def _activation_to_dict(layer: NNLayer) -> dict[str, Any]:
    name = cast(str, layer.activation)
    expr = layer.expression if layer.expression is not None else ACTIVATIONS[name](_X)
    return {"name": name, "expression": sympy_to_mathml(expr).to_dict()}


def _block_scale_value(
    model: KineticModelBuilder | OdeModelBuilder, name: str, block: HiddenStates
) -> float | None:
    if block.scale is None:
        return None
    if block.scale != f"{name}_scale":
        msg = f"nn_block {name!r}: its scale has to be the parameter {name}_scale"
        raise SerializationError(msg)
    return float(model.get_parameter_values()[block.scale])


def _blocks_to_dict(
    model: KineticModelBuilder | OdeModelBuilder,
) -> tuple[dict[str, Any], dict[str, dict[str, list[Any]]]]:
    """`nn_blocks` entries and weights files of the hidden-state blocks."""
    blocks: dict[str, Any] = {}
    weights_files: dict[str, dict[str, list[Any]]] = {}
    for name, block in model.get_raw_hidden_states(as_copy=False).items():
        if block.h0_assignments:
            msg = f"nn_block {name!r}: initial assignments of h(0) can't be saved"
            raise SerializationError(msg)
        weights_ref = f"{name}.weights.json"
        spec: dict[str, Any] = {
            "inputs": list(block.inputs),
            "layers": [
                {
                    "type": "dense",
                    "width": layer.width,
                    **(
                        {"activation": _activation_to_dict(layer)}
                        if layer.activation is not None
                        else {}
                    ),
                }
                for layer in block.layers
            ],
            "seed": block.seed,
            "target_kind": _target_kind(model, block.targets, origin=name),
            "targets": list(block.targets),
            "trained": True,
            "weights_ref": weights_ref,
            "mechanism": _mechanism_to_node_dict(block.mechanism, origin=name),
        }
        if (scale := _block_scale_value(model, name, block)) is not None:
            spec["scale"] = scale
        elif block.n_hidden == 0:
            spec["scale"] = 1.0
        if block.n_hidden > 0:
            spec["hidden_states"] = block.n_hidden
        if np.any(block.input_scales != 1):
            spec["input_scales"] = [float(i) for i in block.input_scales]
        if np.any(block.output_scales != 1):
            spec["output_scales"] = [float(i) for i in block.output_scales]
        weights: dict[str, list[Any]] = {}
        for i, (w, b) in enumerate(zip(block.weights, block.biases, strict=True)):
            weights[f"w{i + 1}"] = w.tolist()
            weights[f"b{i + 1}"] = b.tolist()
        if block.n_hidden > 0:
            weights["W"] = block.readout.tolist()
            weights["h0"] = block.h0.tolist()
        blocks[name] = spec
        weights_files[weights_ref] = weights
    return blocks, weights_files


def _array(values: Any, shape: tuple[int, ...], *, origin: str) -> Any:
    out = np.asarray(values, dtype=float)
    if out.shape != shape:
        msg = f"{origin}: expected shape {shape}, got {out.shape}"
        raise SerializationError(msg)
    return out


def _block_from_mxl_json(
    model: KineticModelBuilder | OdeModelBuilder,
    name: str,
    block: Mapping[str, Any],
    weights_by_ref: Mapping[str, Mapping[str, list[Any]]],
) -> None:
    """Add one `nn_blocks[name]` entry to `model` as a symbolic block.

    A block's `scale` is the parameter `{name}_scale` (as in mxlweb-core),
    added if the file doesn't hold it. An untrained block is initialised from
    its seed with MxlPy's own Glorot initialisation, not mxlweb-core's.
    """
    layers: list[NNLayer] = []
    for layer in block["layers"]:
        if layer.get("type") != "dense":
            msg = f"nn_block {name!r}: only dense layers are supported"
            raise SerializationError(msg)
        act = layer.get("activation")
        if act is None:
            layers.append(NNLayer(int(layer["width"])))
            continue
        expr = mathml_to_sympy(mml.node_from_dict(act["expression"]))
        known = (
            act["name"] in ACTIVATIONS
            and sympy.simplify(expr - ACTIVATIONS[act["name"]](_X)) == 0
        )
        layers.append(
            NNLayer(int(layer["width"]), act["name"], None if known else expr)
        )
    n_hidden = int(block.get("hidden_states", 0))
    inputs = list(block["inputs"])
    targets = list(block["targets"])
    widths = [layer.width for layer in layers]
    out = n_hidden if n_hidden > 0 else len(targets)
    if widths[-1] != out:
        msg = f"nn_block {name!r}: the last layer needs {out} units, not {widths[-1]}"
        raise SerializationError(msg)
    scales = np.asarray(block.get("input_scales", [1.0] * len(inputs)), dtype=float)
    hs = init_hidden_states(
        n_hidden=n_hidden,
        layers=layers,
        inputs=inputs,
        input_scales=_array(
            scales, (len(inputs),), origin=f"nn_block {name!r} input_scales"
        ),
        targets=targets,
        mechanism=_mechanism_from_node_dict(block["mechanism"], origin=name),
        init_scale=1e-2,
        seed=int(block["seed"]),
    )
    if block["trained"]:
        if (weights := weights_by_ref.get(block.get("weights_ref", ""))) is None:
            msg = f"nn_block {name!r}: no weights file supplied for weights_ref {block.get('weights_ref')!r}"
            raise SerializationError(msg)
        expected = {f"{k}{i + 1}" for i in range(len(layers)) for k in "wb"}
        if n_hidden > 0:
            expected |= {"W", "h0"}
        if set(weights) != expected:
            msg = (
                f"nn_block {name!r}: the weights file needs {sorted(expected)}, "
                f"not {sorted(weights)}"
            )
            raise SerializationError(msg)
        for i, w in enumerate(hs.weights):
            hs.weights[i] = _array(
                weights[f"w{i + 1}"], w.shape, origin=f"nn_block {name!r} w{i + 1}"
            )
            hs.biases[i] = _array(
                weights[f"b{i + 1}"],
                hs.biases[i].shape,
                origin=f"nn_block {name!r} b{i + 1}",
            )
        if n_hidden > 0:
            hs.readout = _array(
                weights["W"], hs.readout.shape, origin=f"nn_block {name!r} W"
            )
            hs.h0 = _array(weights["h0"], hs.h0.shape, origin=f"nn_block {name!r} h0")
    if "output_scales" in block:
        hs.output_scales = _array(
            block["output_scales"],
            (len(targets),),
            origin=f"nn_block {name!r} output_scales",
        )
    if "scale" in block:
        scale = f"{name}_scale"
        if scale not in model.get_parameter_names():
            model.add_parameter(scale, float(block["scale"]))
        hs.scale = scale
    try:
        model._add_block(name, hs)  # noqa: SLF001
    except (KeyError, ValueError) as exc:
        msg = f"nn_block {name!r}: {exc}"
        raise SerializationError(msg) from exc


def _add_terms_and_blocks(
    model: KineticModelBuilder | OdeModelBuilder,
    spec: Mapping[str, Any],
    weights_by_ref: Mapping[str, Mapping[str, list[Any]]],
    customs: Mapping[str, Any],
    arrays: Collection[str],
) -> None:
    for name, entry in spec.get("terms", {}).items():
        fn, args = _node_dict_to_fn(entry["fn"], arrays)
        model.add_term(
            name,
            fn,
            args=args,
            target=entry["target"],
            mechanism=_mechanism_from_node_dict(entry["mechanism"], origin=name),
            unit=_entry_unit(entry, customs, origin=name),
            **_names_from_dict(entry),
        )
    for name, block in spec.get("nn_blocks", {}).items():
        _block_from_mxl_json(model, name, block, weights_by_ref)


###############################################################################
# Load
###############################################################################


class _Names(TypedDict):
    display_name: str | None
    tex_name: str | None


def _slider_from_dict(entity: Mapping[str, Any]) -> Slider | None:
    return None if (s := entity.get("slider")) is None else Slider(**s)


def _names_from_dict(entity: Mapping[str, Any]) -> _Names:
    return {
        "display_name": entity.get("displayName"),
        "tex_name": entity.get("texName"),
    }


def _node_dict_to_fn(
    node_dict: dict[str, Any],
    arrays: Collection[str] = (),
) -> tuple[Callable[..., float], list[str]]:
    expr = mathml_to_sympy(mml.node_from_dict(node_dict))
    args = sorted(str(s) for s in expr.free_symbols)
    elementwise = not set(args).isdisjoint(arrays)
    return compile_fn(expr, args, name=_FN_NAME, elementwise=elementwise), args


def _array_names(
    spec: Mapping[str, Any], data_file: Mapping[str, Any] | None
) -> set[str]:
    """Data and array-valued derived quantities, whose readers need numpy code (ADR 0015).

    Raises
    ------
    SerializationError
        If an array breaks the rules of ADR 0015, e.g. a reaction rate,
        stoichiometry or readout that is an array.

    """
    if not data_file:
        return set()
    data = {k: np.asarray(v["values"], dtype=float) for k, v in data_file.items()}

    def expr(node: dict[str, Any]) -> sympy.Expr:
        return mathml_to_sympy(mml.node_from_dict(node))

    derived = [(k, expr(v["fn"])) for k, v in spec["derived"].items()]
    # must be numbers; named so they can't collide with model names
    scalars = [
        (f"{kind} {k!r}", expr(v["fn"]))
        for kind in ("variables", "reactions", "readouts")
        for k, v in spec.get(kind, {}).items()
        if "fn" in v
    ] + [
        (f"stoichiometry of {k!r} on {var!r}", expr(node))
        for k, v in spec.get("reactions", {}).items()
        for var, node in v.get("stoichiometry", {}).items()
    ]
    try:
        arrays = array_lengths(derived + scalars, data, {k for k, _ in scalars})
    except ValueError as e:
        raise SerializationError(str(e)) from e
    return {k for k, v in data.items() if v.ndim == 1} | set(arrays)


def _entry_unit(
    entry: Mapping[str, Any],
    customs: Mapping[str, Any],
    *,
    origin: str,
) -> sympy.Expr | None:
    """Revive an entity's optional `unit`, resolving custom kinds from `model.units`."""
    if (unit := entry.get("unit")) is None:
        return None
    try:
        return unit_from_json(unit, customs)
    except ValueError as exc:
        msg = f"unit of {origin!r}: {exc}"
        raise SerializationError(msg) from exc


def _node_dict_to_value(node_dict: dict[str, Any]) -> float | InitialAssignment:
    if node_dict["type"] == "Num":
        return float(node_dict["value"])
    fn, args = _node_dict_to_fn(node_dict)
    return InitialAssignment(fn=fn, args=args)


def _node_dict_to_stoich(node_dict: dict[str, Any]) -> float | Derived:
    expr = mathml_to_sympy(mml.node_from_dict(node_dict))
    if expr.is_number:
        return float(expr)
    fn, args = _node_dict_to_fn(node_dict)
    return Derived(fn=fn, args=args)


###############################################################################
# Load — OdeModelBuilder
###############################################################################


def _add_data(
    model: KineticModelBuilder | OdeModelBuilder,
    data: Mapping[str, Any],
    data_file: Mapping[str, Any] | None,
) -> None:
    """Attach the data sets of `data_file`, the sidecar `data["data_ref"]` names."""
    if (data_ref := data.get("data_ref")) is None:
        return
    if data_file is None:
        msg = f"no data file supplied for data_ref {data_ref!r}"
        raise SerializationError(msg)
    for name, entry in data_file.items():
        model.add_data(name, pd.Series(entry["values"], dtype=float))


def ode_model_from_dict(
    data: Mapping[str, Any],
    *,
    weights_by_ref: Mapping[str, Mapping[str, list[Any]]] | None = None,
    data_file: Mapping[str, Any] | None = None,
) -> OdeModelBuilder:
    """Reconstruct an `OdeModelBuilder` model from its ``.mxl.json`` dict representation.

    Ode-model counterpart of :func:`model_from_dict`.

    Parameters
    ----------
    data
        Mapping following the ``mxl-ode-model`` schema
    weights_by_ref
        Every referenced NN block weights sidecar's already-parsed content,
        keyed by the relative filename its owning block's `weights_ref`
        names — see :func:`model_from_dict`'s identical parameter.

    Returns
    -------
    OdeModelBuilder
        The reconstructed model

    Raises
    ------
    SerializationError
        If a `nn_blocks` entry can't be read (e.g. non-dense layers, weights
        not matching its layers) or a weights file its `weights_ref` names
        is missing.

    """
    spec = data["model"]
    customs = spec.get("units", {})
    model = OdeModelBuilder()
    _add_data(model, data, data_file)
    arrays = _array_names(spec, data_file)

    for name, par in spec["parameters"].items():
        model.add_parameter(
            name,
            _node_dict_to_value(par["value"]),
            unit=_entry_unit(par, customs, origin=name),
            **_names_from_dict(par),
            slider=_slider_from_dict(par),
        )
    for name, var in spec["variables"].items():
        fn, args = _node_dict_to_fn(var["fn"], arrays)
        model.add_diff_eq(
            name,
            initial_value=_node_dict_to_value(var["value"]),
            fn=fn,
            args=args,
            unit=_entry_unit(var, customs, origin=name),
            **_names_from_dict(var),
            slider=_slider_from_dict(var),
        )
    for name, der in spec["derived"].items():
        fn, args = _node_dict_to_fn(der["fn"], arrays)
        model.add_derived(
            name,
            fn,
            args=args,
            unit=_entry_unit(der, customs, origin=name),
            **_names_from_dict(der),
        )
    for name, rdt in spec["readouts"].items():
        fn, args = _node_dict_to_fn(rdt["fn"], arrays)
        model.add_readout(
            name,
            fn,
            args=args,
            unit=_entry_unit(rdt, customs, origin=name),
            **_names_from_dict(rdt),
        )
    _add_terms_and_blocks(model, spec, weights_by_ref or {}, customs, arrays)

    return model


###############################################################################
# Load — SteadyStateModelBuilder
###############################################################################


def steady_state_model_from_dict(data: Mapping[str, Any]) -> SteadyStateModelBuilder:
    """Reconstruct a `SteadyStateModelBuilder` model from its ``.mxl.json`` dict representation.

    Steady-state counterpart of :func:`model_from_dict` / :func:`ode_model_from_dict`.
    Unlike those, there is no `weights_by_ref` parameter — the schema
    forbids an `nn_blocks` section for this model kind entirely (matching
    `SteadyStateModelBuilder`'s lack of a surrogates concept), so no
    weights sidecar can ever be referenced.

    Parameters
    ----------
    data
        Mapping following the ``mxl-steady-state-model`` schema

    Returns
    -------
    SteadyStateModelBuilder
        The reconstructed model

    """
    spec = data["model"]
    customs = spec.get("units", {})
    model = SteadyStateModelBuilder()

    for name, par in spec["parameters"].items():
        model.add_parameter(
            name,
            _node_dict_to_value(par["value"]),
            unit=_entry_unit(par, customs, origin=name),
            **_names_from_dict(par),
            slider=_slider_from_dict(par),
        )
    for name, der in spec["derived"].items():
        fn, args = _node_dict_to_fn(der["fn"])
        model.add_derived(
            name,
            fn,
            args=args,
            unit=_entry_unit(der, customs, origin=name),
            **_names_from_dict(der),
        )

    return model


def model_from_dict(
    data: Mapping[str, Any],
    *,
    weights_by_ref: Mapping[str, Mapping[str, list[Any]]] | None = None,
    data_file: Mapping[str, Any] | None = None,
) -> KineticModelBuilder:
    """Reconstruct a model from its ``.mxl.json`` dict representation.

    Parameters
    ----------
    data
        Mapping following the ``mxl-model`` schema
    weights_by_ref
        Every referenced NN block weights sidecar's already-parsed content,
        keyed by the relative filename its owning block's `weights_ref`
        names. `load` resolves and reads these from disk itself; this
        function never touches the filesystem, so a caller with a document
        obtained some other way (e.g. fetched over the network) must
        resolve them independently.

    Returns
    -------
    Model
        The reconstructed model

    Raises
    ------
    SerializationError
        If a `nn_blocks` entry can't be read (e.g. non-dense layers, weights
        not matching its layers) or a weights file its `weights_ref` names
        is missing.

    """
    spec = data["model"]
    customs = spec.get("units", {})
    model = KineticModelBuilder()
    _add_data(model, data, data_file)
    arrays = _array_names(spec, data_file)

    for name, par in spec["parameters"].items():
        model.add_parameter(
            name,
            _node_dict_to_value(par["value"]),
            unit=_entry_unit(par, customs, origin=name),
            **_names_from_dict(par),
            slider=_slider_from_dict(par),
        )
    for name, var in spec["variables"].items():
        model.add_variable(
            name,
            _node_dict_to_value(var["value"]),
            unit=_entry_unit(var, customs, origin=name),
            **_names_from_dict(var),
            slider=_slider_from_dict(var),
        )
    for name, der in spec["derived"].items():
        fn, args = _node_dict_to_fn(der["fn"], arrays)
        model.add_derived(
            name,
            fn,
            args=args,
            unit=_entry_unit(der, customs, origin=name),
            **_names_from_dict(der),
        )
    for name, rxn in spec["reactions"].items():
        fn, args = _node_dict_to_fn(rxn["fn"], arrays)
        stoichiometry = {
            var: _node_dict_to_stoich(node)
            for var, node in rxn["stoichiometry"].items()
        }
        model.add_reaction(
            name,
            fn,
            args=args,
            stoichiometry=stoichiometry,
            unit=_entry_unit(rxn, customs, origin=name),
            **_names_from_dict(rxn),
        )
    for name, rdt in spec["readouts"].items():
        fn, args = _node_dict_to_fn(rdt["fn"], arrays)
        model.add_readout(
            name,
            fn,
            args=args,
            unit=_entry_unit(rdt, customs, origin=name),
            **_names_from_dict(rdt),
        )
    _add_terms_and_blocks(model, spec, weights_by_ref or {}, customs, arrays)

    return model


def load(
    path: str | Path,
) -> KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder:
    """Load a model from the native ``.mxl.json`` format.

    Dispatches on the document's `kind` field (`"kinetic"`, the default
    for documents written before this field existed, `"ode"`, or
    `"steady-state"`) to :func:`model_from_dict`, :func:`ode_model_from_dict`
    or :func:`steady_state_model_from_dict` respectively.

    Every `nn_blocks[id].weights_ref` referenced in the document is
    resolved relative to `path`'s own directory and read from disk
    automatically (mirrors `save`'s multi-file output).

    Parameters
    ----------
    path
        Source file path

    Returns
    -------
    Model
        The reconstructed model

    Raises
    ------
    SerializationError
        If the document has a `nn_blocks` entry that isn't representable
        as a live MxlPy surrogate, or its `weights_ref` file is missing.

    """
    path = Path(path)
    data = json.loads(path.read_text())
    weights_by_ref: dict[str, Any] = {}
    for name, block in data.get("model", {}).get("nn_blocks", {}).items():
        if not block.get("trained") or "weights_ref" not in block:
            continue
        weights_ref = block["weights_ref"]
        weights_path = path.parent / weights_ref
        try:
            weights_by_ref[weights_ref] = json.loads(weights_path.read_text())
        except FileNotFoundError as exc:
            msg = (
                f"nn_block {name!r}: weights file {str(weights_path)!r} "
                f"(from weights_ref {weights_ref!r}) does not exist"
            )
            raise SerializationError(msg) from exc
    data_file = None
    if (data_ref := data.get("data_ref")) is not None:
        data_path = path.parent / data_ref
        try:
            data_file = json.loads(data_path.read_text())
        except FileNotFoundError as exc:
            msg = (
                f"data file {str(data_path)!r} (from data_ref {data_ref!r}) "
                "does not exist"
            )
            raise SerializationError(msg) from exc
    if data.get("kind") == "ode":
        return ode_model_from_dict(
            data, weights_by_ref=weights_by_ref, data_file=data_file
        )
    if data.get("kind") == "steady-state":
        return steady_state_model_from_dict(data)
    return model_from_dict(data, weights_by_ref=weights_by_ref, data_file=data_file)
