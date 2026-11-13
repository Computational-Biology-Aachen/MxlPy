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
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import pandas as pd
import sympy

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
from mxlpy.surrogates.abstract import mxl_json_mechanism_additive
from mxlpy.types import Derived, InitialAssignment, SerializationError, Slider
from mxlpy.units import unit_from_json, unit_to_json

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    from mxlpy.surrogates.abstract import SurrogateProtocol
    from mxlpy.surrogates.abstract_ode import OdeSurrogateProtocol
    from mxlpy.types import DiffEq, Parameter, RateFn, Reaction, Readout, Variable

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

SPEC_VERSION = "1.1"
# Written instead of SPEC_VERSION by a model with data (`data_ref`).
DATA_SPEC_VERSION = "1.2"
# Written by a model whose arrays depend on more than data (ADR 0015).
ARRAY_SPEC_VERSION = "1.3"
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
    entity: Variable | Parameter | Derived | Readout | Reaction | DiffEq,
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
        if spec["trained"]:
            weights_ref = f"{name}.weights.json"
            spec["weights_ref"] = weights_ref
            weights_files[weights_ref] = export.weights
        nn_blocks[name] = spec
    return nn_blocks, weights_files


def mxl_json_weights_files(
    model: KineticModelBuilder,
) -> dict[str, dict[str, list[Any]]]:
    """Every trained, exportable surrogate's weights sidecar content, keyed by the relative filename its `nn_blocks[id].weights_ref` names.

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
    """`spec_version` and, for a model with data, `data_ref`."""
    if not (data := model.get_raw_data()):
        return {"spec_version": SPEC_VERSION}
    try:
        dynamic = _dynamic_arrays(section, data)
    except ValueError as e:
        raise SerializationError(str(e)) from e
    version = ARRAY_SPEC_VERSION if dynamic else DATA_SPEC_VERSION
    return {"spec_version": version, "data_ref": data_ref}


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
        if spec["trained"]:
            weights_ref = f"{name}.weights.json"
            spec["weights_ref"] = weights_ref
            weights_files[weights_ref] = export.weights
        nn_blocks[name] = spec
    return nn_blocks, weights_files


def ode_mxl_json_weights_files(
    model: OdeModelBuilder,
) -> dict[str, dict[str, list[Any]]]:
    """Every trained, exportable ode surrogate's weights sidecar content, keyed by its `weights_ref`.

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


def _validate_dense_nn_block(name: str, block: Mapping[str, Any]) -> None:
    """Structural checks shared by the kinetic and ode reconstruction paths.

    No backend's `*_surrogate_from_mxl_json` checks a layer's `type` — only
    `width`/`activation`/weights, since every layer any backend can
    currently produce is `"dense"` — so a non-`"dense"` layer (schema-valid,
    not yet representable in MxlPy) must be caught here instead, before
    delegating. Per-layer `activation` correctness is left to the backend
    (`mxlpy.surrogates._torch._revive_dense_sequential` and its keras/
    equinox counterparts), which already validates it structurally against
    `mxlpy.surrogates.abstract.ACTIVATION_BUILDERS`.
    """
    if any(layer.get("type") != "dense" for layer in block["layers"]):
        msg = f"nn_block {name!r}: only dense layers are supported"
        raise SerializationError(msg)
    if not block["trained"]:
        msg = (
            f"nn_block {name!r}: untrained blocks (no weights_ref) aren't "
            "supported yet — MxlPy has no from-seed Glorot initialization "
            "path for a loaded block"
        )
        raise SerializationError(msg)


def _surrogate_from_mxl_json(
    name: str,
    block: Mapping[str, Any],
    weights_by_ref: Mapping[str, Mapping[str, list[Any]]],
) -> SurrogateProtocol:
    """Reconstruct a live surrogate from one `nn_blocks[name]` entry.

    Only the exact shape :meth:`AbstractSurrogate.to_mxl_json` itself
    produces round-trips back into a real surrogate — additive mechanism,
    dense layers with a recognized per-layer activation. A block authored
    elsewhere (mxlweb, a different mechanism, a future non-dense layer
    type) isn't representable in MxlPy yet; raising here (rather than
    silently dropping the block) matches every other "can't faithfully
    represent this" boundary in this codebase — a model missing a block's
    dynamical contribution is worse than a load that refuses.
    """
    if block["mechanism"] != mxl_json_mechanism_additive():
        msg = (
            f"nn_block {name!r}: only the additive mechanism can be "
            "reconstructed into a live MxlPy surrogate"
        )
        raise SerializationError(msg)
    _validate_dense_nn_block(name, block)
    weights_ref = block["weights_ref"]
    weights = weights_by_ref.get(weights_ref)
    if weights is None:
        msg = f"nn_block {name!r}: no weights file supplied for weights_ref {weights_ref!r}"
        raise SerializationError(msg)

    try:
        # Deliberately lazy: torch is an optional extra, and this module
        # must stay importable (and usable for every model with no
        # nn_blocks, the common case) without it installed.
        from mxlpy.surrogates._torch import surrogate_from_mxl_json
    except ImportError as exc:
        msg = (
            f"nn_block {name!r}: reconstructing it requires torch "
            "(install mxlpy with the torch extra)"
        )
        raise SerializationError(msg) from exc
    try:
        return surrogate_from_mxl_json(name, block, weights)
    except SerializationError as exc:
        msg = f"nn_block {name!r}: {exc}"
        raise SerializationError(msg) from exc


###############################################################################
# Load — OdeModelBuilder
###############################################################################


def _ode_surrogate_from_mxl_json(
    name: str,
    block: Mapping[str, Any],
    weights_by_ref: Mapping[str, Mapping[str, list[Any]]],
) -> OdeSurrogateProtocol:
    """Reconstruct a live ode surrogate from one `nn_blocks[name]` entry.

    Ode-model counterpart of :func:`_surrogate_from_mxl_json` — same
    additive-mechanism-only constraint (an ode surrogate is always summed
    into its `targets`' dx/dt, exactly like a kinetic surrogate is always
    summed via stoichiometry; see `mxlpy.surrogates.abstract_ode`'s module
    docstring for why there's no selectable composition operator here).
    """
    if block["mechanism"] != mxl_json_mechanism_additive():
        msg = (
            f"nn_block {name!r}: only the additive mechanism can be "
            "reconstructed into a live MxlPy surrogate"
        )
        raise SerializationError(msg)
    _validate_dense_nn_block(name, block)
    weights_ref = block["weights_ref"]
    weights = weights_by_ref.get(weights_ref)
    if weights is None:
        msg = f"nn_block {name!r}: no weights file supplied for weights_ref {weights_ref!r}"
        raise SerializationError(msg)

    try:
        # Deliberately lazy: torch is an optional extra, see
        # `_surrogate_from_mxl_json`'s identical import.
        from mxlpy.surrogates._torch import (
            ode_surrogate_from_mxl_json,
        )
    except ImportError as exc:
        msg = (
            f"nn_block {name!r}: reconstructing it requires torch "
            "(install mxlpy with the torch extra)"
        )
        raise SerializationError(msg) from exc
    try:
        return ode_surrogate_from_mxl_json(name, block, weights)
    except SerializationError as exc:
        msg = f"nn_block {name!r}: {exc}"
        raise SerializationError(msg) from exc


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
        If the document has a `nn_blocks` entry that isn't representable
        as a live MxlPy surrogate (see `_ode_surrogate_from_mxl_json`),
        or is missing a weights file its `weights_ref` names.

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
    for name, block in spec.get("nn_blocks", {}).items():
        surrogate = _ode_surrogate_from_mxl_json(name, block, weights_by_ref or {})
        model.add_surrogate(name, surrogate)

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
        If the document has a `nn_blocks` entry that isn't representable
        as a live MxlPy surrogate (see `_surrogate_from_mxl_json`), or is
        missing a weights file its `weights_ref` names.

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
    for name, block in spec.get("nn_blocks", {}).items():
        surrogate = _surrogate_from_mxl_json(name, block, weights_by_ref or {})
        model.add_surrogate(name, surrogate)

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
