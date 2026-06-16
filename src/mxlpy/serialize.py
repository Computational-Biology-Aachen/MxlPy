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

import json
import linecache
import math
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.special  # bound as `scipy`, used by generated rate fns

from mxlpy.meta import _mathml as mml
from mxlpy.meta.source_tools import fn_to_sympy_expr
from mxlpy.meta.sympy_tools import (
    list_of_symbols,
    mathml_to_sympy,
    sympy_to_mathml,
    sympy_to_python_fn,
)
from mxlpy.model import Model
from mxlpy.types import Derived, InitialAssignment, SerializationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import sympy

    from mxlpy.types import RateFn

__all__ = [
    "SCHEMA_URL",
    "SPEC_VERSION",
    "load",
    "model_from_dict",
    "model_to_dict",
    "save",
]

SPEC_VERSION = "1.0"
# The schema is maintained in the shared, language-agnostic `mxl-schemas` repo
# so it can be consumed by both mxlpy and mxlweb. It is intentionally not
# vendored into this package. The URL is pinned to the major spec version, so a
# file's $schema always matches the spec_version it was written with.
_SCHEMA_MAJOR = SPEC_VERSION.split(".", 1)[0]
SCHEMA_URL = (
    "https://raw.githubusercontent.com/Computational-Biology-Aachen/"
    f"mxl-schemas/main/v{_SCHEMA_MAJOR}/kinetic-model.schema.json"
)

_FN_NAME = "_mxl_fn"


###############################################################################
# Save
###############################################################################


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


def model_to_dict(
    model: Model,
    *,
    model_id: str,
    description: str = "",
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
        If the model contains surrogates or a rate function that cannot be
        converted into an expression.

    """
    if model._surrogates:  # noqa: SLF001
        names = ", ".join(sorted(model._surrogates))  # noqa: SLF001
        msg = f"surrogates are not supported: {names}"
        raise SerializationError(msg)

    variables = {
        name: {"value": _value_to_node_dict(var.initial_value, origin=name)}
        for name, var in model.get_raw_variables().items()
    }
    parameters = {
        name: {"value": _value_to_node_dict(par.value, origin=name)}
        for name, par in model.get_raw_parameters().items()
    }
    reactions = {
        name: {
            "fn": _fn_to_node_dict(rxn.fn, origin=name, args=rxn.args),
            "stoichiometry": {
                var: _stoich_to_node_dict(stoich, origin=f"{name}:{var}")
                for var, stoich in rxn.stoichiometry.items()
            },
        }
        for name, rxn in model.get_raw_reactions().items()
    }
    derived = {
        name: {"fn": _fn_to_node_dict(der.fn, origin=name, args=der.args)}
        for name, der in model.get_raw_derived().items()
    }
    readouts = {
        name: {"fn": _fn_to_node_dict(rdt.fn, origin=name, args=rdt.args)}
        for name, rdt in model.get_raw_readouts().items()
    }

    return {
        "$schema": SCHEMA_URL,
        "spec_version": SPEC_VERSION,
        "model_id": model_id,
        "description": description,
        "model": {
            "variables": variables,
            "parameters": parameters,
            "reactions": reactions,
            "derived": derived,
            "readouts": readouts,
        },
    }


def save(
    model: Model,
    path: str | Path,
    *,
    model_id: str | None = None,
    description: str = "",
) -> None:
    """Save a model to the native ``.mxl.json`` format.

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
        If the model contains surrogates or a rate function that cannot be
        converted into an expression.

    """
    path = Path(path)
    if model_id is None:
        model_id = path.name.removesuffix(".json").removesuffix(".mxl")
    data = model_to_dict(model, model_id=model_id, description=description)
    path.write_text(json.dumps(data, indent=2) + "\n")


###############################################################################
# Load
###############################################################################


def _compile_fn(expr: sympy.Expr, args: list[str]) -> Callable[..., float]:
    src = sympy_to_python_fn(fn_name=_FN_NAME, args=args, expr=expr)
    # Register the source in linecache under a unique filename so that
    # inspect.getsource (and thus fn_to_sympy_expr / codegen) can recover it,
    # which keeps loaded models re-serialisable and introspectable.
    filename = f"<mxlpy-generated-{uuid.uuid4().hex}>"
    linecache.cache[filename] = (
        len(src),
        None,
        src.splitlines(keepends=True),
        filename,
    )
    namespace: dict[str, Any] = {
        "math": math,
        "numpy": np,
        "np": np,
        "scipy": scipy,
    }
    exec(compile(src, filename, "exec"), namespace)  # noqa: S102
    return namespace[_FN_NAME]


def _node_dict_to_fn(
    node_dict: dict[str, Any],
) -> tuple[Callable[..., float], list[str]]:
    expr = mathml_to_sympy(mml.node_from_dict(node_dict))
    args = sorted(str(s) for s in expr.free_symbols)
    return _compile_fn(expr, args), args


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


def model_from_dict(data: Mapping[str, Any]) -> Model:
    """Reconstruct a model from its ``.mxl.json`` dict representation.

    Parameters
    ----------
    data
        Mapping following the ``mxl-model`` schema

    Returns
    -------
    Model
        The reconstructed model

    """
    spec = data["model"]
    model = Model()

    for name, par in spec["parameters"].items():
        model.add_parameter(name, _node_dict_to_value(par["value"]))
    for name, var in spec["variables"].items():
        model.add_variable(name, _node_dict_to_value(var["value"]))
    for name, der in spec["derived"].items():
        fn, args = _node_dict_to_fn(der["fn"])
        model.add_derived(name, fn, args=args)
    for name, rxn in spec["reactions"].items():
        fn, args = _node_dict_to_fn(rxn["fn"])
        stoichiometry = {
            var: _node_dict_to_stoich(node)
            for var, node in rxn["stoichiometry"].items()
        }
        model.add_reaction(name, fn, args=args, stoichiometry=stoichiometry)
    for name, rdt in spec["readouts"].items():
        fn, args = _node_dict_to_fn(rdt["fn"])
        model.add_readout(name, fn, args=args)

    return model


def load(path: str | Path) -> Model:
    """Load a model from the native ``.mxl.json`` format.

    Parameters
    ----------
    path
        Source file path

    Returns
    -------
    Model
        The reconstructed model

    """
    data = json.loads(Path(path).read_text())
    return model_from_dict(data)
