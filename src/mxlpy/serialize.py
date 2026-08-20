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

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy.meta import _mathml as mml
from mxlpy.meta.source_tools import fn_to_sympy_expr
from mxlpy.meta.sympy_tools import (
    list_of_symbols,
    mathml_to_sympy,
    sympy_to_mathml,
    sympy_to_python_fn,
)
from mxlpy.surrogates.abstract import (
    nn_block_activation_softplus,
    nn_block_mechanism_additive,
)
from mxlpy.types import Derived, InitialAssignment, SerializationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import sympy

    from mxlpy.surrogates.abstract import SurrogateProtocol
    from mxlpy.types import RateFn

__all__ = [
    "SCHEMA_URL",
    "SPEC_VERSION",
    "load",
    "model_from_dict",
    "model_to_dict",
    "nn_block_weights_files",
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


def _unexportable_surrogates(model: KineticModelBuilder) -> list[str]:
    """Names of every attached surrogate whose `to_nn_block_export()` returns `None`."""
    return sorted(
        name
        for name, surrogate in model._surrogates.items()  # noqa: SLF001
        if surrogate.to_nn_block_export() is None
    )


def _nn_blocks_and_weights(
    model: KineticModelBuilder,
) -> tuple[dict[str, Any], dict[str, dict[str, list[Any]]]]:
    """The `nn_blocks` section and, separately, every trained block's weights sidecar content.

    Weight/bias values are never inlined into `nn_blocks` itself (mirrors
    mxl-schemas' own split from `parameters`) — `spec["weights_ref"]` names
    a `<block>.weights.json` sidecar file whose content is this function's
    second return value, keyed by the same relative filename `save()`
    writes it under.
    """
    nn_blocks: dict[str, Any] = {}
    weights_files: dict[str, dict[str, list[Any]]] = {}
    for name, surrogate in model._surrogates.items():  # noqa: SLF001
        export = surrogate.to_nn_block_export()
        if export is None:
            continue
        spec = dict(export.spec)
        if spec["trained"]:
            weights_ref = f"{name}.weights.json"
            spec["weights_ref"] = weights_ref
            weights_files[weights_ref] = export.weights
        nn_blocks[name] = spec
    return nn_blocks, weights_files


def nn_block_weights_files(
    model: KineticModelBuilder,
) -> dict[str, dict[str, list[Any]]]:
    """Every trained, exportable surrogate's weights sidecar content, keyed by the relative filename its `nn_blocks[id].weights_ref` names.

    Pairs with :func:`model_to_dict`'s output (which references these same
    filenames but never inlines their content) to produce the full
    multi-file export; :func:`save` calls this and writes each file
    alongside the main one. A model with no NN blocks (the common case)
    returns an empty dict.
    """
    _, weights_files = _nn_blocks_and_weights(model)
    return weights_files


def model_to_dict(
    model: KineticModelBuilder,
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
        If the model contains a surrogate that isn't representable as a
        mxl-schemas ``nn_blocks`` entry (see
        :meth:`mxlpy.surrogates.abstract.AbstractSurrogate.to_nn_block_export`),
        or a rate function that cannot be converted into an expression.

    """
    if unexportable := _unexportable_surrogates(model):
        names = ", ".join(unexportable)
        msg = f"surrogates are not representable as nn_blocks: {names}"
        raise SerializationError(msg)
    nn_blocks, _ = _nn_blocks_and_weights(model)

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

    model_section: dict[str, Any] = {
        "variables": variables,
        "parameters": parameters,
        "reactions": reactions,
        "derived": derived,
        "readouts": readouts,
    }
    if nn_blocks:
        model_section["nn_blocks"] = nn_blocks

    return {
        "$schema": SCHEMA_URL,
        "spec_version": SPEC_VERSION,
        "model_id": model_id,
        "description": description,
        "model": model_section,
    }


def save(
    model: KineticModelBuilder,
    path: str | Path,
    *,
    model_id: str | None = None,
    description: str = "",
) -> None:
    """Save a model to the native ``.mxl.json`` format.

    Every trained, exportable NN block's weights are written alongside the
    main file as `<block>.weights.json`, in the same directory `path`
    itself lands in (`nn_blocks[id].weights_ref` names it as a path
    relative to the main file, per mxl-schemas).

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
    data = model_to_dict(model, model_id=model_id, description=description)
    path.write_text(json.dumps(data, indent=2) + "\n")
    for weights_ref, weights in nn_block_weights_files(model).items():
        (path.parent / weights_ref).write_text(json.dumps(weights, indent=2) + "\n")


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


def _surrogate_from_nn_block(
    name: str,
    block: Mapping[str, Any],
    weights_by_ref: Mapping[str, Mapping[str, list[Any]]],
) -> SurrogateProtocol:
    """Reconstruct a live surrogate from one `nn_blocks[name]` entry.

    Only the exact shape :meth:`AbstractSurrogate.to_nn_block_export`
    itself produces round-trips back into a real surrogate — additive
    mechanism, softplus-activated dense layers, every layer `type: dense`.
    A block authored elsewhere (mxlweb, a different mechanism, a future
    non-dense layer type) isn't representable in MxlPy yet; raising here
    (rather than silently dropping the block) matches every other "can't
    faithfully represent this" boundary in this codebase — a model missing
    a block's dynamical contribution is worse than a load that refuses.
    """
    if block["mechanism"] != nn_block_mechanism_additive():
        msg = (
            f"nn_block {name!r}: only the additive mechanism can be "
            "reconstructed into a live MxlPy surrogate"
        )
        raise SerializationError(msg)
    activation = block["activation"]
    if activation.get("name") != "softplus" or activation.get(
        "expression"
    ) != nn_block_activation_softplus():
        msg = f"nn_block {name!r}: only the softplus activation is supported"
        raise SerializationError(msg)
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
    weights_ref = block["weights_ref"]
    weights = weights_by_ref.get(weights_ref)
    if weights is None:
        msg = f"nn_block {name!r}: no weights file supplied for weights_ref {weights_ref!r}"
        raise SerializationError(msg)

    try:
        # Deliberately lazy: torch is an optional extra, and this module
        # must stay importable (and usable for every model with no
        # nn_blocks, the common case) without it installed.
        from mxlpy.surrogates._torch import surrogate_from_nn_block  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            f"nn_block {name!r}: reconstructing it requires torch "
            "(install mxlpy with the torch extra)"
        )
        raise SerializationError(msg) from exc
    return surrogate_from_nn_block(name, block, weights)


def model_from_dict(
    data: Mapping[str, Any],
    *,
    weights_by_ref: Mapping[str, Mapping[str, list[Any]]] | None = None,
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
        as a live MxlPy surrogate (see `_surrogate_from_nn_block`), or is
        missing a weights file its `weights_ref` names.

    """
    spec = data["model"]
    model = KineticModelBuilder()

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
    for name, block in spec.get("nn_blocks", {}).items():
        surrogate = _surrogate_from_nn_block(name, block, weights_by_ref or {})
        model.add_surrogate(name, surrogate)

    return model


def load(path: str | Path) -> KineticModelBuilder:
    """Load a model from the native ``.mxl.json`` format.

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
    return model_from_dict(data, weights_by_ref=weights_by_ref)
