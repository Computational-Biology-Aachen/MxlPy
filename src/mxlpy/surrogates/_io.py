# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Backend-native save/load of single surrogates.

A saved surrogate is a directory holding ``surrogate.json`` (format version,
backend, args, outputs and stoichiometries/units or targets) next to one
file in the backend's own format (``model.pt``, ``model.keras``,
``model.eqx``, ``model.json`` or ``model.dill``).

This is independent of `mxlpy.save`'s ``.mxl.json`` ``nn_blocks``: that one
is the portable, cross-language format restricted to dense networks; this
one keeps any architecture the backend itself can persist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mxlpy.surrogates.abstract_ode import AbstractOdeSurrogate
from mxlpy.types import SerializationError
from mxlpy.units import unit_from_json, unit_to_json

if TYPE_CHECKING:
    from mxlpy.surrogates.abstract import AbstractSurrogate

__all__ = [
    "FORMAT_VERSION",
    "META_FILE",
    "Backend",
    "load",
    "read_meta",
    "surrogate_fields",
    "write_meta",
]

FORMAT_VERSION = 1
META_FILE = "surrogate.json"

type Backend = Literal["torch", "keras", "equinox", "poly", "qss"]
type Kind = Literal["kinetic", "ode"]


def write_meta(
    path: str | Path,
    surrogate: AbstractSurrogate | AbstractOdeSurrogate,
    *,
    backend: Backend,
) -> Path:
    """Create the surrogate directory and write its ``surrogate.json``.

    Overwrites an existing ``surrogate.json``.

    Returns
    -------
    Path
        The surrogate directory, for the caller to write its model file into.

    Raises
    ------
    SerializationError
        If a `Derived` stoichiometry can't be converted to an expression,
        or a unit isn't a unit expression.

    """
    from mxlpy.serialize import _stoich_to_node_dict  # noqa: PLC0415  # serialize -> builders -> surrogates cycle

    meta: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "backend": backend,
        "args": list(surrogate.args),
        "outputs": list(surrogate.outputs),
    }
    if isinstance(surrogate, AbstractOdeSurrogate):
        meta["kind"] = "ode"
        meta["targets"] = {k: list(v) for k, v in surrogate.targets.items()}
    else:
        meta["kind"] = "kinetic"
        meta["stoichiometries"] = {
            rxn: {
                var: _stoich_to_node_dict(
                    value, origin=f"stoichiometry of {var!r} on {rxn!r}"
                )
                for var, value in stoich.items()
            }
            for rxn, stoich in surrogate.stoichiometries.items()
        }
        customs: dict[str, dict[str, str]] = {}
        units: dict[str, Any] = {}
        for name, unit in surrogate.units.items():
            try:
                units[name] = unit_to_json(unit, customs)
            except ValueError as exc:
                msg = f"unit of {name!r}: {exc}"
                raise SerializationError(msg) from exc
        meta["units"] = units
        meta["unit_kinds"] = customs

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / META_FILE).write_text(json.dumps(meta, indent=2))
    return path


def _read_raw(path: str | Path) -> dict[str, Any]:
    file = Path(path) / META_FILE
    if not file.is_file():
        msg = f"{str(path)!r} is not a saved surrogate: no {META_FILE} found."
        raise FileNotFoundError(msg)
    meta = json.loads(file.read_text())
    if (version := meta.get("format_version")) != FORMAT_VERSION:
        msg = (
            f"Unsupported surrogate format_version {version!r} "
            f"(this mxlpy reads {FORMAT_VERSION})."
        )
        raise SerializationError(msg)
    return meta


def read_meta(path: str | Path, *, backend: Backend) -> dict[str, Any]:
    """Read ``surrogate.json`` and check it was written by ``backend``.

    Raises
    ------
    FileNotFoundError
        If ``path`` holds no ``surrogate.json``.
    SerializationError
        On an unsupported format version or a different backend.

    """
    meta = _read_raw(path)
    if meta["backend"] != backend:
        msg = (
            f"Surrogate at {str(path)!r} was saved with backend "
            f"{meta['backend']!r}, not {backend!r}. "
            "Use mxlpy.surrogates.load to dispatch automatically."
        )
        raise SerializationError(msg)
    return meta


def surrogate_fields(meta: dict[str, Any]) -> tuple[Kind, dict[str, Any]]:
    """The surrogate kind plus the constructor kwargs (all but ``model``) stored in ``meta``."""
    from mxlpy.serialize import _node_dict_to_stoich  # noqa: PLC0415  # serialize -> builders -> surrogates cycle

    fields: dict[str, Any] = {"args": meta["args"], "outputs": meta["outputs"]}
    kind: Kind = meta["kind"]
    if kind == "ode":
        fields["targets"] = meta["targets"]
        return kind, fields

    fields["stoichiometries"] = {
        rxn: {var: _node_dict_to_stoich(node) for var, node in stoich.items()}
        for rxn, stoich in meta["stoichiometries"].items()
    }
    customs = meta.get("unit_kinds", {})
    fields["units"] = {
        name: unit_from_json(unit, customs) for name, unit in meta["units"].items()
    }
    return kind, fields


def load(
    path: str | Path,
    *,
    model: Any = None,
    **backend_kwargs: Any,
) -> AbstractSurrogate | AbstractOdeSurrogate:
    """Load a surrogate saved with ``surrogate.save(path)``, whatever its backend.

    Reads the backend from ``surrogate.json`` and only imports that backend.

    Parameters
    ----------
    path
        Directory written by ``save``.
    model
        Template network with the saved architecture, into which the weights
        are loaded. Required for torch and equinox (their native formats
        only store weights), rejected for keras, poly and qss (whose files
        store the full model).
    backend_kwargs
        Forwarded to the backend's ``load``. Only keras takes any
        (``custom_objects``, ``compile``, ... of `keras.saving.load_model`).

    Returns
    -------
    AbstractSurrogate | AbstractOdeSurrogate
        The backend's ``Surrogate`` or ``OdeSurrogate``.

    Raises
    ------
    ValueError
        If ``model`` is missing for torch/equinox or given for any other backend.
    SerializationError
        On an unsupported format version or unknown backend.

    """
    backend = _read_raw(path)["backend"]
    if backend in ("torch", "equinox"):
        if model is None:
            msg = (
                f"Loading a {backend} surrogate needs `model=`: a freshly "
                "constructed network with the saved architecture, since "
                f"{backend}'s native format only stores the weights."
            )
            raise ValueError(msg)
        if backend == "torch":
            from mxlpy.surrogates import _torch  # noqa: PLC0415  # only import the saved backend

            return _torch.load(path, model=model, **backend_kwargs)
        from mxlpy.surrogates import _equinox  # noqa: PLC0415  # only import the saved backend

        return _equinox.load(path, model=model, **backend_kwargs)

    if model is not None:
        msg = f"{backend} surrogates store their full model, don't pass `model=`."
        raise ValueError(msg)
    if backend == "keras":
        from mxlpy.surrogates import _keras  # noqa: PLC0415  # only import the saved backend

        return _keras.load(path, **backend_kwargs)
    if backend == "poly":
        from mxlpy.surrogates import _poly  # noqa: PLC0415  # only import the saved backend

        return _poly.load(path, **backend_kwargs)
    if backend == "qss":
        from mxlpy.surrogates import _qss  # noqa: PLC0415  # only import the saved backend

        return _qss.load(path, **backend_kwargs)
    msg = f"Unknown surrogate backend {backend!r}."
    raise SerializationError(msg)
