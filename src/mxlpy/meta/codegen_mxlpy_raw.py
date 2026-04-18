"""Generate mxlpy code from a model."""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from typing import TYPE_CHECKING

try:
    import dill as _dill  # type: ignore[import-untyped]
except ImportError:
    _dill = None  # type: ignore[assignment]

from mxlpy.types import Derived, InitialAssignment

if TYPE_CHECKING:
    from collections.abc import Callable

    from mxlpy.model import Model

__all__ = [
    "generate_mxlpy_code_raw",
]

_LOGGER = logging.getLogger()


def _fn_to_source(fn: Callable, fn_name: str, args: list[str]) -> str:
    """Return a clean function definition string with renamed parameters.

    Strips docstrings, renames positional parameters to *args*, and renames
    the function to *fn_name*.  Lambdas are converted to a proper ``def``.
    """
    try:
        raw = inspect.getsource(fn)
    except OSError:
        if _dill is None:
            msg = f"Cannot retrieve source for '{fn_name}': dill not installed"
            raise ValueError(msg) from None
        try:
            raw = _dill.source.getsource(fn)
        except OSError as exc:
            msg = f"Cannot retrieve source for '{fn_name}'"
            raise ValueError(msg) from exc

    source = textwrap.dedent(raw)

    if fn.__name__ == "<lambda>":
        return _lambda_to_def(source, fn_name, args)

    return source


def _lambda_to_def(source: str, fn_name: str, args: list[str]) -> str:
    """Convert a lambda expression found in *source* to a named function."""
    tree = ast.parse(source)
    lambda_node = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.Lambda)),
        None,
    )
    if lambda_node is None:
        msg = f"Could not find lambda in source for '{fn_name}'"
        raise ValueError(msg)

    # orig_params = [arg.arg for arg in lambda_node.args.args]
    # rename_map = dict(zip(orig_params, args, strict=True))
    # _NameRenamer(rename_map).visit(lambda_node)

    fn_args = ast.arguments(
        posonlyargs=[],
        args=[
            ast.arg(arg=a, annotation=ast.Name(id="float", ctx=ast.Load()))
            for a in args
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    fn_def = ast.FunctionDef(
        name=fn_name,
        args=fn_args,
        body=[ast.Return(value=lambda_node.body)],
        decorator_list=[],
        returns=ast.Name(id="float", ctx=ast.Load()),
        type_params=[],
    )
    ast.fix_missing_locations(fn_def)
    return ast.unparse(fn_def)


def generate_mxlpy_code_raw(model: Model, imports: list[str] | None = None) -> str:
    """Generate MxlPy source code without a symbolic representation round-trip.

    Parameters
    ----------
    model
        Model
    imports
        Optional list of import statements to include at the top of the file

    Returns
    -------
    str
        Python source code that constructs the model using MxlPy API

    """
    imports = [] if imports is None else imports
    # Maps fn_name -> source string (last writer wins for shared functions)
    functions: dict[str, str] = {}

    # Variables
    variable_source = []
    for k, variable in model.get_raw_variables().items():
        val = variable.initial_value
        if isinstance(val, InitialAssignment):
            fn_name = f"init_{val.fn.__name__}"
            functions[fn_name] = _fn_to_source(val.fn, fn_name, val.args)
            variable_source.append(
                f"""        .add_variable(
            {k!r},
            initial_value=InitialAssignment(fn={fn_name}, args={val.args!r}),
        )"""
            )
        else:
            variable_source.append(
                f"        .add_variable({k!r}, initial_value={val!r})"
            )

    # Parameters
    parameter_source = []
    for k, parameter in model.get_raw_parameters().items():
        val = parameter.value
        if isinstance(val, InitialAssignment):
            fn_name = f"init_{val.fn.__name__}"
            functions[fn_name] = _fn_to_source(val.fn, fn_name, val.args)
            parameter_source.append(
                f"""        .add_parameter(
            {k!r},
            value=InitialAssignment(fn={fn_name}, args={val.args!r}),
        )"""
            )
        else:
            parameter_source.append(f"        .add_parameter({k!r}, value={val!r})")

    # Derived
    derived_source = []
    for k, der in model.get_raw_derived().items():
        fn_name = der.fn.__name__
        functions[fn_name] = _fn_to_source(der.fn, fn_name, der.args)
        derived_source.append(
            f"""        .add_derived(
                {k!r},
                fn={fn_name},
                args={der.args!r},
            )"""
        )

    # Reactions
    reactions_source = []
    for k, rxn in model.get_raw_reactions().items():
        fn_name = rxn.fn.__name__
        functions[fn_name] = _fn_to_source(rxn.fn, fn_name, rxn.args)

        stoichiometry: list[str] = []
        for var, stoich in rxn.stoichiometry.items():
            if isinstance(stoich, Derived):
                stoich_fn_name = stoich.fn.__name__
                functions[stoich_fn_name] = _fn_to_source(
                    stoich.fn, stoich_fn_name, stoich.args
                )
                stoichiometry.append(
                    f'"{var}": Derived(fn={stoich_fn_name}, args={stoich.args!r})'
                )
            else:
                stoichiometry.append(f'"{var}": {stoich!r}')

        reactions_source.append(
            f"""        .add_reaction(
                "{k}",
                fn={fn_name},
                args={rxn.args!r},
                stoichiometry={{{",".join(stoichiometry)}}},
            )"""
        )

    # Surrogates
    surrogate_imports: set[str] = set()
    surrogate_source = []
    for k, surrogate in model.get_raw_surrogates().items():
        stype = type(surrogate)
        module = stype.__module__

        if not module.startswith("mxlpy.surrogates."):
            _LOGGER.warning(
                "Cannot generate code for surrogate type '%s'", stype.__qualname__
            )
            continue

        # mxlpy.surrogates._qss -> 'qss', mxlpy.surrogates.abstract -> 'abstract'
        raw_submod = module.split(".")[-1]
        submod = raw_submod.lstrip("_")
        class_ref = f"{submod}.{stype.__name__}"
        surrogate_imports.add(f"from mxlpy.surrogates import {submod}")

        # Generate source for callable model (e.g. QSS)
        model_fn_name = None
        model_attr = getattr(surrogate, "model", None)
        if callable(model_attr):
            fn = model_attr
            fn_name = fn.__name__ if fn.__name__ != "<lambda>" else f"{k}_model"
            try:
                functions[fn_name] = _fn_to_source(fn, fn_name, surrogate.args)
                model_fn_name = fn_name
            except (ValueError, TypeError, OSError) as exc:
                _LOGGER.warning(
                    "Cannot extract source for surrogate '%s' model: %s", k, exc
                )

        # Build constructor kwargs
        ctor_parts: list[str] = []
        if model_fn_name is not None:
            ctor_parts.append(f"model={model_fn_name}")
        ctor_parts.append(f"args={surrogate.args!r}")
        ctor_parts.append(f"outputs={surrogate.outputs!r}")

        if surrogate.stoichiometries:
            stoich_entries: list[str] = []
            for rxn_name, rxn_stoich in surrogate.stoichiometries.items():
                rxn_parts: list[str] = []
                for species, factor in rxn_stoich.items():
                    if isinstance(factor, Derived):
                        sf_name = f"{k}_{rxn_name}_stoich_{factor.fn.__name__}"
                        functions[sf_name] = _fn_to_source(
                            factor.fn, sf_name, factor.args
                        )
                        rxn_parts.append(
                            f'"{species}": Derived(fn={sf_name}, args={factor.args!r})'
                        )
                    else:
                        rxn_parts.append(f'"{species}": {factor!r}')
                stoich_entries.append(f'"{rxn_name}": {{{",".join(rxn_parts)}}}')
            ctor_parts.append(f"stoichiometries={{{','.join(stoich_entries)}}}")

        ctor_str = ",\n                    ".join(ctor_parts)
        surrogate_source.append(
            f"""        .add_surrogate(
                {k!r},
                {class_ref}(
                    {ctor_str},
                ),
            )"""
        )

    functions_source = "\n\n".join(functions.values())
    source = [
        *imports,
        *sorted(surrogate_imports),
        "from mxlpy import Model, Derived, InitialAssignment\n",
        functions_source,
        "",
        "def create_model() -> Model:",
        "    return (",
        "        Model()",
    ]
    if variable_source:
        source.append("\n".join(variable_source))
    if parameter_source:
        source.append("\n".join(parameter_source))
    if derived_source:
        source.append("\n".join(derived_source))
    if reactions_source:
        source.append("\n".join(reactions_source))
    if surrogate_source:
        source.append("\n".join(surrogate_source))
    source.append("    )")
    return "\n".join(source)
