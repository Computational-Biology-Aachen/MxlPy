"""String-based DSL for specifying reaction networks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import sympy

from mxlpy import fns
from mxlpy.fns import hill_1s, michaelis_menten_1s

if TYPE_CHECKING:
    from mxlpy.model import Model
    from mxlpy.types import RateFn

__all__ = ["from_dsl"]

_REVERSIBLE_SEP = re.compile(r"\s*<-->\s*")
_IRREVERSIBLE_SEP = re.compile(r"\s*-->\s*")
_SPECIES_TOKEN = re.compile(r"^\s*(\d+)?\s*\*?\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")


def _parse_side(side: str) -> dict[str, float]:
    """Parse one side of a reaction arrow into {variable: coefficient}."""
    side = side.strip()
    if not side:
        return {}
    result: dict[str, float] = {}
    for token in side.split("+"):
        m = _SPECIES_TOKEN.match(token)
        if m is None:
            msg = f"Cannot parse variable token: {token!r}"
            raise ValueError(msg)
        coeff = float(m.group(1)) if m.group(1) else 1.0
        name = m.group(2)
        result[name] = result.get(name, 0.0) + coeff
    return result


def _net_stoichiometry(
    reactants: dict[str, float], products: dict[str, float]
) -> dict[str, float]:
    names = set(reactants) | set(products)
    stoich: dict[str, float] = {}
    for n in names:
        coeff = products.get(n, 0.0) - reactants.get(n, 0.0)
        if coeff != 0.0:
            stoich[n] = coeff
    return stoich


def _consumed(reactants: dict[str, float], products: dict[str, float]) -> list[str]:
    """Species with negative net stoichiometry (net consumed)."""
    net = _net_stoichiometry(reactants, products)
    return [s for s, c in net.items() if c < 0]


def _resolve_rate(
    rate_str: str,
    reactants: dict[str, float],
    products: dict[str, float],
    all_names: set[str],
) -> tuple[RateFn, list[str]]:
    """Resolve a rate expression string to a (fn, args) pair."""
    rate_str = rate_str.strip()

    # --- mm(kcat, Km) shorthand ---
    m = re.fullmatch(r"mm\s*\(([^)]+)\)", rate_str)
    if m is not None:
        param_args = [a.strip() for a in m.group(1).split(",")]
        consumed = _consumed(reactants, products)
        if len(consumed) != 1:
            msg = (
                f"mm() requires exactly one consumed reactant, "
                f"got {consumed!r} from rate {rate_str!r}"
            )
            raise ValueError(msg)
        return michaelis_menten_1s, [consumed[0], *param_args]

    # --- hill(v, K, n) shorthand ---
    m = re.fullmatch(r"hill\s*\(([^)]+)\)", rate_str)
    if m is not None:
        param_args = [a.strip() for a in m.group(1).split(",")]
        consumed = _consumed(reactants, products)
        if len(consumed) != 1:
            msg = (
                f"hill() requires exactly one consumed reactant, "
                f"got {consumed!r} from rate {rate_str!r}"
            )
            raise ValueError(msg)
        return hill_1s, [consumed[0], *param_args]

    # --- ma(k) shorthand: mass-action over all consumed reactants ---
    m = re.fullmatch(r"ma\s*\(([^)]+)\)", rate_str)
    if m is not None:
        k_name = m.group(1).strip()
        consumed = _consumed(reactants, products)
        # Build expression k * s1 * s2 * ...
        sym_list = [sympy.Symbol(name) for name in [k_name, *consumed]]
        expr: sympy.Expr = sympy.Mul(*sym_list)
        arg_names = [k_name, *consumed]
        fn = sympy.lambdify([sympy.Symbol(a) for a in arg_names], expr, modules="math")
        return fn, arg_names

    # --- named fn call: fn_name(arg1, arg2, ...) ---
    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", rate_str)
    if m is not None:
        fn_name = m.group(1)
        fn_obj = getattr(fns, fn_name, None)
        if fn_obj is not None:
            arg_names = [a.strip() for a in m.group(2).split(",") if a.strip()]
            return fn_obj, arg_names

    # --- arbitrary sympy expression ---
    try:
        expr = sympy.sympify(rate_str)
    except sympy.SympifyError as e:
        msg = f"Cannot parse rate expression: {rate_str!r}"
        raise ValueError(msg) from e

    free = sorted(str(s) for s in expr.free_symbols)
    # validate all free symbols are known
    unknown = [s for s in free if s not in all_names]
    if unknown:
        msg = f"Unknown symbols in rate expression {rate_str!r}: {unknown}"
        raise ValueError(msg)
    fn = sympy.lambdify([sympy.Symbol(s) for s in free], expr, modules="math")
    return fn, free


def _first_top_level_comma(s: str) -> int:
    """Return index of first comma not inside parentheses, or -1."""
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            return i
    return -1


def _validate_variables(
    reactants: dict[str, float],
    products: dict[str, float],
    variables: dict[str, float],
) -> None:
    all_rxn_vars = set(reactants) | set(products)
    missing = all_rxn_vars - set(variables)
    if missing:
        msg = f"Species in reaction not found in variables dict: {sorted(missing)}"
        raise ValueError(msg)


def _validate_args(
    args: list[str],
    all_names: set[str],
    rate_str: str,
) -> None:
    missing = [a for a in args if a not in all_names]
    if missing:
        msg = f"Unknown symbols {missing!r} in rate expression {rate_str!r}"
        raise ValueError(msg)


def from_dsl[T: Model](
    model: T,
    network: str,
    *,
    variables: dict[str, float],
    parameters: dict[str, float],
) -> T:
    """Build a Model from a string-based reaction network DSL.

    Each non-empty, non-comment line has the form::

        rate_expr, reactants --> products
        (fwd_expr, rev_expr), reactants <--> products   # reversible

    Rate expression shorthands:

    - ``mm(kcat, Km)``  — Michaelis-Menten; substrate inferred from single consumed reactant
    - ``hill(v, K, n)`` — Hill kinetics; substrate inferred from single consumed reactant
    - ``ma(k)``         — mass-action over all consumed reactants
    - ``fn_name(...)``  — any function defined in ``mxlpy.fns``
    - anything else     — treated as a SymPy expression; free symbols become args

    Parameters
    ----------
    network
        Multi-line string describing the reaction network.
    variables
        Initial conditions for every variable that appears in the network.
        Raises ``ValueError`` for variables not listed here.
    parameters
        Values for every parameter that appears in rate expressions.
        Raises ``ValueError`` for parameters not listed here.

    Returns
    -------
    Model
        Fully constructed model.

    Examples
    --------
    >>> model = model.add_reactions_from_dsl(
    ...     "k1, A --> B",
    ...     variables={"A": 1.0, "B": 0.0},
    ...     parameters={"k1": 0.1},
    ... )

    """
    all_names: set[str] = set(variables) | set(parameters)

    model.add_variables(variables)
    model.add_parameters(parameters)

    reaction_counter = 0

    for raw_line in network.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Split on first comma NOT inside parentheses
        comma_idx = _first_top_level_comma(line)
        if comma_idx == -1:
            msg = f"Missing comma separator in line: {line!r}"
            raise ValueError(msg)
        rate_part = line[:comma_idx].strip()
        rxn_part = line[comma_idx + 1 :].strip()

        # Detect reversible
        is_reversible = bool(_REVERSIBLE_SEP.search(rxn_part))

        reaction_counter += 1
        r_idx = reaction_counter

        if is_reversible:
            # rate_part must be (fwd, rev)
            rev_m = re.fullmatch(r"\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", rate_part)
            if rev_m is None:
                msg = (
                    f"Reversible reaction requires rate tuple (fwd, rev), "
                    f"got: {rate_part!r}"
                )
                raise ValueError(msg)
            fwd_rate_str = rev_m.group(1).strip()
            rev_rate_str = rev_m.group(2).strip()

            parts = _REVERSIBLE_SEP.split(rxn_part, maxsplit=1)
            reactants = _parse_side(parts[0])
            products = _parse_side(parts[1])

            _validate_variables(reactants, products, variables)

            stoich_fwd = _net_stoichiometry(reactants, products)
            stoich_rev = {k: -v for k, v in stoich_fwd.items()}

            fwd_fn, fwd_args = _resolve_rate(
                fwd_rate_str, reactants, products, all_names
            )
            _validate_args(fwd_args, all_names, fwd_rate_str)

            # For reverse, swap reactant/product perspective
            rev_fn, rev_args = _resolve_rate(
                rev_rate_str, products, reactants, all_names
            )
            _validate_args(rev_args, all_names, rev_rate_str)

            model.add_reaction(
                f"r{r_idx}_fwd", fwd_fn, args=fwd_args, stoichiometry=stoich_fwd
            )
            model.add_reaction(
                f"r{r_idx}_rev", rev_fn, args=rev_args, stoichiometry=stoich_rev
            )

        else:
            parts = _IRREVERSIBLE_SEP.split(rxn_part, maxsplit=1)
            if len(parts) != 2:  # noqa: PLR2004
                msg = f"Could not parse reaction: {rxn_part!r}"
                raise ValueError(msg)
            reactants = _parse_side(parts[0])
            products = _parse_side(parts[1])

            _validate_variables(reactants, products, variables)

            stoich = _net_stoichiometry(reactants, products)
            fn, args = _resolve_rate(rate_part, reactants, products, all_names)
            _validate_args(args, all_names, rate_part)

            model.add_reaction(f"r{r_idx}", fn, args=args, stoichiometry=stoich)

    return model
