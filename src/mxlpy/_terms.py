# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Lowering of terms into contribution nodes.

A term rewrites a variable's derivative or a reaction's rate through its
mechanism. Its contribution `Δ = g(prev, term) - prev` is a named model
quantity, so `dx/dt = base + Σ Δ` for any mechanism:

- a reaction term's `prev` is the reaction's rate plus the contributions of
  earlier terms on the same reaction. Its contribution fans out through the
  reaction's stoichiometry, so it corrects every variable the reaction
  touches consistently.
- a variable term's `prev` is the assembled derivative: every flux
  (including reaction-term contributions) times its stoichiometry, plus the
  contributions of earlier terms on the same variable.

Contributions never enter the stoichiometric matrix.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, overload

import pandas as pd

from mxlpy.meta.sympy_tools import fn_and_args, updated_fn_and_args
from mxlpy.types import Derived, Mechanism, StructureToken, Term

if TYPE_CHECKING:
    from collections.abc import Mapping

    import sympy

    from mxlpy.types import MechanismLike, RateFn

__all__ = [
    "TermNode",
    "TermsMixin",
    "corrected_fluxes",
    "lower_terms",
]

type Coefficient = float | Derived


@dataclass(slots=True)
class TermNode:
    """Computes one term's contribution.

    Attributes
    ----------
    term
        The term
    base
        `(name, coefficient)` pairs summing to the value the term acts on
    fan_out
        Coefficients the contribution enters the derivatives with

    """

    term: Term
    base: list[tuple[str, Coefficient]]
    fan_out: dict[str, Coefficient]

    @property
    def args(self) -> list[str]:
        """Names the contribution depends on."""
        args = list(self.term.args)
        if self.term.mechanism.needs_base:
            for name, coef in self.base:
                args.append(name)
                if isinstance(coef, Derived):
                    args.extend(coef.args)
        return list(dict.fromkeys(args))

    def base_value(self, args: Mapping[str, Any]) -> Any:
        """Value the term acts on."""
        out: Any = 0.0
        for name, coef in self.base:
            n = (
                coef.fn(*(args[a] for a in coef.args))
                if isinstance(coef, Derived)
                else coef
            )
            out = out + n * args[name]
        return out

    def calculate(self, args: Mapping[str, Any]) -> Any:
        """Contribution of the term."""
        term = self.term.fn(*(args[a] for a in self.term.args))
        mechanism = self.term.mechanism
        base = self.base_value(args) if mechanism.needs_base else 0.0
        return mechanism.contribution(base, term)

    def calculate_inpl(self, name: str, args: dict[str, Any]) -> None:
        """Write the contribution into `args`."""
        args[name] = self.calculate(args)


def corrected_fluxes[T: (pd.Series, pd.DataFrame)](
    fluxes: T, args: T, terms: Mapping[str, Term]
) -> T:
    """Reaction rates plus the contributions of the terms on them.

    `fluxes` and `args` are indexed by name (Series) or have names as
    columns (DataFrame); `args` has to contain the contributions.
    """
    out = fluxes.copy()
    names = out.index if isinstance(out, pd.Series) else out.columns
    for name, term in terms.items():
        if term.target in names:
            out[term.target] = out[term.target] + args[name]
    return out


def lower_terms(
    terms: Mapping[str, Term],
    fluxes: Mapping[str, Mapping[str, Coefficient]],
    reaction_names: set[str],
) -> dict[str, TermNode]:
    """Lower terms into contribution nodes, in insertion order.

    Parameters
    ----------
    terms
        Terms by name
    fluxes
        Stoichiometry of every flux by flux name. For an `OdeModelBuilder`
        these are the diff eq rates, each with coefficient 1 on its variable.
    reaction_names
        Fluxes terms may target as reactions

    Returns
    -------
    dict[str, TermNode]
        Contribution nodes by term name

    """
    nodes: dict[str, TermNode] = {}
    by_reaction: dict[str, list[str]] = {}
    for name, term in terms.items():
        if term.target in reaction_names:
            prev = by_reaction.setdefault(term.target, [])
            nodes[name] = TermNode(
                term=term,
                base=[(term.target, 1.0), *((p, 1.0) for p in prev)],
                fan_out=dict(fluxes[term.target]),
            )
            prev.append(name)

    # Fluxes and reaction-term contributions, by the variables they touch
    into: dict[str, list[tuple[str, Coefficient]]] = {}
    for flux, stoichiometry in fluxes.items():
        for contribution in (flux, *by_reaction.get(flux, [])):
            for variable, coef in stoichiometry.items():
                into.setdefault(variable, []).append((contribution, coef))

    by_variable: dict[str, list[str]] = {}
    for name, term in terms.items():
        if term.target in reaction_names:
            continue
        prev = by_variable.setdefault(term.target, [])
        nodes[name] = TermNode(
            term=term,
            base=[*into.get(term.target, []), *((p, 1.0) for p in prev)],
            fan_out={term.target: 1.0},
        )
        prev.append(name)
    return {name: nodes[name] for name in terms}


def _takes_two_args(fn: Any) -> bool:
    try:
        inspect.signature(fn).bind(0.0, 0.0)
    except TypeError:
        return False
    except ValueError:  # no signature, e.g. some builtins
        return True
    return True


def _checked_mechanism(name: str, mechanism: MechanismLike | Mechanism) -> Mechanism:
    mech = Mechanism.of(mechanism)
    if not _takes_two_args(mech.fn):
        msg = f"Term {name!r}: mechanism must take two arguments (base, term)"
        raise ValueError(msg)
    return mech


class TermsMixin:
    """Term API shared by `KineticModelBuilder` and `OdeModelBuilder`."""

    __slots__ = ()

    if TYPE_CHECKING:
        _terms: dict[str, Term]
        _cache: Any
        _structure: StructureToken

        def _insert_id(self, *, name: str, ctx: str) -> None: ...
        def _remove_id(self, *, name: str) -> None: ...
        def get_args(
            self, variables: dict[str, float] | None = None, time: float = 0.0
        ) -> pd.Series: ...
        def get_args_time_course(self, variables: pd.DataFrame) -> pd.DataFrame: ...
        def _all_terms(self) -> dict[str, Term]: ...
        def _check_not_block_input(self, name: str) -> None: ...

    def _invalidate(self) -> None:
        self._cache = None
        self._structure = StructureToken()

    def _check_term_target(self, name: str, target: str) -> None:
        raise NotImplementedError

    def _remove_targeting_terms(self, name: str, *, remove: bool) -> None:
        """Remove the terms targeting `name` (if `remove`), before removing it.

        Hidden-state blocks can't lose a target or an input, so removing one
        raises. Without `remove`, terms and blocks keep their references, like
        reactions with `remove_stoichiometries=False`, e.g. for `compose`,
        which adds the name again.
        """
        if not remove:
            return
        self._check_not_block_input(name)
        blocks = getattr(self, "_hidden_states", {})
        if targeting := [k for k, v in blocks.items() if name in v.targets]:
            msg = (
                f"{name!r} is a target of the hidden-state blocks {targeting}, "
                "remove them first"
            )
            raise ValueError(msg)
        for k in [k for k, v in self._terms.items() if v.target == name]:
            self.remove_term(k)

    @overload
    def add_term(
        self,
        name: str,
        fn: sympy.Expr,
        *,
        target: str,
        mechanism: MechanismLike | Mechanism,
        unit: sympy.Expr | None = None,
        display_name: str | None = None,
        tex_name: str | None = None,
    ) -> Self: ...

    @overload
    def add_term(
        self,
        name: str,
        fn: RateFn,
        *,
        args: list[str],
        target: str,
        mechanism: MechanismLike | Mechanism,
        unit: sympy.Expr | None = None,
        display_name: str | None = None,
        tex_name: str | None = None,
    ) -> Self: ...

    def add_term(
        self,
        name: str,
        fn: RateFn | sympy.Expr,
        *,
        args: list[str] | None = None,
        target: str,
        mechanism: MechanismLike | Mechanism,
        unit: sympy.Expr | None = None,
        display_name: str | None = None,
        tex_name: str | None = None,
    ) -> Self:
        """Add a term rewriting a variable's derivative or a reaction's rate.

        The term value `fn(*args)` is combined with the target through
        `mechanism`: `"additive"` (`base + term`), `"relative"`
        (`base * (1 + term)`) or a callable `(base, term) -> value`. Several
        terms on the same target are applied in insertion order, each acting on
        the result of the previous ones. A variable term acts on the assembled
        derivative, a reaction term on the rate before the stoichiometry fans
        it out, so it corrects every variable the reaction touches. Terms never
        enter the stoichiometric matrix.

        The term's contribution `Δ = mechanism(prev, term) - prev` is available
        under `name`, so `dx/dt = base + Σ Δ`.

        Examples
        --------
            >>> model.add_term(
            ...     "leak",
            ...     fn=fns.mass_action_1s,
            ...     args=["S", "k_leak"],
            ...     target="S",
            ...     mechanism="additive",
            ... )

        Parameters
        ----------
        name
            Name of the term, also the name of its contribution
        fn
            Function or sympy expression of the term value
        args
            Model names to call `fn` with
        target
            Variable or reaction the term acts on
        mechanism
            How the term combines with its target, required
        unit
            Unit of the term value
        display_name
            Human-readable name, e.g. for user interfaces.
        tex_name
            LaTeX rendering of the name.

        Returns
        -------
        Self
            The instance of the model with the added term.

        """
        fn, args = fn_and_args(fn, args, origin=name)
        self._check_term_target(name, target)
        mech = _checked_mechanism(name, mechanism)
        self._insert_id(name=name, ctx="term")
        self._invalidate()
        self._terms[name] = Term(
            fn=fn,
            args=args,
            target=target,
            mechanism=mech,
            unit=unit,
            display_name=display_name,
            tex_name=tex_name,
        )
        return self

    def update_term(
        self,
        name: str,
        fn: RateFn | sympy.Expr | None = None,
        *,
        args: list[str] | None = None,
        target: str | None = None,
        mechanism: MechanismLike | Mechanism | None = None,
        unit: sympy.Expr | None = None,
        display_name: str | None = None,
        tex_name: str | None = None,
    ) -> Self:
        """Update a term, keeping its position in the composition order.

        Parameters
        ----------
        name
            Name of the term
        fn
            Function or sympy expression of the term value
        args
            Model names to call `fn` with
        target
            Variable or reaction the term acts on
        mechanism
            How the term combines with its target
        unit
            Unit of the term value
        display_name
            Human-readable name, e.g. for user interfaces.
        tex_name
            LaTeX rendering of the name.

        Returns
        -------
        Self
            The instance of the model with the updated term.

        """
        if (term := self._terms.get(name)) is None:
            msg = f"Term {name!r} not found. Available terms: {sorted(self._terms)}"
            raise KeyError(msg)
        new_fn, new_args = updated_fn_and_args(
            fn, args, origin=name, old_fn=term.fn, old_args=term.args
        )
        if target is not None:
            self._check_term_target(name, target)
        mech = None if mechanism is None else _checked_mechanism(name, mechanism)
        self._invalidate()
        term.fn, term.args = new_fn, new_args
        if target is not None:
            term.target = target
        if mech is not None:
            term.mechanism = mech
        if unit is not None:
            term.unit = unit
        if display_name is not None:
            term.display_name = display_name
        if tex_name is not None:
            term.tex_name = tex_name
        return self

    def remove_term(self, name: str) -> Self:
        """Remove a term.

        Parameters
        ----------
        name
            Name of the term

        Returns
        -------
        Self
            The instance of the model with the term removed.

        """
        self._remove_id(name=name)
        self._invalidate()
        del self._terms[name]
        return self

    def get_raw_terms(self, *, as_copy: bool = True) -> dict[str, Term]:
        """Get the terms, in composition order.

        Parameters
        ----------
        as_copy
            If True, return a deep copy of the terms dictionary.
            If False, return the internal dictionary directly.

        """
        if as_copy:
            return copy.deepcopy(self._terms)
        return self._terms

    def get_term_names(self) -> list[str]:
        """Names of all terms and their contributions, including those of blocks."""
        return list(self._all_terms())

    def get_term_contributions(
        self,
        variables: dict[str, float] | None = None,
        time: float = 0.0,
    ) -> pd.Series:
        """Contribution of every term to its target.

        Parameters
        ----------
        variables
            Variable values, the initial conditions by default
        time
            Time point

        Returns
        -------
        pd.Series
            Contributions by term name

        """
        return self.get_args(variables, time).loc[self.get_term_names()]
