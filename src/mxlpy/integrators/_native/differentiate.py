# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Forward- and reverse-mode differentiation of lowered models.

The derivatives are new outputs (and assignments) of a `Lowered`, so every
backend compiles them like a model: forward sensitivities and derived-target
gradients for the compiled Levenberg-Marquardt loop, the adjoint function for
compiled adjoint gradients.
"""

from __future__ import annotations

from typing import Any

import sympy

from mxlpy.integrators._native.lower import Lowered
from mxlpy.symbolic.jacobian import piecewise_extrema as _piecewise

__all__ = ["adjoint", "augmented", "gradients", "jacobian"]


type _Seed = dict[sympy.Symbol, Any]


def _differentiate(
    low: Lowered, seeds: list[_Seed], outputs: list[Any]
) -> tuple[list[tuple[str, Any]], list[list[Any]]]:
    """Directional derivatives of `outputs` along each seed.

    A seed maps input symbols (`y0`, `p0`, ...) to their derivative along that
    direction. Returns the assignments with their derivatives interleaved and
    the derivative of every output along every seed.
    """
    assignments: list[tuple[str, Any]] = []
    tangents = [dict(seed) for seed in seeds]
    counter = len(low.assignments)

    def tangent(expr: Any, t: _Seed) -> Any:
        expr = _piecewise(expr)
        terms = [
            sympy.diff(expr, s) * t[s]
            for s in getattr(expr, "free_symbols", set())
            if t.get(s, 0) != 0
        ]
        return sympy.Add(*terms) if terms else sympy.Integer(0)

    for name, expr in low.assignments:
        assignments.append((name, expr))
        symbol = sympy.Symbol(name)
        for t in tangents:
            d = tangent(expr, t)
            if d == 0:
                continue
            new = sympy.Symbol(f"a{counter}")
            counter += 1
            assignments.append((new.name, d))
            t[symbol] = new
    return assignments, [[tangent(out, t) for t in tangents] for out in outputs]


def augmented(low: Lowered, theta: list[int]) -> Lowered:
    """States and their sensitivities to the parameters `theta` (indices into pars).

    The sensitivities are inputs `y{n}`, ... and outputs after the states,
    parameter-major, as mxl-solvers' jacobian_wrapper.c reads them.
    """
    n = low.n
    seeds: list[_Seed] = [
        {sympy.Symbol(f"y{i}"): sympy.Symbol(f"y{n + j * n + i}") for i in range(n)}
        | {sympy.Symbol(f"p{k}"): sympy.Integer(1)}
        for j, k in enumerate(theta)
    ]
    assignments, d = _differentiate(low, seeds, low.dydt)
    sens = [d[i][j] for j in range(len(theta)) for i in range(n)]
    return Lowered(
        n=n * (1 + len(theta)),
        par_names=low.par_names,
        assignments=assignments,
        dydt=[*low.dydt, *sens],
    )


def gradients(low: Lowered, theta: list[int]) -> Lowered:
    """Per output: its derivatives to every state, then to the parameters `theta`."""
    seeds: list[_Seed] = [
        {sympy.Symbol(f"y{i}"): sympy.Integer(1)} for i in range(low.n)
    ]
    seeds += [{sympy.Symbol(f"p{k}"): sympy.Integer(1)} for k in theta]
    assignments, d = _differentiate(low, seeds, low.dydt)
    return Lowered(
        n=low.n,
        par_names=low.par_names,
        assignments=assignments,
        dydt=[g for row in d for g in row],
    )


def jacobian(low: Lowered) -> Lowered:
    """The Jacobian df/dy as n x n outputs, column-major (output `i + j n` is df_i/dy_j).

    The layout the solvers' analytic-Jacobian hooks and mxl-solvers' discrete adjoint
    read (RADAU5's DFY, LSODA's PD, SUNDIALS' dense matrices).
    """
    n = low.n
    seeds: list[_Seed] = [{sympy.Symbol(f"y{j}"): sympy.Integer(1)} for j in range(n)]
    assignments, d = _differentiate(low, seeds, low.dydt)
    return Lowered(
        n=n,
        par_names=low.par_names,
        assignments=assignments,
        dydt=[d[i][j] for j in range(n) for i in range(n)],
    )


def adjoint(low: Lowered, theta: list[int]) -> Lowered:
    """mxl-solvers' adjoint function of the model, by reverse-mode differentiation.

    Inputs are the states `y0..y{n-1}`, then the adjoint states `y{n}..y{2n-1}`.
    Outputs are `dlam/dt = -(df/dy)ᵀ lam`, then `-lamᵀ df/dp_k` for the parameters
    `theta` (indices into pars), as `llvm_backend.compile_model(..., adjoint=n)` emits
    them. Each assignment's adjoint is accumulated from its consumers, so the cost
    doesn't grow with the number of parameters.
    """
    n = low.n
    bars: dict[sympy.Symbol, list[Any]] = {}

    def push(expr: Any, bar: Any) -> None:
        expr = _piecewise(expr)
        for s in getattr(expr, "free_symbols", set()):
            d = sympy.diff(expr, s)
            if d != 0:
                bars.setdefault(s, []).append(d * bar)

    # Seeded with -lam, so the outputs are the negative products
    for i, f in enumerate(low.dydt):
        push(f, -sympy.Symbol(f"y{n + i}"))
    assignments = list(low.assignments)
    for k, (name, expr) in enumerate(reversed(low.assignments)):
        if (terms := bars.get(sympy.Symbol(name))) is None:
            continue
        bar = sympy.Symbol(f"r{k}")
        assignments.append((bar.name, sympy.Add(*terms)))
        push(expr, bar)

    def total(name: str) -> Any:
        return sympy.Add(*bars.get(sympy.Symbol(name), [sympy.Integer(0)]))

    return Lowered(
        n=2 * n,
        par_names=low.par_names,
        assignments=assignments,
        dydt=[total(f"y{i}") for i in range(n)] + [total(f"p{k}") for k in theta],
    )
