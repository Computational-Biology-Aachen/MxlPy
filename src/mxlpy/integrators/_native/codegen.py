# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Sympy expression walking shared by the LLVM and WAT emitters.

Array-valued expressions (mxl-schemas ADR 0015) are evaluated element by element
inside counted loops. An index is `(register, offset)`, the register being the
loop counter (None for a constant index). Inside a loop, `value` evaluates the
element at the current index; every scalar subexpression is computed before the
loop starts (`_hoist`), so it is neither recomputed per element nor defined inside
the loop. A reduction is a loop with accumulators; a named array is a loop
storing into its buffer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import sympy

from mxlpy.integrators._native.lower import NativeLoweringError
from mxlpy.meta.array_fns import (
    ArrayIndex,
    ArrayMaxOf,
    ArrayMean,
    ArrayMinOf,
    ArraySlice,
    ArrayStd,
    ArraySum,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["ExprEmitter", "Index"]

# (loop counter register or None, constant offset)
type Index = tuple[str | None, int]
_REDUCTIONS = (ArraySum, ArrayMean, ArrayStd, ArrayMaxOf, ArrayMinOf)


def _int(bound: Any) -> int:
    """An index bound, an integer after lowering."""
    return int(bound)


_CMP = {
    sympy.StrictGreaterThan: "gt",
    sympy.GreaterThan: "ge",
    sympy.StrictLessThan: "lt",
    sympy.LessThan: "le",
    sympy.Equality: "eq",
    sympy.Unequality: "ne",
}
_MATH = {
    sympy.exp: "exp",
    sympy.log: "log",
    sympy.sin: "sin",
    sympy.cos: "cos",
    sympy.tan: "tan",
    sympy.asin: "asin",
    sympy.acos: "acos",
    sympy.atan: "atan",
    sympy.sinh: "sinh",
    sympy.cosh: "cosh",
    sympy.tanh: "tanh",
    sympy.asinh: "asinh",
    sympy.acosh: "acosh",
    sympy.atanh: "atanh",
}
_MIN_INLINE_POWER = 2
_MAX_INLINE_POWER = 4


class ExprEmitter(ABC):
    """Turns expressions into target operands, computing each subexpression once."""

    def __init__(self, lengths: dict[str, int] | None = None) -> None:
        """Start with no known symbols; `lengths` holds every array name (data, buffers)."""
        self.env: dict[str, str] = {}
        self.memo: dict[Any, str] = {}
        self.lengths: dict[str, int] = lengths or {}
        self.idx: Index | None = None
        self.elem_memo: dict[Any, str] = {}
        self._is_array: dict[Any, bool] = {}

    @abstractmethod
    def const(self, value: float) -> str:
        """Emit a floating point constant."""

    @abstractmethod
    def binary(self, op: str, a: str, b: str) -> str:
        """Emit `op` (add, mul or div) on two floats."""

    @abstractmethod
    def call(self, fn: str, *args: str) -> str:
        """Emit a call to a math function."""

    @abstractmethod
    def sqrt(self, a: str) -> str:
        """Emit a square root."""

    @abstractmethod
    def fabs(self, a: str) -> str:
        """Emit an absolute value."""

    @abstractmethod
    def compare(self, pred: str, a: str, b: str) -> str:
        """Emit an ordered comparison: gt, ge, lt, le, eq, ne."""

    @abstractmethod
    def select(self, cond: str, a: str, b: str) -> str:
        """Emit `a if cond else b`."""

    @abstractmethod
    def boolean(self, value: bool) -> str:  # noqa: FBT001
        """Emit a boolean constant."""

    @abstractmethod
    def logic(self, op: str, a: str, b: str) -> str:
        """Emit `and` or `or` on two booleans."""

    @abstractmethod
    def negate(self, a: str) -> str:
        """Emit a boolean not."""

    def load(self, array: str, idx: Index) -> str:
        """Emit a load of element `idx` of a data array or buffer."""
        raise NotImplementedError

    def store(self, array: str, idx: Index, value: str) -> None:
        """Emit a store of `value` into element `idx` of a buffer."""
        raise NotImplementedError

    def loop(
        self,
        n: int,
        inits: list[str],
        body: Callable[[str, list[str]], list[str]],
    ) -> list[str]:
        """Emit a loop over 0..n-1 with accumulators starting at `inits`.

        `body(counter, accs)` emits one iteration and returns the accumulators'
        next values; the loop returns their final values.
        """
        raise NotImplementedError

    # Arrays

    def is_array(self, e: Any) -> bool:
        """Whether `e` is an array: it references an array outside of a reduction."""
        if e not in self._is_array:
            if e.is_Symbol:
                self._is_array[e] = e.name in self.lengths
            elif isinstance(e, ArraySlice):
                self._is_array[e] = True
            elif isinstance(e, (*_REDUCTIONS, ArrayIndex)):
                self._is_array[e] = False
            else:
                self._is_array[e] = any(self.is_array(a) for a in e.args)
        return self._is_array[e]

    def length(self, e: Any) -> int:
        """The length of the array `e`."""
        if e.is_Symbol:
            return self.lengths[e.name]
        if isinstance(e, ArraySlice):
            return _int(e.args[2]) - _int(e.args[1])
        return next(self.length(a) for a in e.args if self.is_array(a))

    def _hoist(self, e: Any) -> None:
        if not self.is_array(e):
            self.value(e)
        elif isinstance(e, ArraySlice):
            self._hoist(e.args[0])
        else:
            for a in e.args:
                self._hoist(a)

    def _at(self, idx: Index, fn: Callable[[], Any]) -> Any:
        saved = self.idx, self.elem_memo
        self.idx, self.elem_memo = idx, {}
        try:
            return fn()
        finally:
            self.idx, self.elem_memo = saved

    def _loop(
        self, a: Any, inits: list[str], step: Callable[[list[str]], list[str]]
    ) -> list[str]:
        self._hoist(a)
        return self.loop(
            self.length(a),
            inits,
            lambda counter, accs: self._at((counter, 0), lambda: step(accs)),
        )

    def fill(self, name: str, expr: Any) -> None:
        """Emit the loop computing the named array `name` into its buffer."""

        def step(_: list[str]) -> list[str]:
            assert self.idx is not None  # noqa: S101
            self.store(name, self.idx, self.value(expr))
            return []

        self._loop(expr, [], step)

    def _sum(self, a: Any) -> str:
        return self._loop(
            a,
            [self.const(0.0)],
            lambda accs: [self.binary("add", accs[0], self.value(a))],
        )[0]

    def _reduce(self, e: Any) -> str:
        a = e.args[0]
        if isinstance(e, ArraySum):
            return self._sum(a)
        n = self.const(float(self.length(a)))
        if isinstance(e, ArrayMean):
            return self.binary("div", self._sum(a), n)
        if isinstance(e, ArrayStd):
            mean = self.binary("div", self._sum(a), n)
            neg_mean = self.binary("mul", self.const(-1.0), mean)

            def sq(accs: list[str]) -> list[str]:
                d = self.binary("add", self.value(a), neg_mean)
                return [self.binary("add", accs[0], self.binary("mul", d, d))]

            ss = self._loop(a, [self.const(0.0)], sq)[0]
            return self.sqrt(
                self.binary("div", ss, self.const(float(self.length(a) - 1)))
            )
        pred = "gt" if isinstance(e, ArrayMaxOf) else "lt"
        start = self.const(float("-inf") if pred == "gt" else float("inf"))

        def pick(accs: list[str]) -> list[str]:
            v = self.value(a)
            return [self.select(self.compare(pred, v, accs[0]), v, accs[0])]

        return self._loop(a, [start], pick)[0]

    def _fold(self, op: str, args: tuple[Any, ...]) -> str:
        out = self.value(args[0])
        for a in args[1:]:
            out = self.binary(op, out, self.value(a))
        return out

    def _pow(self, expr: Any) -> str:
        base, exponent = expr.args
        if exponent == -1:
            return self.binary("div", self.const(1.0), self.value(base))
        if (
            exponent.is_Integer
            and _MIN_INLINE_POWER <= int(exponent) <= _MAX_INLINE_POWER
        ):
            b = self.value(base)
            out = b
            for _ in range(int(exponent) - 1):
                out = self.binary("mul", out, b)
            return out
        if exponent == sympy.Rational(1, 2):
            return self.sqrt(self.value(base))
        return self.call("pow", self.value(base), self.value(exponent))

    def _cond(self, cond: Any) -> str:
        if cond in (sympy.true, sympy.false):
            return self.boolean(cond == sympy.true)
        if isinstance(cond, sympy.And | sympy.Or):
            op = "and" if isinstance(cond, sympy.And) else "or"
            out = self._cond(cond.args[0])
            for c in cond.args[1:]:
                out = self.logic(op, out, self._cond(c))
            return out
        if isinstance(cond, sympy.Not):
            return self.negate(self._cond(cond.args[0]))
        if type(cond) not in _CMP:
            msg = f"Native can't compile the condition {cond}"
            raise NativeLoweringError(msg)
        return self.compare(
            _CMP[type(cond)], self.value(cond.lhs), self.value(cond.rhs)
        )

    def _extreme(self, pred: str, args: tuple[Any, ...]) -> str:
        out = self.value(args[0])
        for a in args[1:]:
            v = self.value(a)
            out = self.select(self.compare(pred, out, v), out, v)
        return out

    def _compute_array(self, e: Any) -> str | None:
        if e.is_Symbol:
            if e.name not in self.lengths:
                return None
            if self.idx is None:
                msg = f"Native found the array {e.name} where a number is needed"
                raise NativeLoweringError(msg)
            return self.load(e.name, self.idx)
        if isinstance(e, ArraySlice):
            assert self.idx is not None  # noqa: S101
            reg, offset = self.idx
            return self._at(
                (reg, offset + _int(e.args[1])), lambda: self.value(e.args[0])
            )
        if isinstance(e, _REDUCTIONS):
            return self._reduce(e)
        if isinstance(e, ArrayIndex):
            return self._at((None, _int(e.args[1])), lambda: self.value(e.args[0]))
        return None

    def _compute(self, e: Any) -> str:
        # models without arrays skip the array checks
        if self.lengths and (out := self._compute_array(e)) is not None:
            return out
        if e.is_Symbol:
            if e.name not in self.env:
                msg = f"Native found an unknown symbol: {e.name}"
                raise NativeLoweringError(msg)
            return self.env[e.name]
        if e.is_Number or e in (sympy.pi, sympy.E):
            return self.const(float(e))
        if isinstance(e, sympy.Add):
            return self._fold("add", e.args)
        if isinstance(e, sympy.Mul):
            return self._fold("mul", e.args)
        if isinstance(e, sympy.Pow):
            return self._pow(e)
        if isinstance(e, sympy.Piecewise):
            out = self.const(float("nan"))
            for val, cond in reversed(e.args):  # pyright: ignore[reportGeneralTypeIssues]
                out = self.select(self._cond(cond), self.value(val), out)
            return out
        if isinstance(e, sympy.Max):
            return self._extreme("gt", e.args)
        if isinstance(e, sympy.Min):
            return self._extreme("lt", e.args)
        if isinstance(e, sympy.Abs):
            return self.fabs(self.value(e.args[0]))
        if type(e) in _MATH:
            return self.call(_MATH[type(e)], self.value(e.args[0]))
        msg = f"Native can't compile {type(e).__name__}: {e}"
        raise NativeLoweringError(msg)

    def value(self, e: Any) -> str:
        """Return the operand holding `e`, emitting code on first use.

        Inside a loop, an array `e` stands for its element at the current index.
        """
        if self.idx is not None and self.is_array(e):
            if e not in self.elem_memo:
                self.elem_memo[e] = self._compute(e)
            return self.elem_memo[e]
        if e not in self.memo:
            self.memo[e] = self._compute(e)
        return self.memo[e]
