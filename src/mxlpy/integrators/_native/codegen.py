# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Sympy expression walking shared by the LLVM and WAT emitters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import sympy

from mxlpy.integrators._native.lower import NativeLoweringError

__all__ = ["ExprEmitter"]

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

    def __init__(self) -> None:
        """Start with no known symbols."""
        self.env: dict[str, str] = {}
        self.memo: dict[Any, str] = {}

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

    def _compute(self, e: Any) -> str:
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
        """Return the operand holding `e`, emitting code on first use."""
        if e not in self.memo:
            self.memo[e] = self._compute(e)
        return self.memo[e]
