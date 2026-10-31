# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Model -> WASM, solved by the radau.wasm shipped with MxlPy, all run by wasmtime."""

from __future__ import annotations

import atexit
import ctypes as ct
import math
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import wasmtime as wt

from mxlpy.integrators._native.codegen import ExprEmitter

if TYPE_CHECKING:
    from mxlpy.integrators._native.lower import Lowered

__all__ = ["compile_model"]

_ASSET = Path(__file__).parent / "assets" / "radau.wasm"
_BINARY = {"add": "f64.add", "mul": "f64.mul", "div": "f64.div"}
_DOUBLE = 8


class _Emit(ExprEmitter):
    def __init__(self) -> None:
        super().__init__()
        self.body: list[str] = []
        self.imports: dict[str, int] = {}
        self.locals: list[str] = []
        self.tmp = 0

    def set_local(self, name: str, kind: str, expr: str) -> str:
        self.locals.append(f"(local ${name} {kind})")
        self.body.append(f"(local.set ${name} {expr})")
        return f"(local.get ${name})"

    def _new(self, expr: str, kind: str = "f64") -> str:
        self.tmp += 1
        return self.set_local(f"r{self.tmp}", kind, expr)

    def const(self, value: float) -> str:
        if math.isnan(value):
            return "(f64.const nan)"
        if math.isinf(value):
            return f"(f64.const {'-' if value < 0 else ''}inf)"
        return f"(f64.const {value!r})"

    def binary(self, op: str, a: str, b: str) -> str:
        return self._new(f"({_BINARY[op]} {a} {b})")

    def call(self, fn: str, *args: str) -> str:
        self.imports[fn] = len(args)
        return self._new(f"(call ${fn} {' '.join(args)})")

    def sqrt(self, a: str) -> str:
        return self._new(f"(f64.sqrt {a})")

    def fabs(self, a: str) -> str:
        return self._new(f"(f64.abs {a})")

    def compare(self, pred: str, a: str, b: str) -> str:
        return self._new(f"(f64.{pred} {a} {b})", "i32")

    def select(self, cond: str, a: str, b: str) -> str:
        return self._new(f"(select {a} {b} {cond})")

    def boolean(self, value: bool) -> str:  # noqa: FBT001
        return f"(i32.const {int(value)})"

    def logic(self, op: str, a: str, b: str) -> str:
        return self._new(f"(i32.{op} {a} {b})", "i32")

    def negate(self, a: str) -> str:
        return self._new(f"(i32.eqz {a})", "i32")


def to_wat(low: Lowered) -> str:
    """Emit a module exporting `fcn(n, t, y, dydt, pars)`, sharing the solver's memory."""
    em = _Emit()
    em.env["t"] = "(local.get $t)"
    for i in range(low.n):
        em.env[f"y{i}"] = em.set_local(
            f"y{i}", "f64", f"(f64.load offset={i * _DOUBLE} (local.get $y))"
        )
    for i in range(len(low.par_names)):
        em.env[f"p{i}"] = em.set_local(
            f"p{i}", "f64", f"(f64.load offset={i * _DOUBLE} (local.get $pars))"
        )
    for name, expr in low.assignments:
        em.env[name] = em.value(expr)
        em.memo.clear()
    for i, expr in enumerate(low.dydt):
        em.body.append(
            f"(f64.store offset={i * _DOUBLE} (local.get $dydt) {em.value(expr)})"
        )
    imports = [
        f'(import "math" "{fn}" (func ${fn} (param {" ".join(["f64"] * nargs)}) (result f64)))'
        for fn, nargs in sorted(em.imports.items())
    ]
    return "\n".join(
        [
            "(module",
            '(import "env" "memory" (memory 1))',
            *imports,
            '(func (export "fcn") (param $n i32) (param $t f64) (param $y i32) (param $dydt i32) (param $pars i32)',
            *em.locals,
            *em.body,
            "))",
        ]
    )


def _stub(store: wt.Store, item: Any) -> wt.Func:
    return wt.Func(store, item.type, lambda *_: None)


class _Runtime:
    """One solver instance per process, shared by all compiled models."""

    def __init__(self) -> None:
        self.engine = wt.Engine()
        self.store = wt.Store(self.engine)
        module = wt.Module.from_file(self.engine, str(_ASSET))
        imports: list[Any] = [_stub(self.store, i) for i in module.imports]
        # wasmtime types exports as a union of everything a module can export
        self.solver: Any = wt.Instance(self.store, module, imports).exports(self.store)
        self.solver["_initialize"](self.store)
        self.memory: Any = self.solver["memory"]
        self.table: Any = self.solver["__indirect_function_table"]

    def malloc(self, n_doubles: int) -> int:
        return self.solver["malloc"](self.store, max(n_doubles, 1) * _DOUBLE)

    def free(self, *ptrs: int) -> None:
        for ptr in ptrs:
            self.solver["free"](self.store, ptr)

    def view(self, ptr: int, n_doubles: int) -> np.ndarray:
        # Memory can move when it grows, so this must not be kept across solver calls.
        base = ct.addressof(self.memory.data_ptr(self.store).contents)
        return np.ctypeslib.as_array((ct.c_double * n_doubles).from_address(base + ptr))


@lru_cache(maxsize=1)
def _runtime() -> _Runtime:
    return _Runtime()


# wasmtime objects can't be freed once the interpreter has started shutting down.
atexit.register(_runtime.cache_clear)


class _Compiled:
    def __init__(self, low: Lowered) -> None:
        rt = _runtime()
        self._rt = rt
        module = wt.Module(rt.engine, to_wat(low))
        imports: list[Any] = [
            rt.memory if i.module == "env" else rt.solver[f"mxn_{i.name}"]
            for i in module.imports
        ]
        self._fcn: Any = wt.Instance(rt.store, module, imports).exports(rt.store)["fcn"]
        self._index = rt.table.grow(rt.store, 1, self._fcn)

    def _write(self, values: np.ndarray) -> int:
        ptr = self._rt.malloc(len(values))
        if len(values):
            self._rt.view(ptr, len(values))[:] = values
        return ptr

    def rhs(self, t: float, y: np.ndarray, pars: np.ndarray) -> np.ndarray:
        rt, n = self._rt, len(y)
        py, pp, pd = (
            self._write(np.asarray(y, dtype=float)),
            self._write(pars),
            rt.malloc(n),
        )
        self._fcn(rt.store, n, float(t), py, pd, pp)
        out = rt.view(pd, n).copy()
        rt.free(py, pp, pd)
        return out

    def eval_rows(
        self, t: np.ndarray, y: np.ndarray, pars: np.ndarray, n_out: int
    ) -> np.ndarray:
        rt = self._rt
        rows, n = np.shape(y)
        pt, py = (
            self._write(np.asarray(t, dtype=float)),
            self._write(np.asarray(y, dtype=float).ravel()),
        )
        pp, po = self._write(pars), rt.malloc(rows * n_out)
        rt.solver["mxn_eval_rows"](
            rt.store, self._index, n, n_out, rows, pt, py, po, pp
        )
        out = rt.view(po, rows * n_out).reshape(rows, n_out).copy()
        rt.free(pt, py, pp, po)
        return out

    def solve(
        self,
        y0: np.ndarray,
        pars: np.ndarray,
        t_out: np.ndarray,
        *,
        rtol: float,
        atol: float,
        nmax: int,
        reuse_jacobian: bool,
    ) -> tuple[int, np.ndarray]:
        rt, n, n_out = self._rt, len(y0), len(t_out)
        py, pp, pt = (
            self._write(np.asarray(y0, dtype=float)),
            self._write(pars),
            self._write(t_out),
        )
        po = rt.malloc(n * n_out)
        rt.solver["mxn_set_model"](rt.store, self._index)
        try:
            idid = rt.solver["mxn_radau"](
                rt.store, n, py, pp, rtol, atol, nmax, int(reuse_jacobian), n_out, pt, po,
            )  # fmt: skip
        except wt.Trap:
            idid = -5
        out = rt.view(po, n * n_out).reshape(n_out, n).copy()
        rt.free(py, pp, pt, po)
        return int(idid), out


def compile_model(low: Lowered) -> _Compiled:
    """Compile a lowered model to WASM."""
    return _Compiled(low)
