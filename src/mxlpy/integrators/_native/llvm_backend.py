# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Model -> machine code with llvmlite, solved by the compiled mxlpy-native library."""

from __future__ import annotations

import ctypes as ct
import struct
import sys
from ctypes.util import find_library
from functools import lru_cache
from typing import TYPE_CHECKING

import llvmlite.binding as llvm
import mxlpy_native
import numpy as np

from mxlpy.integrators._native.codegen import ExprEmitter

if TYPE_CHECKING:
    from mxlpy.integrators._native.lower import Lowered

__all__ = ["compile_model"]

_DP = ct.POINTER(ct.c_double)
_MODEL_FN = ct.CFUNCTYPE(None, ct.c_int, ct.c_double, _DP, _DP, _DP)
_CMP = {"gt": "ogt", "ge": "oge", "lt": "olt", "le": "ole", "eq": "oeq", "ne": "one"}
_BINARY = {"add": "fadd", "mul": "fmul", "div": "fdiv"}


class _Emit(ExprEmitter):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.decls: set[str] = set()
        self.tmp = 0

    def _new(self, instr: str) -> str:
        self.tmp += 1
        reg = f"%r{self.tmp}"
        self.lines.append(f"  {reg} = {instr}")
        return reg

    def const(self, value: float) -> str:
        return "0x" + struct.pack(">d", value).hex().upper()

    def binary(self, op: str, a: str, b: str) -> str:
        return self._new(f"{_BINARY[op]} double {a}, {b}")

    def call(self, fn: str, *args: str) -> str:
        self.decls.add(f"declare double @{fn}({', '.join('double' for _ in args)})")
        joined = ", ".join(f"double {a}" for a in args)
        return self._new(f"call double @{fn}({joined})")

    def sqrt(self, a: str) -> str:
        self.decls.add("declare double @llvm.sqrt.f64(double)")
        return self._new(f"call double @llvm.sqrt.f64(double {a})")

    def fabs(self, a: str) -> str:
        self.decls.add("declare double @llvm.fabs.f64(double)")
        return self._new(f"call double @llvm.fabs.f64(double {a})")

    def compare(self, pred: str, a: str, b: str) -> str:
        return self._new(f"fcmp {_CMP[pred]} double {a}, {b}")

    def select(self, cond: str, a: str, b: str) -> str:
        return self._new(f"select i1 {cond}, double {a}, double {b}")

    def boolean(self, value: bool) -> str:  # noqa: FBT001
        return "true" if value else "false"

    def logic(self, op: str, a: str, b: str) -> str:
        return self._new(f"{op} i1 {a}, {b}")

    def negate(self, a: str) -> str:
        return self._new(f"xor i1 {a}, true")


def to_llvm_ir(low: Lowered) -> str:
    """Emit `void fcn(int n, double t, double *y, double *dydt, double *pars)` as LLVM IR."""
    em = _Emit()
    em.env["t"] = "%t"

    def load(ptr: str, i: int, name: str) -> None:
        em.lines.append(f"  %{name}.p = getelementptr double, ptr {ptr}, i64 {i}")
        em.lines.append(f"  %{name} = load double, ptr %{name}.p")
        em.env[name] = f"%{name}"

    for i in range(low.n):
        load("%y", i, f"y{i}")
    for i in range(len(low.par_names)):
        load("%pars", i, f"p{i}")
    for name, expr in low.assignments:
        em.env[name] = em.value(expr)
        em.memo.clear()
    for i, expr in enumerate(low.dydt):
        v = em.value(expr)
        em.lines.append(f"  %out{i} = getelementptr double, ptr %dydt, i64 {i}")
        em.lines.append(f"  store double {v}, ptr %out{i}")
    return "\n".join(
        [
            *sorted(em.decls),
            "define void @fcn(i32 %n, double %t, ptr %y, ptr %dydt, ptr %pars) {",
            *em.lines,
            "  ret void",
            "}",
            "",
        ]
    )


@lru_cache(maxsize=1)
def _initialise() -> None:
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    libm = "ucrtbase" if sys.platform == "win32" else find_library("m")
    if libm:
        llvm.load_library_permanently(libm)


@lru_cache(maxsize=1)
def _solver() -> ct.CDLL:
    lib = ct.CDLL(str(mxlpy_native.library_path()))
    lib.mxn_set_model.argtypes = [ct.c_void_p]
    lib.mxn_set_model.restype = None
    lib.mxn_radau.restype = ct.c_int
    lib.mxn_radau.argtypes = [
        ct.c_int, _DP, _DP, ct.c_double, ct.c_double, ct.c_int, ct.c_int, ct.c_int, _DP, _DP,
    ]  # fmt: skip
    return lib


class _Compiled:
    def __init__(self, low: Lowered) -> None:
        _initialise()
        # The engine takes ownership of its target machine, so they can't be shared.
        tm = llvm.Target.from_triple(llvm.get_default_triple()).create_target_machine(
            opt=2
        )
        mod = llvm.parse_assembly(to_llvm_ir(low))
        mod.verify()
        pto = llvm.create_pipeline_tuning_options(speed_level=2)
        pb = llvm.create_pass_builder(tm, pto)
        pb.getModulePassManager().run(mod, pb)
        self._engine = llvm.create_mcjit_compiler(mod, tm)
        self._engine.finalize_object()
        self._ptr = self._engine.get_function_address("fcn")
        self._fn = _MODEL_FN(self._ptr)
        self._lib = _solver()

    def rhs(self, t: float, y: np.ndarray, pars: np.ndarray) -> np.ndarray:
        y, dydt = np.array(y, dtype=float), np.empty(len(y))
        self._fn(
            len(y),
            t,
            y.ctypes.data_as(_DP),
            dydt.ctypes.data_as(_DP),
            pars.ctypes.data_as(_DP),
        )
        return dydt

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
        y = np.array(y0, dtype=float)
        out = np.empty((len(t_out), len(y)))
        self._lib.mxn_set_model(self._ptr)
        idid = self._lib.mxn_radau(
            len(y), y.ctypes.data_as(_DP), pars.ctypes.data_as(_DP), rtol, atol, nmax,
            int(reuse_jacobian), len(t_out), t_out.ctypes.data_as(_DP), out.ctypes.data_as(_DP),
        )  # fmt: skip
        return int(idid), out


def compile_model(low: Lowered) -> _Compiled:
    """Compile a lowered model to machine code."""
    return _Compiled(low)
