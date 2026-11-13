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
    from collections.abc import Callable

    from mxlpy.integrators._native.codegen import Index
    from mxlpy.integrators._native.lower import Lowered

__all__ = ["compile_model", "load"]

_DP = ct.POINTER(ct.c_double)
_MODEL_FN = ct.CFUNCTYPE(None, ct.c_int, ct.c_double, _DP, _DP, _DP)
_CMP = {"gt": "ogt", "ge": "oge", "lt": "olt", "le": "ole", "eq": "oeq", "ne": "one"}
_BINARY = {"add": "fadd", "mul": "fmul", "div": "fdiv"}


class _Emit(ExprEmitter):
    def __init__(self, low: Lowered) -> None:
        super().__init__({k: len(v) for k, v in low.data.items()} | low.arrays)
        self.lines: list[str] = []
        self.decls: set[str] = set()
        self.tmp = 0
        self.block = "entry"
        self.loops = 0
        self.data = set(low.data)
        # data are constants of the module, array buffers live on the stack
        self.globals = [
            f"@{k} = private unnamed_addr constant [{len(v)} x double] "
            f"[{', '.join(f'double {self.const(float(x))}' for x in v)}]"
            for k, v in low.data.items()
        ]
        self.allocas = [
            f"  %{k} = alloca [{n} x double]" for k, n in low.arrays.items()
        ]

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

    def _ptr(self, array: str, idx: Index) -> str:
        reg, offset = idx
        if reg is None:
            i = str(offset)
        elif offset == 0:
            i = reg
        else:
            i = self._new(f"add i64 {reg}, {offset}")
        base = f"@{array}" if array in self.data else f"%{array}"
        n = self.lengths[array]
        return self._new(f"getelementptr [{n} x double], ptr {base}, i64 0, i64 {i}")

    def load(self, array: str, idx: Index) -> str:
        return self._new(f"load double, ptr {self._ptr(array, idx)}")

    def store(self, array: str, idx: Index, value: str) -> None:
        self.lines.append(f"  store double {value}, ptr {self._ptr(array, idx)}")

    def loop(
        self,
        n: int,
        inits: list[str],
        body: Callable[[str, list[str]], list[str]],
    ) -> list[str]:
        self.loops += 1
        label, pred = f"L{self.loops}", self.block
        counter = f"%{label}.i"
        accs = [f"%{label}.acc{j}" for j in range(len(inits))]
        self.lines += [f"  br label %{label}", f"{label}:"]
        self.block = label
        phis = len(self.lines)
        nexts = body(counter, accs)
        if self.block != label:
            msg = "internal error: a loop body must stay in one block"
            raise RuntimeError(msg)
        self.lines[phis:phis] = [
            f"  {counter} = phi i64 [0, %{pred}], [{counter}.next, %{label}]",
            *(
                f"  {acc} = phi double [{init}, %{pred}], [{nxt}, %{label}]"
                for acc, init, nxt in zip(accs, inits, nexts, strict=True)
            ),
        ]
        self.lines += [
            f"  {counter}.next = add i64 {counter}, 1",
            f"  %{label}.more = icmp ult i64 {counter}.next, {n}",
            f"  br i1 %{label}.more, label %{label}, label %{label}.end",
            f"{label}.end:",
        ]
        self.block = f"{label}.end"
        return nexts


def to_llvm_ir(low: Lowered) -> str:
    """Emit `void fcn(int n, double t, double *y, double *dydt, double *pars)` as LLVM IR."""
    em = _Emit(low)
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
        if name in low.arrays:
            em.fill(name, expr)
        else:
            em.env[name] = em.value(expr)
        em.memo.clear()
    for i, expr in enumerate(low.dydt):
        v = em.value(expr)
        em.lines.append(f"  %out{i} = getelementptr double, ptr %dydt, i64 {i}")
        em.lines.append(f"  store double {v}, ptr %out{i}")
    return "\n".join(
        [
            *sorted(em.decls),
            *em.globals,
            "define void @fcn(i32 %n, double %t, ptr %y, ptr %dydt, ptr %pars) {",
            "entry:",
            *em.allocas,
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
    lib = mxlpy_native.load()
    lib.mxn_set_model.argtypes = [ct.c_void_p]
    lib.mxn_set_model.restype = None
    lib.mxn_eval_rows.restype = None
    lib.mxn_eval_rows.argtypes = [
        ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, _DP, _DP, _DP, _DP,
    ]  # fmt: skip
    lib.mxn_radau.restype = ct.c_int
    lib.mxn_radau.argtypes = [
        ct.c_int, _DP, _DP, ct.c_double, ct.c_double, ct.c_int, ct.c_int, ct.c_int, _DP, _DP,
    ]  # fmt: skip
    lib.mxn_lsoda.restype = ct.c_int
    lib.mxn_lsoda.argtypes = [
        ct.c_int, _DP, _DP, ct.c_double, ct.c_double, ct.c_int, ct.c_int, _DP, _DP,
    ]  # fmt: skip
    lib.mxn_cvode.restype = ct.c_int
    lib.mxn_cvode.argtypes = [
        ct.c_int, _DP, _DP, ct.c_double, ct.c_double, ct.c_int, ct.c_int, ct.c_int, _DP, _DP,
    ]  # fmt: skip
    return lib


def _target_machine() -> llvm.TargetMachine:
    _initialise()
    return llvm.Target.from_triple(llvm.get_default_triple()).create_target_machine(
        opt=2
    )


class _Compiled:
    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        # The engine takes ownership of its target machine, so they can't be shared.
        self._engine = llvm.create_mcjit_compiler(
            llvm.parse_assembly(""), _target_machine()
        )
        self._engine.add_object_file(llvm.ObjectFileRef.from_data(blob))
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

    def eval_rows(
        self, t: np.ndarray, y: np.ndarray, pars: np.ndarray, n_out: int
    ) -> np.ndarray:
        t, y = np.array(t, dtype=float), np.array(y, dtype=float, order="C")
        out = np.empty((len(t), n_out))
        self._lib.mxn_eval_rows(
            self._ptr, y.shape[1], n_out, len(t), t.ctypes.data_as(_DP), y.ctypes.data_as(_DP),
            out.ctypes.data_as(_DP), pars.ctypes.data_as(_DP),
        )  # fmt: skip
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
        method: str,
        reuse_jacobian: bool,
    ) -> tuple[int, np.ndarray]:
        y = np.array(y0, dtype=float)
        out = np.empty((len(t_out), len(y)))
        head = (
            len(y),
            y.ctypes.data_as(_DP),
            pars.ctypes.data_as(_DP),
            rtol,
            atol,
            nmax,
        )
        tail = (len(t_out), t_out.ctypes.data_as(_DP), out.ctypes.data_as(_DP))
        self._lib.mxn_set_model(self._ptr)
        if method == "LSODA":
            idid = self._lib.mxn_lsoda(*head, *tail)
        elif method.startswith("CVODE"):
            idid = self._lib.mxn_cvode(*head, int(method == "CVODE_Adams"), *tail)
        else:
            idid = self._lib.mxn_radau(*head, int(reuse_jacobian), *tail)
        return int(idid), out


def compile_model(low: Lowered) -> _Compiled:
    """Compile a lowered model to machine code."""
    tm = _target_machine()
    mod = llvm.parse_assembly(to_llvm_ir(low))
    mod.verify()
    pto = llvm.create_pipeline_tuning_options(speed_level=2)
    pb = llvm.create_pass_builder(tm, pto)
    pb.getModulePassManager().run(mod, pb)
    return _Compiled(tm.emit_object(mod))


def load(blob: bytes) -> _Compiled:
    """Load the machine code of a `compile_model` result, e.g. in another process."""
    return _Compiled(blob)
