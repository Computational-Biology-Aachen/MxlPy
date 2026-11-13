# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Array-valued derived quantities (mxl-schemas ADR 0015).

The shared fixture model has state-dependent arrays (attenuated light per
wavelength) that are reduced before the right-hand side. Every evaluation path
has to reproduce its reference right-hand side and exact Jacobians.
"""

from __future__ import annotations

import importlib.util
import json
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import scipy
import scipy.integrate
import sympy

from mxlpy import Model, Scipy, Simulator
from mxlpy.serialize import (
    ARRAY_SPEC_VERSION,
    DATA_SPEC_VERSION,
    model_from_dict,
    model_to_dict,
)
from mxlpy.symbolic import to_symbolic_model
from mxlpy.types import SerializationError

if TYPE_CHECKING:
    from mxlpy import KineticModelBuilder

ASSETS = Path(__file__).parent / "assets"
FIXTURES = json.loads(
    (ASSETS / "array-models.fixtures.json").read_text(encoding="utf-8")
)
UPSTREAM = (
    "https://raw.githubusercontent.com/Computational-Biology-Aachen/mxl-schemas/main"
)
META_SCHEMAS = Path(__file__).resolve().parents[3] / "misc" / "mxl-schemas"
REL = 1e-10

BACKENDS = [
    pytest.param(
        "llvm",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("llvmlite") is None
            or importlib.util.find_spec("mxlpy_native") is None,
            reason="llvm backend not installed",
        ),
    ),
    pytest.param(
        "wasm",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("wasmtime") is None,
            reason="wasm backend not installed",
        ),
    ),
]


def _model() -> KineticModelBuilder:
    return model_from_dict(FIXTURES["model"], data_file=FIXTURES["data"])


def _close(got: float, expected: float) -> bool:
    return abs(got - expected) <= REL * abs(expected) + 1e-300


def test_vendored_fixtures_match_mxl_schemas() -> None:
    rel = "tests/array-models.fixtures.json"
    local = META_SCHEMAS / rel
    if local.exists():
        canonical = local.read_text(encoding="utf-8")
    else:
        try:
            with urllib.request.urlopen(f"{UPSTREAM}/{rel}", timeout=10) as resp:  # noqa: S310
                canonical = resp.read().decode("utf-8")
        except OSError as exc:
            pytest.skip(f"canonical {rel} unavailable: {exc}")
    assert json.loads(canonical) == FIXTURES


@pytest.mark.parametrize("point", FIXTURES["points"])
def test_python_rhs_and_derived(point: dict[str, Any]) -> None:
    model = _model()
    rhs = model.get_right_hand_side(point["variables"])
    for name, expected in point["rhs"].items():
        assert _close(rhs[name], expected), name
    args = model.get_args(point["variables"])
    for name, expected in point["derived"].items():
        assert _close(args[name], expected), name


def test_array_derived_are_not_result_columns() -> None:
    model = _model()
    args = model.get_args()
    assert "I_loc" not in args.index
    assert "k_att" not in args.index
    res = Simulator(model).simulate(0.01).get_result().unwrap_or_err()
    assert "I_loc" not in res.get_args().columns
    assert "abs1" in res.get_variables(include_derived_variables=True).columns


@pytest.mark.parametrize("point", FIXTURES["points"])
def test_symbolic_jacobians(point: dict[str, Any]) -> None:
    """`sympy.diff` through the array nodes gives the exact Jacobians."""
    sm = to_symbolic_model(_model())
    data = {k: np.asarray(v["values"]) for k, v in FIXTURES["data"].items()}
    values = {sm.variables[k]: v for k, v in point["variables"].items()} | {
        sm.parameters[k]: v for k, v in sm.parameter_values.items()
    }
    for eq, var in zip(sm.eqs, sm.variables, strict=True):
        for kind, wrt in (("dfdy", sm.variables), ("dfdp", sm.parameters)):
            for name, sym in wrt.items():
                d = sympy.diff(eq, sym).subs(values)
                syms = sorted(d.free_symbols, key=str)
                fn = sympy.lambdify(
                    syms, d, modules=[{"scipy": scipy, "numpy": np}, "numpy"]
                )
                got = float(fn(*(data[str(s)] for s in syms)))
                assert _close(got, point[kind][var][name]), (kind, var, name)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("point", FIXTURES["points"])
def test_native_rhs(backend: str, point: dict[str, Any]) -> None:
    from mxlpy.integrators._native import llvm_backend, wasm_backend
    from mxlpy.integrators._native.lower import lower

    model = _model()
    low = lower(model)
    compiled = {"llvm": llvm_backend, "wasm": wasm_backend}[backend].compile_model(low)
    pars = model.get_parameter_values()
    got = compiled.rhs(
        0.0,
        np.array([point["variables"][v] for v in model.get_variable_names()]),
        np.array([pars[p] for p in low.par_names]),
    )
    for name, value in zip(model.get_variable_names(), got, strict=True):
        assert _close(value, point["rhs"][name]), name


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_simulation_matches_scipy(backend: str) -> None:
    from mxlpy import Native

    points = np.linspace(0, 0.002, 11)
    tol = {"rtol": 1e-10, "atol": 1e-12}
    expected = (
        Simulator(_model(), integrator=lambda *a, **k: Scipy(*a, **k, **tol))
        .simulate_time_course(points)
        .get_result()
        .unwrap_or_err()
    )
    got = (
        Simulator(
            _model(), integrator=lambda *a, **k: Native(*a, **k, backend=backend, **tol)
        )
        .simulate_time_course(points)
        .get_result()
        .unwrap_or_err()
    )
    np.testing.assert_allclose(
        got.variables.to_numpy(), expected.variables.to_numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(
        got.fluxes.to_numpy(), expected.fluxes.to_numpy(), rtol=1e-5
    )


@pytest.mark.parametrize("case", FIXTURES["errors"], ids=lambda c: c["description"])
def test_invalid_models_are_rejected(case: dict[str, Any]) -> None:
    from mxlpy.integrators._native.lower import (
        NativeLoweringError,
        lower,
    )

    with pytest.raises((SerializationError, NativeLoweringError)):
        lower(model_from_dict(case["model"], data_file=case["data"]))


def test_save_writes_spec_1_3_and_round_trips() -> None:
    model = _model()
    doc = model_to_dict(model, model_id="m")
    assert doc["spec_version"] == ARRAY_SPEC_VERSION
    point = FIXTURES["points"][1]
    rhs = model_from_dict(doc, data_file=FIXTURES["data"]).get_right_hand_side(
        point["variables"]
    )
    for name, expected in point["rhs"].items():
        assert _close(rhs[name], expected), name


def _absorbed(spectrum: pd.Series, wavelength: pd.Series) -> float:
    return scipy.integrate.simpson(spectrum, x=wavelength)


def _consumption(x: float, light: float) -> float:
    return light * x


def test_data_only_constants_stay_spec_1_2() -> None:
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_data("wavelength", pd.Series(np.linspace(400.0, 700.0, 7)))
        .add_data("spectrum", pd.Series(np.ones(7)))
        .add_derived("light", _absorbed, args=["spectrum", "wavelength"])
        .add_reaction("v", _consumption, args=["x", "light"], stoichiometry={"x": -1})
    )
    assert model_to_dict(model, model_id="m")["spec_version"] == DATA_SPEC_VERSION
