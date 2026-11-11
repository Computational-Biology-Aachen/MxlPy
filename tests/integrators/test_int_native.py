# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Native must give the same answers as Scipy on both backends."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sympy
from scipy.integrate import simpson

import mxlpy
from mxlpy import Model, OdeModelBuilder, Scipy, Simulator, scan
from mxlpy.fns import constant, mass_action_1s, mass_action_2s
from mxlpy.runners import KineticRunner
from mxlpy.types import IntegrationFailure, NoSteadyState, Result

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

METHODS = ["Radau", "LSODA", "CVODE_BDF", "CVODE_Adams"]

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("wasmtime") is None
    and importlib.util.find_spec("llvmlite") is None,
    reason="no Native backend installed",
)


def _hill(x: float, vmax: float, k: float, n: float) -> float:
    return vmax * x**n / (k**n + x**n)


def _clipped(x: float, limit: float) -> float:
    return max(0.0, min(x, limit))


def _time_drive(time: float, a: float) -> float:
    return a * np.exp(-0.1 * time)


def lotka() -> Model:
    return (
        Model()
        .add_variables({"x": 2.0, "y": 1.0})
        .add_parameters({"a": 1.1, "b": 0.4, "c": 0.4, "d": 0.1})
        .add_reaction("birth", mass_action_1s, stoichiometry={"x": 1}, args=["x", "a"])
        .add_reaction("eaten", mass_action_2s, stoichiometry={"x": -1}, args=["x", "y", "b"])
        .add_reaction("death", mass_action_1s, stoichiometry={"y": -1}, args=["y", "c"])
        .add_reaction("grow", mass_action_2s, stoichiometry={"y": 1}, args=["x", "y", "d"])
    )  # fmt: skip


def chain() -> Model:
    """Decays to a steady state, using derived values, powers, and min/max."""
    return (
        Model()
        .add_variables({"s": 5.0, "p": 0.0})
        .add_parameters({"vmax": 2.0, "k": 0.5, "n": 3.0, "out": 0.3, "limit": 4.0, "feed": 0.2})
        .add_derived("s_clip", _clipped, args=["s", "limit"])
        .add_reaction("in", constant, stoichiometry={"s": 1}, args=["feed"])
        .add_reaction("conv", _hill, stoichiometry={"s": -1, "p": 1}, args=["s_clip", "vmax", "k", "n"])
        .add_reaction("loss", mass_action_1s, stoichiometry={"p": -1}, args=["p", "out"])
    )  # fmt: skip


def robertson() -> Model:
    """A classic stiff problem."""
    return (
        Model()
        .add_variables({"a": 1.0, "b": 0.0, "c": 0.0})
        .add_parameters({"k1": 0.04, "k2": 3e7, "k3": 1e4})
        .add_reaction("r1", mass_action_1s, stoichiometry={"a": -1, "b": 1}, args=["a", "k1"])
        .add_reaction("r2", mass_action_2s, stoichiometry={"b": -1, "c": 1}, args=["b", "b", "k2"])
        .add_reaction("r3", mass_action_2s, stoichiometry={"a": 1, "b": -1}, args=["b", "c", "k3"])
    )  # fmt: skip


def driven() -> Model:
    """The right-hand side depends on time."""
    return (
        Model()
        .add_variables({"x": 0.0})
        .add_parameters({"a": 1.0})
        .add_reaction("drive", _time_drive, stoichiometry={"x": 1}, args=["time", "a"])
    )


def _native(backend: Any, **kwargs: object) -> partial:
    return partial(mxlpy.Native, backend=backend, **kwargs)


def _tight_scipy() -> partial:
    return partial(Scipy, method="Radau", rtol=1e-11, atol=1e-11)


def _values(sim: Simulator) -> tuple[np.ndarray, np.ndarray]:
    res = sim.get_result().unwrap_or_err().variables
    return res.index.to_numpy(), res.to_numpy()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("factory", "t_end"),
    [(lotka, 20), (chain, 50), (robertson, 100), (driven, 10)],
)
def test_matches_scipy(backend: Any, method: str, factory, t_end: float) -> None:  # noqa: ANN001
    _, expected = _values(
        Simulator(factory(), integrator=_tight_scipy()).simulate(t_end)
    )
    time, got = _values(
        Simulator(
            factory(),
            integrator=_native(backend, method=method, rtol=1e-10, atol=1e-10),
        ).simulate(t_end)
    )
    assert time[-1] == pytest.approx(t_end)
    assert got == pytest.approx(expected, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", METHODS)
def test_requested_time_points_are_returned_exactly(backend: Any, method: str) -> None:
    points = np.array([0.0, 0.1, 0.25, 3.0, 7.5, 20.0])
    sim = Simulator(
        lotka(), integrator=_native(backend, method=method, rtol=1e-10, atol=1e-10)
    )
    time, got = _values(sim.simulate_time_course(points))
    _, expected = _values(
        Simulator(lotka(), integrator=_tight_scipy()).simulate_time_course(points)
    )
    assert time == pytest.approx(points)
    assert got == pytest.approx(expected, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", METHODS)
def test_continues_across_calls(backend: Any, method: str) -> None:
    integrator = _native(backend, method=method, rtol=1e-10, atol=1e-10)
    sim = Simulator(lotka(), integrator=integrator)
    sim.simulate(5).simulate(10)
    time, got = _values(sim)
    _, expected = _values(Simulator(lotka(), integrator=integrator).simulate(10))
    assert time[-1] == pytest.approx(10)
    assert got[-1] == pytest.approx(expected[-1], rel=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_reset(backend: Any) -> None:
    model = lotka()
    integrator = mxlpy.Native(model, (2.0, 1.0), backend=backend)
    first = integrator.integrate(t_end=5).unwrap_or_err()
    integrator.reset()
    second = integrator.integrate(t_end=5).unwrap_or_err()
    assert second.time == pytest.approx(first.time)
    assert np.asarray(second.values) == pytest.approx(np.asarray(first.values))


@pytest.mark.parametrize("backend", BACKENDS)
def test_parameter_update_applies_without_recompiling(backend: Any) -> None:
    from mxlpy.integrators import int_native  # noqa: PLC0415

    model = chain()
    sim = Simulator(model, integrator=_native(backend))
    key = (backend, int_native.fingerprint(model))
    compiled = int_native._CACHE[key]
    sim.update_parameter("out", 0.9)
    _, got = _values(sim.simulate(20))

    assert int_native._CACHE[(backend, int_native.fingerprint(model))] is compiled
    changed = chain().update_parameter("out", 0.9)
    _, expected = _values(Simulator(changed, integrator=_tight_scipy()).simulate(20))
    assert got[-1] == pytest.approx(expected[-1], rel=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_structure_change_recompiles(backend: Any) -> None:
    model = chain()
    integrator = mxlpy.Native(model, (5.0, 0.0), backend=backend)
    before = integrator.integrate(t_end=5).unwrap_or_err().values  # noqa: PD011
    model.add_reaction(
        "extra", mass_action_1s, stoichiometry={"p": -1}, args=["p", "feed"]
    )
    integrator.reset()
    after = integrator.integrate(t_end=5).unwrap_or_err().values  # noqa: PD011
    assert np.asarray(after)[-1, 1] < np.asarray(before)[-1, 1]


@pytest.mark.parametrize("backend", BACKENDS)
def test_cache_eviction_keeps_other_models_working(backend: Any) -> None:
    from mxlpy.integrators import int_native  # noqa: PLC0415

    kept = lotka()
    integrator = mxlpy.Native(kept, (2.0, 1.0), backend=backend)
    for i in range(int_native._CACHE_SIZE + 2):
        model = (
            Model()
            .add_variable(f"x{i}", 1.0)
            .add_parameter("k", 0.1)
            .add_reaction(
                "r", mass_action_1s, stoichiometry={f"x{i}": -1}, args=[f"x{i}", "k"]
            )
        )
        assert (
            mxlpy.Native(model, (1.0,), backend=backend).integrate(t_end=1).value
            is not None
        )
    assert integrator.integrate(t_end=1).unwrap_or_err().time[-1] == pytest.approx(1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_steady_state(backend: Any) -> None:
    got = (
        Simulator(chain(), integrator=_native(backend))
        .simulate_to_steady_state(tolerance=1e-8)
        .get_result()
        .unwrap_or_err()
        .variables.iloc[-1]
    )
    # Inflow balances loss: out * p = feed
    assert got["p"] == pytest.approx(0.2 / 0.3, rel=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_steady_state_reported(backend: Any) -> None:
    integrator = mxlpy.Native(lotka(), (2.0, 1.0), backend=backend)
    res = integrator.integrate_to_steady_state(
        tolerance=1e-12, rel_norm=False, oscillation_detector=None, max_steps=2
    )
    assert isinstance(res, Result)
    assert isinstance(res.value, NoSteadyState)


@pytest.mark.parametrize("backend", BACKENDS)
def test_reuse_jacobian_stays_correct(backend: Any) -> None:
    _, expected = _values(
        Simulator(robertson(), integrator=_tight_scipy()).simulate(100)
    )
    _, got = _values(
        Simulator(
            robertson(), integrator=_native(backend, reuse_jacobian=True)
        ).simulate(100)
    )
    assert got[-1] == pytest.approx(expected[-1], rel=1e-3, abs=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("factory", [lotka, chain])
def test_lsoda_matches_scipy_lsoda(backend: Any, factory) -> None:  # noqa: ANN001
    """Both run SciPy's C LSODA, so they agree far below the solver tolerance."""
    _, expected = _values(
        Simulator(
            factory(), integrator=partial(Scipy, method="LSODA", rtol=1e-8, atol=1e-8)
        ).simulate(20)
    )
    _, got = _values(
        Simulator(
            factory(),
            integrator=_native(backend, method="LSODA", rtol=1e-8, atol=1e-8),
        ).simulate(20)
    )
    assert got == pytest.approx(expected, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("factory", [lotka, robertson])
@pytest.mark.skipif(
    importlib.util.find_spec("assimulo") is None, reason="assimulo not installed"
)
def test_cvode_matches_assimulo_cvode(backend: Any, factory) -> None:  # noqa: ANN001
    """Both run SUNDIALS' CVODE with BDF, so they agree well below the solver tolerance."""
    from mxlpy import Assimulo  # noqa: PLC0415

    points = np.linspace(0, 20, 41)
    _, expected = _values(
        Simulator(
            factory(), integrator=partial(Assimulo, rtol=1e-8, atol=1e-8)
        ).simulate_time_course(points)
    )
    _, got = _values(
        Simulator(
            factory(),
            integrator=_native(backend, method="CVODE_BDF", rtol=1e-8, atol=1e-8),
        ).simulate_time_course(points)
    )
    assert got == pytest.approx(expected, rel=1e-5, abs=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_reuse_jacobian_needs_radau(backend: Any) -> None:
    with pytest.raises(ValueError, match="reuse_jacobian only applies to Radau"):
        mxlpy.Native(
            lotka(), (2.0, 1.0), backend=backend, method="LSODA", reuse_jacobian=True
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_unknown_method_raises(backend: Any) -> None:
    with pytest.raises(ValueError, match="Radau, LSODA, CVODE_BDF, CVODE_Adams"):
        mxlpy.Native(lotka(), (2.0, 1.0), backend=backend, method="BDF")  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", BACKENDS)
def test_non_lowerable_model_raises_clear_error(backend: Any) -> None:
    from mxlpy.integrators.int_native import NativeLoweringError  # noqa: PLC0415

    def opaque(x: float) -> float:
        return float(np.mean([x, x**2]))  # numpy can't take symbols

    model = (
        Model()
        .add_variable("x", 1.0)
        .add_reaction("r", opaque, stoichiometry={"x": -1}, args=["x"])
    )
    with pytest.raises(NativeLoweringError):
        mxlpy.Native(model, (1.0,), backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_plain_callable_raises_clear_error(backend: Any) -> None:
    from mxlpy.integrators.int_native import NativeLoweringError  # noqa: PLC0415

    with pytest.raises(NativeLoweringError, match="needs a model"):
        mxlpy.Native(lambda _t, y: y, (1.0,), backend=backend)  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", METHODS)
def test_blow_up_is_not_reported_as_success(backend: Any, method: str) -> None:
    """y' = y^2 reaches infinity at t = 1, so the solver must fail rather than return it."""
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 1.0)
        .add_reaction("r", mass_action_2s, stoichiometry={"x": 1}, args=["x", "x", "k"])
    )
    res = mxlpy.Native(model, (1.0,), backend=backend, method=method).integrate(t_end=2)
    assert isinstance(res.value, IntegrationFailure)


@pytest.mark.parametrize("backend", BACKENDS)
def test_decreasing_time_points_raise(backend: Any) -> None:
    integrator = mxlpy.Native(lotka(), (2.0, 1.0), backend=backend)
    with pytest.raises(ValueError, match="increasing"):
        integrator.integrate_time_course(time_points=np.array([0.0, 2.0, 1.0]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_pickles_for_workers(backend: Any) -> None:
    import dill  # noqa: PLC0415

    integrator = mxlpy.Native(lotka(), (2.0, 1.0), backend=backend)
    expected = integrator.integrate(t_end=1).unwrap_or_err().values  # noqa: PD011
    restored = dill.loads(
        dill.dumps(mxlpy.Native(lotka(), (2.0, 1.0), backend=backend))
    )
    got = restored.integrate(t_end=1).unwrap_or_err().values  # noqa: PD011
    assert np.asarray(got) == pytest.approx(np.asarray(expected))


@pytest.mark.parametrize("backend", BACKENDS)
def test_parallel_scan(backend: Any) -> None:
    import pandas as pd  # noqa: PLC0415

    pars = pd.DataFrame({"out": [0.2, 0.4, 0.8]})
    res = scan.steady_state(
        chain(),
        to_scan=pars,
        parallel=True,
        runner=KineticRunner(integrator=_native(backend)),
    )
    got = res.variables["p"].to_numpy()
    assert got == pytest.approx(0.2 / pars["out"].to_numpy(), rel=1e-4)


def test_auto_picks_an_installed_backend() -> None:
    integrator = mxlpy.Native(lotka(), (2.0, 1.0))
    assert integrator.backend in {"llvm", "wasm"}


def test_unavailable_without_backends() -> None:
    code = """
import sys
for m in ("llvmlite", "mxlpy_native", "wasmtime"):
    sys.modules[m] = None
import importlib.util
real = importlib.util.find_spec
importlib.util.find_spec = lambda n, *a: None if n in ("llvmlite", "mxlpy_native", "wasmtime") else real(n, *a)
import mxlpy
try:
    from mxlpy import Native
except ImportError as e:
    print("BLOCKED", e)
try:
    mxlpy.Native
except AttributeError as e:
    print("HINT", e)
"""
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "BLOCKED" in res.stdout
    assert "mxlpy[native]" in res.stdout
    assert "mxlpy[native-wasm]" in res.stdout


def _ratio(a: float, b: float) -> float:
    return a / b


def with_outputs() -> Model:
    """Has every kind of output a simulation can report."""
    return (
        chain()
        .add_parameter("scale", 2.0)
        .add_derived("doubled", mass_action_1s, args=["feed", "scale"])
        .add_derived("s_times_p", mass_action_2s, args=["s", "p", "scale"])
        .add_readout("ratio", _ratio, args=["s", "out"])
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_compiled_outputs_match_python(backend: Any) -> None:
    from mxlpy.integrators import int_native  # noqa: PLC0415

    model = with_outputs()
    res = (
        Simulator(model, integrator=_native(backend))
        .simulate(10)
        .get_result()
        .unwrap_or_err()
    )
    assert res.args_fn is not None
    got = res._compute_args()[0]
    python = res._python_args(res.raw_variables[0])
    assert list(got.columns) == list(python.columns)
    assert got.index.to_numpy() == pytest.approx(python.index.to_numpy())
    assert got.to_numpy() == pytest.approx(python.to_numpy(), rel=1e-10, abs=1e-12)
    key = ("outputs", backend, int_native.fingerprint(model))
    assert int_native._CACHE[key] is not None


@pytest.mark.parametrize("backend", BACKENDS)
def test_outputs_are_selectable_after_simulation(backend: Any) -> None:
    res = (
        Simulator(with_outputs(), integrator=_native(backend))
        .simulate(10)
        .get_result()
        .unwrap_or_err()
    )
    assert {"s", "p", "s_clip", "s_times_p", "ratio"} <= set(res.variables.columns)
    assert list(res.fluxes.columns) == ["in", "conv", "loss"]
    plain = res.get_variables(
        include_derived_variables=False,
        include_surrogate_variables=False,
        include_readouts=False,
    )
    assert list(plain.columns) == ["s", "p"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_outputs_use_each_segments_parameters(backend: Any) -> None:
    sim = Simulator(with_outputs(), integrator=_native(backend)).simulate(5)
    sim.update_parameter("scale", 3.0).simulate(10)
    res = sim.get_result().unwrap_or_err()
    got = pd.concat(res._compute_args())
    assert set(got["doubled"].round(6)) == {0.4, 0.6}
    python = []
    for v, p in zip(res.raw_variables, res.raw_parameters, strict=True):
        res.model.update_parameters(p)
        python.append(res._python_args(v))
    assert got.to_numpy() == pytest.approx(
        pd.concat(python).to_numpy(), rel=1e-10, abs=1e-12
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_uncompilable_readout_falls_back_to_python(backend: Any) -> None:
    def opaque(x: float) -> float:
        return float(np.mean([x, x**2]))

    model = chain().add_readout("opaque", opaque, args=["s"])
    res = (
        Simulator(model, integrator=_native(backend))
        .simulate(5)
        .get_result()
        .unwrap_or_err()
    )
    expected = [float(np.mean([s, s**2])) for s in res.raw_variables[0]["s"]]
    assert res.variables["opaque"].to_numpy() == pytest.approx(expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ode_models_with_the_same_names_do_not_share_code(backend: Any) -> None:
    def grow(x: float) -> float:
        return x

    def shrink(x: float) -> float:
        return -x

    results = []
    for fn in (grow, shrink):
        model = OdeModelBuilder().add_diff_eq("x", 1.0, fn, args=["x"])
        results.append(
            Simulator(model, integrator=_native(backend))
            .simulate(1)
            .get_result()
            .unwrap_or_err()
        )
    assert results[0].variables["x"].iloc[-1] == pytest.approx(np.e, rel=1e-5)
    assert results[1].variables["x"].iloc[-1] == pytest.approx(1 / np.e, rel=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sympy_expression_fns_with_module_calls(backend: Any) -> None:
    """Expression functions print as `math.exp`/`math.log`, which must lower (#202)."""
    x, k = sympy.symbols("x k")
    model = (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 1.0)
        .add_reaction(
            "v", k * sympy.exp(-x) + sympy.log(k + 1), stoichiometry={"x": -1}
        )
    )
    _, expected = _values(Simulator(model, integrator=_tight_scipy()).simulate(1.0))
    _, got = _values(
        Simulator(model, integrator=_native(backend, rtol=1e-10, atol=1e-10)).simulate(
            1.0
        )
    )
    assert got == pytest.approx(expected, rel=1e-5, abs=1e-6)


def _absorbed(intensity: float, spectrum: pd.Series, wavelength: pd.Series) -> float:
    return intensity * simpson(spectrum, x=wavelength)


def _light_consumption(x: float, light: float) -> float:
    return light * x


def _data_model(spectrum: np.ndarray) -> Model:
    wavelength = np.arange(400.0, 701.0)
    return (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("intensity", 1.0)
        .add_data("wavelength", pd.Series(wavelength))
        .add_data("spectrum", pd.Series(spectrum))
        .add_derived("light", _absorbed, args=["intensity", "spectrum", "wavelength"])
        .add_reaction(
            "v", _light_consumption, args=["x", "light"], stoichiometry={"x": -1}
        )
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_model_data_is_folded(backend: Any) -> None:
    """Array subtrees over data become constants, as in mxlweb-core (#201)."""
    model = _data_model(np.ones(301) / 300)
    sim = Simulator(model, integrator=_native(backend, rtol=1e-10, atol=1e-10))
    res = sim.simulate(1.0).get_result().unwrap_or_err()
    assert res.variables["x"].iloc[-1] == pytest.approx(np.exp(-1.0), rel=1e-6)
    assert res.fluxes["v"].iloc[-1] == pytest.approx(np.exp(-1.0), rel=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_changed_model_data_recompiles(backend: Any) -> None:
    model = _data_model(np.ones(301) / 300)
    Simulator(model, integrator=_native(backend)).simulate(1.0)
    model.update_data("spectrum", pd.Series(np.ones(301) / 150))
    res = (
        Simulator(model, integrator=_native(backend, rtol=1e-10, atol=1e-10))
        .simulate(1.0)
        .get_result()
        .unwrap_or_err()
    )
    assert res.variables["x"].iloc[-1] == pytest.approx(np.exp(-2.0), rel=1e-6)


def _state_weighted(x: float, spectrum: pd.Series, wavelength: pd.Series) -> float:
    return simpson(x * spectrum, x=wavelength)


def test_data_depending_on_state_is_rejected() -> None:
    from mxlpy.integrators._native.lower import (  # noqa: PLC0415
        NativeLoweringError,
        lower,
    )

    model = _data_model(np.ones(301) / 300).update_derived(
        "light", _state_weighted, args=["x", "spectrum", "wavelength"]
    )
    with pytest.raises(NativeLoweringError, match="only reference data and numbers"):
        lower(model)
