"""Tests for the JAX ``Ode``/``FluxOde`` models.

Focus areas (all regressions that were previously broken):

* argument-vs-parameter ordering: the rhs receives ``concat(free_args, pars)``,
  matching the generator convention ``args = free_parameters + parameters_to_fit``;
* the generated ``nv`` is a single-argument ``flux_vector -> dydt`` map;
* ``FluxOde.from_mxlpy`` translates ``math.*`` to ``jnp.*``;
* conditional rate laws are emitted as ``jnp.select`` and are jit-integrable;
* special functions (``min``/``max``, trig, ``sign``, ``erf``/``gamma``/...)
  are translated to their ``jnp``/``jax.scipy.special`` equivalents.
"""

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mxlpy import KineticModelBuilder, meta
from mxlpy.jax.models import FluxOde, Ode

# ---------------------------------------------------------------------------
# helper models
# ---------------------------------------------------------------------------


def _rate_a(a: float) -> float:
    return a


def _rate_b(b: float) -> float:
    return b


def _distinguishable_model() -> KineticModelBuilder:
    """``dv/dt = 1*a + 10*b`` so free (``a``) and fit (``b``) params are distinct.

    Swapping the two would change the numeric result, so this model detects any
    misordering between the free (runtime) args and the trainable ``pars``.
    """
    return (
        KineticModelBuilder()
        .add_variable("v", initial_value=0.0)
        .add_parameter("a", value=2.0)
        .add_parameter("b", value=3.0)
        .add_reaction("ra", fn=_rate_a, args=["a"], stoichiometry={"v": 1.0})
        .add_reaction("rb", fn=_rate_b, args=["b"], stoichiometry={"v": 10.0})
    )


def _derived_ab(a: float, b: float) -> float:
    return a + 100.0 * b


def _distinguishable_model_with_derived() -> KineticModelBuilder:
    """Adds a ``d = a + 100*b`` derived value to :func:`_distinguishable_model`.

    ``a``/``b``'s wildly different coefficients make any mix-up between
    which value lands in the free-arg slot vs. the trainable-par slot
    numerically obvious.
    """
    return _distinguishable_model().add_derived("d", fn=_derived_ab, args=["a", "b"])


def _cond_rate(v: float, thr: float) -> float:
    if v > thr:
        return v
    return 0.0


def _conditional_model() -> KineticModelBuilder:
    # Two variables so the generated code unpacks scalars (single-variable models
    # have an unrelated shape quirk in the current codegen).
    return (
        KineticModelBuilder()
        .add_variable("v", initial_value=1.0)
        .add_variable("w", initial_value=1.0)
        .add_parameter("thr", value=0.5)
        .add_reaction(
            "r",
            fn=_cond_rate,
            args=["v", "thr"],
            stoichiometry={"v": -1.0, "w": 1.0},
        )
    )


def _exp_rate(v: float) -> float:
    return math.exp(-v)


def _exp_model() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable("v", initial_value=1.0)
        .add_reaction("r", fn=_exp_rate, args=["v"], stoichiometry={"v": -1.0})
    )


def _sqrt_rate(v: float) -> float:
    return math.sqrt(v)


def _log_rate(v: float) -> float:
    return math.log(v)


def _sin_rate(v: float) -> float:
    return math.sin(v)


def _floor_rate(v: float) -> float:
    return math.floor(v)


def _ceil_rate(v: float) -> float:
    return math.ceil(v)


def _sign_rate(v: float) -> float:
    return np.sign(v)


def _erf_rate(v: float) -> float:
    return math.erf(v)


def _gamma_rate(v: float) -> float:
    return math.gamma(v)


def _loggamma_rate(v: float) -> float:
    return math.lgamma(v)


def _min_rate(v: float, w: float) -> float:
    return min(v, w)


def _max_rate(v: float, w: float) -> float:
    return max(v, w)


def _unary_fn_model(
    fn: Callable[[float], float], initial_value: float
) -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable("v", initial_value=initial_value)
        .add_reaction("r", fn=fn, args=["v"], stoichiometry={"v": -1.0})
    )


def _min_max_model(
    fn: Callable[[float, float], float], v0: float, w0: float
) -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_variable("v", initial_value=v0)
        .add_parameter("w", value=w0)
        .add_reaction("r", fn=fn, args=["v", "w"], stoichiometry={"v": -1.0})
    )


def _decay(v: float, k: float) -> float:
    return k * v


def _single_variable_model() -> KineticModelBuilder:
    """Exponential decay ``dv/dt = -k*v`` with a single state variable."""
    return (
        KineticModelBuilder()
        .add_variable("v", initial_value=1.0)
        .add_parameter("k", value=1.0)
        .add_reaction("r", fn=_decay, args=["v", "k"], stoichiometry={"v": -1.0})
    )


def _rate_source(a: float) -> float:
    return a


def _source_decay_model() -> KineticModelBuilder:
    """``dv/dt = a - k*v``; nonzero steady state ``v_ss = a/k`` depends on ``k``.

    Unlike :func:`_single_variable_model` (whose steady state is always 0,
    regardless of ``k``), this model's steady state actually shifts with the
    trainable parameter -- needed to distinguish "pre-equilibration
    contributes gradient" from "pre-equilibration contributes none".
    """
    return (
        KineticModelBuilder()
        .add_variable("v", initial_value=0.0)
        .add_parameter("a", value=6.0)
        .add_parameter("k", value=2.0)
        .add_reaction("source", fn=_rate_source, args=["a"], stoichiometry={"v": 1.0})
        .add_reaction("decay", fn=_decay, args=["v", "k"], stoichiometry={"v": -1.0})
    )


# ---------------------------------------------------------------------------
# direct __call__ ordering (spy on the args the rhs actually receives)
# ---------------------------------------------------------------------------


def test_ode_call_appends_pars_after_free_args() -> None:
    captured: dict[str, jnp.ndarray] = {}

    def rhs(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:
        captured["args"] = args
        return y

    ode = Ode(rhs=rhs, pars=jnp.array([10.0, 20.0]))
    ode(0.0, jnp.array([1.0]), jnp.array([1.0, 2.0, 3.0]))

    # free/runtime args first, trainable pars last
    assert [float(x) for x in captured["args"]] == [1.0, 2.0, 3.0, 10.0, 20.0]


def test_ode_call_with_empty_pars_passes_args_unchanged() -> None:
    captured: dict[str, jnp.ndarray] = {}

    def rhs(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:
        captured["args"] = args
        return y

    ode = Ode(rhs=rhs, pars=jnp.array([]))
    ode(0.0, jnp.array([1.0]), jnp.array([7.0, 8.0]))

    assert [float(x) for x in captured["args"]] == [7.0, 8.0]


def test_fluxode_call_appends_pars_after_free_args() -> None:
    captured: dict[str, jnp.ndarray] = {}

    def fluxes(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:
        captured["args"] = args
        return jnp.zeros_like(y)

    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    fo = FluxOde(fluxes=fluxes, nv=nv, pars=jnp.array([10.0, 20.0]))
    fo(0.0, jnp.array([1.0]), jnp.array([1.0, 2.0, 3.0]))

    assert [float(x) for x in captured["args"]] == [1.0, 2.0, 3.0, 10.0, 20.0]


# ---------------------------------------------------------------------------
# generator convention: args = free_parameters + parameters_to_fit
# ---------------------------------------------------------------------------


def test_generated_model_unpacks_free_then_fit() -> None:
    code = meta.generate_model_code_jax(
        _distinguishable_model(),
        parameters_to_fit=["b"],
        free_parameters=["a"],
    )
    unpack = next(line for line in code.model.splitlines() if line.endswith("= args"))
    assert unpack.strip() == "a, b = args"


# ---------------------------------------------------------------------------
# end-to-end: from_mxlpy wires free args and pars into the right slots
# ---------------------------------------------------------------------------


def test_ode_from_mxlpy_pars_hold_only_fit_parameters() -> None:
    ode = Ode.from_mxlpy(
        _distinguishable_model(),
        parameters_to_fit=["b"],
        free_parameters=["a"],
    )
    # the trainable vector holds the *fit* parameter (b=3), not the free one
    assert [float(x) for x in ode.pars] == [3.0]


def test_ode_from_mxlpy_uses_runtime_free_arg_and_fit_pars() -> None:
    ode = Ode.from_mxlpy(
        _distinguishable_model(),
        parameters_to_fit=["b"],
        free_parameters=["a"],
    )
    # dv/dt = a + 10*b. Pass free a=5 at call time; fit b=3 lives in pars.
    dvdt = float(ode(0.0, jnp.array([0.0]), jnp.array([5.0]))[0])
    assert dvdt == 35.0  # a misordering would give 3 + 10*5 = 53


def test_ode_and_fluxode_from_mxlpy_agree() -> None:
    m = _distinguishable_model()
    ode = Ode.from_mxlpy(m, parameters_to_fit=["b"], free_parameters=["a"])
    fode = FluxOde.from_mxlpy(m, parameters_to_fit=["b"], free_parameters=["a"])
    y = jnp.array([0.0])
    args = jnp.array([5.0])
    assert np.allclose(np.asarray(ode(0.0, y, args)), np.asarray(fode(0.0, y, args)))


# ---------------------------------------------------------------------------
# from_mxlpy wires up the generated derived function, with free args and
# fit pars landing in the same order derived() actually unpacks them
# ---------------------------------------------------------------------------


def test_ode_from_mxlpy_wires_derived_fn_in_correct_arg_order() -> None:
    ode = Ode.from_mxlpy(
        _distinguishable_model_with_derived(),
        parameters_to_fit=["b"],
        free_parameters=["a"],
        derived_to_calculate=["d"],
    )
    # a=5 (free/call-time), b=3 (fit/default) -> d = 5 + 100*3 = 305.
    # A swapped concat order would instead compute 3 + 100*5 = 503.
    d = float(ode.derived(0.0, jnp.array([0.0]), jnp.array([5.0]))[0])
    assert d == pytest.approx(305.0)


def test_fluxode_from_mxlpy_wires_derived_fn_in_correct_arg_order() -> None:
    fode = FluxOde.from_mxlpy(
        _distinguishable_model_with_derived(),
        parameters_to_fit=["b"],
        free_parameters=["a"],
        derived_to_calculate=["d"],
    )
    d = float(fode.derived(0.0, jnp.array([0.0]), jnp.array([5.0]))[0])
    assert d == pytest.approx(305.0)


def test_ode_derived_scale_multiplies_derived_fn_output() -> None:
    def rhs(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:  # noqa: ARG001
        return y

    def derived_fn(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:  # noqa: ARG001
        return jnp.array([7.0])

    ode = Ode(
        rhs=rhs, pars=jnp.array([]), derived_fn=derived_fn, derived_scale=jnp.array(2.0)
    )
    d = float(ode.derived(0.0, jnp.array([1.0]), jnp.array([]))[0])
    assert d == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# generated nv is a single-argument flux -> dydt map
# ---------------------------------------------------------------------------


def test_generated_nv_takes_single_flux_argument() -> None:
    code = meta.generate_model_code_jax(_distinguishable_model())
    nv_sig = next(line for line in code.nv.splitlines() if line.startswith("def nv("))
    assert nv_sig == "def nv(fluxes: jax.Array) -> jax.Array:"


# ---------------------------------------------------------------------------
# FluxOde.from_mxlpy must translate math.* -> jnp.* (regression: NameError math)
# ---------------------------------------------------------------------------


def test_fluxode_from_mxlpy_translates_math_to_jnp() -> None:
    m = _exp_model()
    ode = Ode.from_mxlpy(m)
    fode = FluxOde.from_mxlpy(m)
    y = jnp.array([1.0])
    args = jnp.array([])
    # before the fix, evaluating fode raised NameError: name 'math' is not defined
    assert np.allclose(np.asarray(ode(0.0, y, args)), np.asarray(fode(0.0, y, args)))


# ---------------------------------------------------------------------------
# special functions (min/max, trig, sign, erf/gamma/...) must round-trip
# through the jax backend with the same numeric result as plain Python
# ---------------------------------------------------------------------------

_UNARY_SPECIAL_FUNCTION_CASES: list[
    tuple[str, Callable[[float], float], float, Callable[[float], float]]
] = [
    ("sqrt", _sqrt_rate, 4.0, math.sqrt),
    ("log", _log_rate, 2.0, math.log),
    ("sin", _sin_rate, 0.5, math.sin),
    ("floor", _floor_rate, 2.7, math.floor),
    ("ceiling", _ceil_rate, 2.3, math.ceil),
    ("sign", _sign_rate, -3.0, lambda v: float(np.sign(v))),
    ("erf", _erf_rate, 0.5, math.erf),
    ("gamma", _gamma_rate, 2.5, math.gamma),
    ("loggamma", _loggamma_rate, 2.5, math.lgamma),
    # factorial is excluded here: mxlpy eagerly calls rate fns with real
    # floats during model construction (derived-parameter caching), and
    # math.factorial rejects non-integral floats regardless of backend.
    # Its jax translation is covered at the string level in
    # tests/meta/test_sympy_tools.py::test_unary_jax_translation.
]


@pytest.mark.parametrize(
    ("name", "fn", "v0", "reference"),
    _UNARY_SPECIAL_FUNCTION_CASES,
    ids=[case[0] for case in _UNARY_SPECIAL_FUNCTION_CASES],
)
def test_unary_special_function_jax_export_matches_reference(
    name: str,
    fn: Callable[[float], float],
    v0: float,
    reference: Callable[[float], float],
) -> None:
    ode = Ode.from_mxlpy(_unary_fn_model(fn, v0))
    dvdt = float(ode(0.0, jnp.array([v0]), jnp.array([]))[0])
    assert dvdt == pytest.approx(-reference(v0), rel=1e-5), name


def test_min_rate_jax_export_matches_reference() -> None:
    v0, w0 = 2.0, 5.0
    ode = Ode.from_mxlpy(_min_max_model(_min_rate, v0, w0))
    dvdt = float(ode(0.0, jnp.array([v0]), jnp.array([]))[0])
    assert dvdt == pytest.approx(-min(v0, w0))


def test_max_rate_jax_export_matches_reference() -> None:
    v0, w0 = 2.0, 5.0
    ode = Ode.from_mxlpy(_min_max_model(_max_rate, v0, w0))
    dvdt = float(ode(0.0, jnp.array([v0]), jnp.array([]))[0])
    assert dvdt == pytest.approx(-max(v0, w0))


def test_min_and_sign_rates_are_jit_compatible() -> None:
    """Regression guard: functools.reduce(jnp.minimum, ...) and jnp.sign must
    trace under jax.jit, unlike the builtin min()/a bool-based sign ternary.

    ``ode`` instances are jitted via a wrapping function rather than
    ``jax.jit(ode)`` directly: jitting an eqx.Module instance itself makes
    jax hash it, which fails once it holds a concrete jnp.array field.
    """
    ode_min = Ode.from_mxlpy(_min_max_model(_min_rate, 2.0, 5.0))

    def call_min(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:
        return ode_min(t, y, args)

    dvdt_min = float(jax.jit(call_min)(0.0, jnp.array([2.0]), jnp.array([]))[0])
    assert dvdt_min == pytest.approx(-2.0)

    ode_sign = Ode.from_mxlpy(_unary_fn_model(_sign_rate, -3.0))

    def call_sign(t: float, y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:
        return ode_sign(t, y, args)

    dvdt_sign = float(jax.jit(call_sign)(0.0, jnp.array([-3.0]), jnp.array([]))[0])
    assert dvdt_sign == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# conditional rate laws -> jnp.select, jit-integrable
# ---------------------------------------------------------------------------


def test_conditional_rate_generates_jnp_select_not_ternary() -> None:
    code = meta.generate_model_code_jax(_conditional_model())
    rate_line = next(
        line for line in code.model.splitlines() if line.strip().startswith("r =")
    )
    assert "jnp.select" in rate_line
    assert " if " not in rate_line
    assert "else" not in rate_line


def test_conditional_model_from_mxlpy_integrates_under_jit() -> None:
    import diffrax

    ode = Ode.from_mxlpy(_conditional_model())
    ts = jnp.array([0.0, 1.0])
    # would raise TracerBoolConversionError with the old Python-ternary codegen
    ys = ode.integrate(
        ts, jnp.array([1.0, 1.0]), 8192, args=jnp.array([]), method=diffrax.Tsit5
    )
    assert bool(jnp.all(jnp.isfinite(ys)))


# ---------------------------------------------------------------------------
# single-variable models unpack a scalar and integrate with the right shape
# ---------------------------------------------------------------------------


def test_single_variable_model_unpacks_scalar_state() -> None:
    code = meta.generate_model_code_jax(_single_variable_model())
    # scalar unpacking via trailing comma, not ``v = variables`` (whole array)
    assert "    (v,) = variables" in code.model


def test_single_variable_model_integrates_with_correct_shape() -> None:
    import diffrax

    ode = Ode.from_mxlpy(_single_variable_model())
    ts = jnp.array([0.0, 1.0])
    # previously produced a (1, 1) rhs and failed to broadcast against y0 (1,)
    ys = ode.integrate(
        ts, jnp.array([1.0]), 8192, args=jnp.array([]), method=diffrax.Tsit5
    )
    assert ys.shape == (2, 1)
    assert bool(jnp.all(jnp.isfinite(ys)))
    # exponential decay from v0=1, k=1 -> v(1) ~ e^-1
    assert float(ys[-1, 0]) == pytest.approx(float(np.exp(-1.0)), rel=1e-3)


# ---------------------------------------------------------------------------
# args=None (the documented default on integrate/integrate_to_steady_state)
# must not crash: every JaxModel.__call__ unconditionally concatenates args
# with its own state, so a bare ``None`` used to blow up as soon as it
# reached diffrax's vector-field call.
# ---------------------------------------------------------------------------


def test_ode_integrate_defaults_args_to_empty_array() -> None:
    ode = Ode.from_mxlpy(_single_variable_model())
    ts = jnp.array([0.0, 1.0])
    ys = ode.integrate(ts, jnp.array([1.0]), 8192)  # no args= at all
    assert ys.shape == (2, 1)
    assert bool(jnp.all(jnp.isfinite(ys)))
    assert float(ys[-1, 0]) == pytest.approx(float(np.exp(-1.0)), rel=1e-3)


def test_ode_integrate_to_steady_state_defaults_args_to_empty_array() -> None:
    ode = Ode.from_mxlpy(_single_variable_model())
    t, y = ode.integrate_to_steady_state(jnp.array([1.0]), 8192)  # no args= at all
    assert t.shape == ()
    assert y.shape == (1,)
    assert float(y[0]) == pytest.approx(0.0, abs=1e-4)


def test_fluxode_simulate_time_course_defaults_args_to_empty_array() -> None:
    fode = FluxOde.from_mxlpy(_single_variable_model())
    sim = fode.simulate_time_course(jnp.array([0.0, 1.0]), jnp.array([1.0]), 8192)
    assert sim.variables.shape == (2, 1)


# ---------------------------------------------------------------------------
# FluxOde.simulate_* / FluxOdeSimulation
# ---------------------------------------------------------------------------


def test_fluxode_direct_construction_has_no_names() -> None:
    def fluxes(_t: float, y: jnp.ndarray, _args: jnp.ndarray) -> jnp.ndarray:
        return jnp.zeros_like(y)

    def nv(flux_vector: jnp.ndarray) -> jnp.ndarray:
        return flux_vector

    fo = FluxOde(fluxes=fluxes, nv=nv, pars=jnp.array([]))
    assert fo.variable_names is None
    assert fo.flux_names is None

    with pytest.raises(ValueError, match="variable_names"):
        fo.simulate_time_course(
            jnp.array([0.0, 1.0]), jnp.array([1.0]), 8192, args=jnp.array([])
        )


def test_fluxode_from_mxlpy_sets_names() -> None:
    fode = FluxOde.from_mxlpy(
        _distinguishable_model(),
        parameters_to_fit=["b"],
        free_parameters=["a"],
    )
    assert fode.variable_names == ("v",)
    assert fode.flux_names == ("ra", "rb")


def test_fluxode_simulate_time_course() -> None:
    # dv/dt = a + 10*b, with a free (=5) and b fit (=3) -> constant dv/dt=35
    fode = FluxOde.from_mxlpy(
        _distinguishable_model(),
        parameters_to_fit=["b"],
        free_parameters=["a"],
    )
    ts = jnp.array([0.0, 1.0, 2.0])
    sim = fode.simulate_time_course(ts, jnp.array([0.0]), 8192, args=jnp.array([5.0]))

    variables = sim.variables
    assert list(variables.columns) == ["v"]
    assert np.allclose(variables.index.to_numpy(), np.asarray(ts))
    assert np.allclose(variables["v"].to_numpy(), 35.0 * np.asarray(ts), atol=1e-4)

    fluxes = sim.fluxes
    assert list(fluxes.columns) == ["ra", "rb"]
    assert np.allclose(fluxes["ra"].to_numpy(), 5.0)
    assert np.allclose(fluxes["rb"].to_numpy(), 3.0)

    combined = sim.get_combined()
    assert list(combined.columns) == ["v", "ra", "rb"]


def test_fluxode_simulate_protocol_time_course_prepends_t0() -> None:
    # dv/dt = a + 10*b, b fit = 3; a switches 1 -> 2 -> 3 at t=1,2,3
    fode = FluxOde.from_mxlpy(
        _distinguishable_model(),
        parameters_to_fit=["b"],
        free_parameters=["a"],
    )
    ts = [jnp.array([1.0]), jnp.array([2.0]), jnp.array([3.0])]
    protocol = jnp.array([[1.0], [2.0], [3.0]])
    y0 = jnp.array([0.0])

    sim = fode.simulate_protocol_time_course(ts, y0, protocol, 8192)

    variables = sim.variables
    assert np.allclose(variables.index.to_numpy(), [0.0, 1.0, 2.0, 3.0])
    # v(1)=31, v(2)=31+32=63, v(3)=63+33=96 (dv/dt = a + 30 per step)
    assert np.allclose(variables["v"].to_numpy(), [0.0, 31.0, 63.0, 96.0], atol=1e-4)

    fluxes = sim.fluxes
    # t=0 row uses the first step's args (a=1); each subsequent row uses the
    # args active during its own step's window
    assert np.allclose(fluxes["ra"].to_numpy(), [1.0, 1.0, 2.0, 3.0])
    assert np.allclose(fluxes["rb"].to_numpy(), 3.0)


def test_integrate_to_steady_state_returns_time_and_state() -> None:
    ode = Ode.from_mxlpy(_single_variable_model())
    t, y = ode.integrate_to_steady_state(jnp.array([1.0]), 8192, args=jnp.array([]))
    assert t.shape == ()
    assert float(t) > 0.0
    assert y.shape == (1,)
    assert float(y[0]) == pytest.approx(0.0, abs=1e-4)


def test_fluxode_simulate_to_steady_state() -> None:
    fode = FluxOde.from_mxlpy(_single_variable_model())
    sim = fode.simulate_to_steady_state(jnp.array([1.0]), 8192, args=jnp.array([]))

    variables = sim.variables
    assert variables.shape == (1, 1)
    assert float(variables.index[0]) > 0.0
    assert float(variables["v"].iloc[0]) == pytest.approx(0.0, abs=1e-4)

    fluxes = sim.fluxes
    assert list(fluxes.columns) == ["r"]
    assert float(fluxes["r"].iloc[0]) == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# integrate_protocol is lax.scan-based internally; per-step save-point
# counts may still differ (ragged), padded/unpadded transparently
# ---------------------------------------------------------------------------


def test_fluxode_integrate_protocol_with_ragged_step_lengths() -> None:
    # dv/dt = a + 10*b, b fit = 3 (constant); a switches 1 -> 2 -> 3 per step.
    # Steps save a different number of points each (2, 1, 3) to exercise the
    # scan implementation's per-step padding/unpadding.
    fode = FluxOde.from_mxlpy(
        _distinguishable_model(),
        parameters_to_fit=["b"],
        free_parameters=["a"],
    )
    ts = [
        jnp.array([0.5, 1.0]),
        jnp.array([1.5]),
        jnp.array([2.0, 2.5, 3.0]),
    ]
    protocol = jnp.array([[1.0], [2.0], [3.0]])
    y0 = jnp.array([0.0])

    ys = fode.integrate_protocol(ts, y0, protocol, 8192)
    assert ys.shape == (6, 1)

    # slope per step = a + 30 (10*b): 31, 32, 33; each step continues from
    # the previous step's last saved value, at its own last saved time.
    expected = jnp.array([15.5, 31.0, 47.0, 63.5, 80.0, 96.5])
    assert np.allclose(np.asarray(ys[:, 0]), np.asarray(expected), atol=1e-3)


# ---------------------------------------------------------------------------
# integrate_protocol_from_steady_state composes integrate_to_steady_state
# and integrate_protocol
# ---------------------------------------------------------------------------


def test_ode_integrate_protocol_from_steady_state_preequilibrates_first() -> None:
    ode = Ode.from_mxlpy(_single_variable_model())
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    protocol = jnp.zeros((2, 0))
    y0 = jnp.array([5.0])  # far from the steady state (0)

    ys = ode.integrate_protocol_from_steady_state(ts, y0, protocol, 8192)
    assert ys.shape == (2, 1)
    # Already pre-equilibrated to ~0 before the protocol starts, so the
    # trajectory stays near zero throughout -- not decaying from y0=5 as it
    # would if the protocol ran directly from y0 without pre-equilibrating.
    assert np.allclose(np.asarray(ys[:, 0]), 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# integrate_protocol_from_steady_state(freeze_preeq_gradient=True): the
# pre-equilibration solve runs identically, but its sensitivity to the
# model's trainable parameters is cut from the gradient
# ---------------------------------------------------------------------------


def test_freeze_preeq_gradient_does_not_change_forward_value() -> None:
    ode = Ode.from_mxlpy(
        _source_decay_model(), parameters_to_fit=["k"], free_parameters=["a"]
    )
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    protocol = jnp.array([[6.0], [6.0]])
    y0 = jnp.array([0.0])

    ys_normal = ode.integrate_protocol_from_steady_state(ts, y0, protocol, 8192)
    ys_frozen = ode.integrate_protocol_from_steady_state(
        ts, y0, protocol, 8192, freeze_preeq_gradient=True
    )
    # stop_gradient is the identity on forward values -- only backward
    # sensitivity should differ, never the actual trajectory.
    assert np.allclose(np.asarray(ys_normal), np.asarray(ys_frozen))


def test_freeze_preeq_gradient_cuts_preeq_path_sensitivity() -> None:
    ode = Ode.from_mxlpy(
        _source_decay_model(), parameters_to_fit=["k"], free_parameters=["a"]
    )
    ts = [jnp.array([1.0]), jnp.array([2.0])]
    protocol = jnp.array([[6.0], [6.0]])
    y0 = jnp.array([0.0])

    def _loss(pars: jax.Array, *, freeze: bool) -> jax.Array:
        m = eqx.tree_at(lambda m: m.pars, ode, pars)
        return jnp.sum(
            m.integrate_protocol_from_steady_state(
                ts, y0, protocol, 8192, freeze_preeq_gradient=freeze
            )
        )

    grad_normal = jax.grad(lambda p: _loss(p, freeze=False))(ode.pars)
    grad_frozen = jax.grad(lambda p: _loss(p, freeze=True))(ode.pars)

    assert bool(jnp.all(jnp.isfinite(grad_normal)))
    assert bool(jnp.all(jnp.isfinite(grad_frozen)))
    # The frozen gradient is missing the preeq-path sensitivity (how k
    # shifts the steady state itself), so it must differ from the full one.
    assert not np.allclose(np.asarray(grad_normal), np.asarray(grad_frozen))

    # It should match *exactly* differentiating integrate_protocol alone,
    # seeded from a steady state fixed to a concrete (non-differentiable)
    # value -- i.e. only the preeq-path sensitivity was cut, nothing more.
    _, y_ss = ode.integrate_to_steady_state(y0, 8192, args=protocol[0])

    def _protocol_only_loss(pars: jax.Array) -> jax.Array:
        m = eqx.tree_at(lambda m: m.pars, ode, pars)
        return jnp.sum(m.integrate_protocol(ts, y_ss, protocol, 8192))

    grad_manual = jax.grad(_protocol_only_loss)(ode.pars)
    assert np.allclose(np.asarray(grad_frozen), np.asarray(grad_manual), atol=1e-6)
