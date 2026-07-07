"""Tests for the JAX ``Ode``/``FluxOde`` models.

Focus areas (all regressions that were previously broken):

* argument-vs-parameter ordering: the rhs receives ``concat(free_args, pars)``,
  matching the generator convention ``args = free_parameters + parameters_to_fit``;
* the generated ``nv`` is a single-argument ``flux_vector -> dydt`` map;
* ``FluxOde.from_mxlpy`` translates ``math.*`` to ``jnp.*``;
* conditional rate laws are emitted as ``jnp.where`` and are jit-integrable.
"""

import math

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
# conditional rate laws -> jnp.where, jit-integrable
# ---------------------------------------------------------------------------


def test_conditional_rate_generates_jnp_where_not_ternary() -> None:
    code = meta.generate_model_code_jax(_conditional_model())
    rate_line = next(
        line for line in code.model.splitlines() if line.strip().startswith("r =")
    )
    assert "jnp.where" in rate_line
    assert " if " not in rate_line
    assert "else" not in rate_line


def test_conditional_model_from_mxlpy_integrates_under_jit() -> None:
    import diffrax

    ode = Ode.from_mxlpy(_conditional_model())
    ts = jnp.array([0.0, 1.0])
    # would raise TracerBoolConversionError with the old Python-ternary codegen
    ys = ode.integrate(ts, jnp.array([1.0, 1.0]), jnp.array([]), method=diffrax.Tsit5)
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
    ys = ode.integrate(ts, jnp.array([1.0]), jnp.array([]), method=diffrax.Tsit5)
    assert ys.shape == (2, 1)
    assert bool(jnp.all(jnp.isfinite(ys)))
    # exponential decay from v0=1, k=1 -> v(1) ~ e^-1
    assert float(ys[-1, 0]) == pytest.approx(float(np.exp(-1.0)), rel=1e-3)


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
            jnp.array([0.0, 1.0]), jnp.array([1.0]), jnp.array([])
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
    sim = fode.simulate_time_course(ts, jnp.array([0.0]), jnp.array([5.0]))

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

    sim = fode.simulate_protocol_time_course(ts, y0, protocol)

    variables = sim.variables
    assert np.allclose(variables.index.to_numpy(), [0.0, 1.0, 2.0, 3.0])
    # v(1)=31, v(2)=31+32=63, v(3)=63+33=96 (dv/dt = a + 30 per step)
    assert np.allclose(
        variables["v"].to_numpy(), [0.0, 31.0, 63.0, 96.0], atol=1e-4
    )

    fluxes = sim.fluxes
    # t=0 row uses the first step's args (a=1); each subsequent row uses the
    # args active during its own step's window
    assert np.allclose(fluxes["ra"].to_numpy(), [1.0, 1.0, 2.0, 3.0])
    assert np.allclose(fluxes["rb"].to_numpy(), 3.0)


def test_integrate_to_steady_state_returns_time_and_state() -> None:
    ode = Ode.from_mxlpy(_single_variable_model())
    t, y = ode.integrate_to_steady_state(jnp.array([1.0]), jnp.array([]))
    assert t.shape == ()
    assert float(t) > 0.0
    assert y.shape == (1,)
    assert float(y[0]) == pytest.approx(0.0, abs=1e-4)


def test_fluxode_simulate_to_steady_state() -> None:
    fode = FluxOde.from_mxlpy(_single_variable_model())
    sim = fode.simulate_to_steady_state(jnp.array([1.0]), jnp.array([]))

    variables = sim.variables
    assert variables.shape == (1, 1)
    assert float(variables.index[0]) > 0.0
    assert float(variables["v"].iloc[0]) == pytest.approx(0.0, abs=1e-4)

    fluxes = sim.fluxes
    assert list(fluxes.columns) == ["r"]
    assert float(fluxes["r"].iloc[0]) == pytest.approx(0.0, abs=1e-4)
