"""Tests for the JAX export module."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("diffrax")

import jax
import jax.numpy as jnp

from mxlpy import Model, Simulator, fns
from mxlpy.jax import JaxExport, to_jax_export

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _simple_model() -> Model:
    """Single variable, single parameter, single reaction."""
    return (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 0.1)
        .add_reaction("v", fns.mass_action_1s, stoichiometry={"x": -1}, args=["x", "k"])
    )


def _two_var_model() -> Model:
    """Two variables, two parameters, one reaction."""
    return (
        Model()
        .add_variables({"x": 1.0, "y": 0.0})
        .add_parameters({"k1": 0.5, "k2": 0.2})
        .add_reaction(
            "v1",
            fns.mass_action_1s,
            stoichiometry={"x": -1, "y": 1},
            args=["x", "k1"],
        )
        .add_reaction(
            "v2",
            fns.mass_action_1s,
            stoichiometry={"y": -1},
            args=["y", "k2"],
        )
    )


def _model_with_derived() -> Model:
    return (
        Model()
        .add_variables({"x": 2.0, "y": 1.0})
        .add_parameter("k", 1.0)
        .add_derived("total", fns.add, args=["x", "y"])
        .add_reaction(
            "v",
            fns.mass_action_1s,
            stoichiometry={"x": -1, "y": 1},
            args=["x", "k"],
        )
    )


# ---------------------------------------------------------------------------
# Metadata / structure tests
# ---------------------------------------------------------------------------


def test_returns_jax_export() -> None:
    export = to_jax_export(_simple_model())
    assert isinstance(export, JaxExport)


def test_variable_names_match_model() -> None:
    model = _simple_model()
    export = to_jax_export(model)
    assert export.variable_names == list(model.get_initial_conditions().keys())


def test_parameter_names_match_model() -> None:
    model = _simple_model()
    export = to_jax_export(model)
    assert export.parameter_names == list(model.get_parameter_values().keys())


def test_initial_conditions_match_model() -> None:
    model = _simple_model()
    export = to_jax_export(model)
    assert export.initial_conditions == model.get_initial_conditions()


def test_parameter_values_match_model() -> None:
    model = _simple_model()
    export = to_jax_export(model)
    assert export.parameter_values == model.get_parameter_values()


def test_has_surrogates_false_for_pure_model() -> None:
    assert not to_jax_export(_simple_model()).has_surrogates


# ---------------------------------------------------------------------------
# get_y0 / get_args helpers
# ---------------------------------------------------------------------------


def test_get_y0_shape() -> None:
    export = to_jax_export(_simple_model())
    y0 = export.get_y0()
    assert y0.shape == (1,)


def test_get_y0_values() -> None:
    export = to_jax_export(_simple_model())
    y0 = export.get_y0()
    np.testing.assert_allclose(y0, [1.0])


def test_get_args_shape() -> None:
    export = to_jax_export(_simple_model())
    args = export.get_args()
    assert args.shape == (1,)


def test_get_args_default_values() -> None:
    export = to_jax_export(_simple_model())
    args = export.get_args()
    np.testing.assert_allclose(args, [0.1])


def test_get_args_override() -> None:
    export = to_jax_export(_simple_model())
    args = export.get_args(k=99.0)
    idx = export.parameter_names.index("k")
    np.testing.assert_allclose(float(args[idx]), 99.0)


def test_get_args_two_params() -> None:
    model = _two_var_model()
    export = to_jax_export(model)
    args = export.get_args()
    assert args.shape == (2,)
    idx_k1 = export.parameter_names.index("k1")
    idx_k2 = export.parameter_names.index("k2")
    np.testing.assert_allclose(float(args[idx_k1]), 0.5)
    np.testing.assert_allclose(float(args[idx_k2]), 0.2)


# ---------------------------------------------------------------------------
# RHS correctness
# ---------------------------------------------------------------------------


def test_rhs_output_shape() -> None:
    export = to_jax_export(_two_var_model())
    y0 = export.get_y0()
    args = export.get_args()
    result = export.rhs(0.0, y0, args)
    assert result.shape == (2,)


def test_rhs_values_match_model() -> None:
    """rhs(t, y0, args) should equal model(t, y0)."""
    model = _two_var_model()
    export = to_jax_export(model)
    y0 = export.get_y0()
    args = export.get_args()
    jax_result = export.rhs(0.0, y0, args)
    model_result = np.array(model(0.0, np.array(y0)))
    np.testing.assert_allclose(np.array(jax_result), model_result, rtol=1e-5)


def test_rhs_with_derived_matches_model() -> None:
    model = _model_with_derived()
    export = to_jax_export(model)
    y0 = export.get_y0()
    args = export.get_args()
    jax_result = export.rhs(0.0, y0, args)
    model_result = np.array(model(0.0, np.array(y0)))
    np.testing.assert_allclose(np.array(jax_result), model_result, rtol=1e-5)


def test_rhs_with_overridden_params() -> None:
    model = _simple_model()
    export = to_jax_export(model)
    y0 = export.get_y0()
    args = export.get_args(k=2.0)
    jax_result = float(export.rhs(0.0, y0, args)[0])
    # dx/dt = -k * x = -2.0 * 1.0 = -2.0
    np.testing.assert_allclose(jax_result, -2.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# JAX transformations
# ---------------------------------------------------------------------------


def test_jit_compatible() -> None:
    export = to_jax_export(_simple_model())
    y0 = export.get_y0()
    args = export.get_args()
    jit_rhs = jax.jit(export.rhs)
    result = jit_rhs(0.0, y0, args)
    assert result.shape == (1,)


def test_grad_wrt_args() -> None:
    export = to_jax_export(_simple_model())
    y0 = export.get_y0()
    args = export.get_args()

    def loss(a: jnp.ndarray) -> jnp.ndarray:
        return export.rhs(0.0, y0, a).sum()

    grad = jax.grad(loss)(args)
    assert grad.shape == args.shape


def test_vmap_over_args() -> None:
    export = to_jax_export(_simple_model())
    y0 = export.get_y0()
    # Batch of 5 different parameter vectors
    batch_args = jnp.stack(
        [export.get_args(k=float(k)) for k in [0.1, 0.2, 0.5, 1.0, 2.0]]
    )

    batched_rhs = jax.vmap(export.rhs, in_axes=(None, None, 0))
    results = batched_rhs(0.0, y0, batch_args)
    assert results.shape == (5, 1)


# ---------------------------------------------------------------------------
# simulate helper
# ---------------------------------------------------------------------------


def test_simulate_shapes() -> None:
    export = to_jax_export(_two_var_model())
    y0 = export.get_y0()
    args = export.get_args()
    times, values = export.simulate(y0, 0.0, 5.0, args, steps=50)
    assert times.shape == (50,)
    assert values.shape == (50, 2)


def test_simulate_time_range() -> None:
    export = to_jax_export(_simple_model())
    y0 = export.get_y0()
    args = export.get_args()
    times, _ = export.simulate(y0, 0.0, 3.0, args, steps=30)
    np.testing.assert_allclose(float(times[0]), 0.0, atol=1e-6)
    np.testing.assert_allclose(float(times[-1]), 3.0, atol=1e-6)


def test_simulate_custom_time_points() -> None:
    export = to_jax_export(_simple_model())
    y0 = export.get_y0()
    args = export.get_args()
    tp = jnp.array([0.0, 1.0, 2.0, 5.0])
    times, values = export.simulate(y0, 0.0, 5.0, args, time_points=tp)
    assert times.shape == (4,)
    assert values.shape == (4, 1)


def test_simulate_matches_scipy() -> None:
    """JAX simulation should produce results close to the SciPy baseline."""
    model = _two_var_model()
    export = to_jax_export(model)
    y0 = export.get_y0()
    args = export.get_args()

    t_end = 5.0
    steps = 100
    _, jax_values = export.simulate(y0, 0.0, t_end, args, steps=steps)

    sim = Simulator(model)
    sim.simulate(t_end=t_end, steps=steps)
    assert sim.variables is not None
    scipy_df = sim.variables[-1]  # DataFrame: rows=time, cols=variables

    # Compare final state (the solvers use different tolerances so allow 1e-3)
    scipy_final = scipy_df.iloc[-1].to_numpy()
    np.testing.assert_allclose(
        np.array(jax_values[-1]),
        scipy_final,
        rtol=1e-3,
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unparseable_function_raises() -> None:
    """A rate function that cannot be parsed to SymPy should raise ValueError."""

    def bad_rate(x: float) -> float:
        # np.interp is unknown to the SymPy AST parser
        return float(np.interp(x, [0.0, 1.0], [0.0, 1.0]))

    model = (
        Model()
        .add_variable("x", 1.0)
        .add_reaction("v", bad_rate, stoichiometry={"x": -1}, args=["x"])
    )
    with pytest.raises(ValueError, match="Unable to parse"):
        to_jax_export(model)
