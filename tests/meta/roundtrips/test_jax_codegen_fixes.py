"""Regression tests for jax codegen dependency-closure filtering.

``derived``/``fluxes``/``model`` used to share one unfiltered list of
"everything topologically before the readouts", so a derived term or
readout dependency that only fed a readout (or wasn't requested via
``derived_to_calculate``) still showed up as a dead local variable in
``fluxes``/``model``.

Also covers the generated ``functools`` import being unconditional even
when nothing in the generated code actually needs ``functools.reduce``,
and the ``time`` parameter of ``model``/``derived``/``fluxes`` being
named ``ts`` in the signature while the sympy symbol substituted into
the body is ``time``, so rate laws that explicitly depend on time
referenced an undefined name.

Also covers two silent-wrong-answer bugs in the Python-source-to-sympy
conversion (as opposed to the dependency-closure/import/naming issues
above, which produced code that errored or included dead code):

- An ``if``/``else`` chain that assigns local variables in each branch,
  combined by a statement *after* the chain, used to have that statement
  dropped -- the chain's last-assigned symbol per branch was taken as the
  function's whole return value. ``_v_alpha_vde``-shaped (assign two
  branch-local variables, multiply them together afterwards) and
  ``_frbss_r``-shaped (assign one branch-local variable, clip it with an
  outer ``min()`` afterwards) bugs from mxlmodels' lam2026/morales2018 both
  come from this.
- ``x == y`` / ``x != y`` conditions used sympy's structural ``__eq__``
  (a plain Python bool) instead of ``sympy.Eq``/``sympy.Ne`` (a symbolic
  relational), so e.g. ``if ppfd == 0`` silently compiled to "always take
  the else branch" -- correct for callers who never pass exactly 0, but
  wrong in general.
"""

import jax.numpy as jnp

from mxlpy import KineticModelBuilder, meta


def one_argument(x: float) -> float:
    return x


def two_arguments(x: float, y: float) -> float:
    return x * y


def clip_at_zero(x: float) -> float:
    return max(x, 0.0)


def scale_by_time(x: float, time: float) -> float:
    return x * time


def light_dependent_rate(
    light: float, k_light: float, k_dark: float, factor_light: float, s: float
) -> float:
    """`_v_alpha_vde`-shaped: if/else assigns 2 locals, combined afterwards."""
    if light == 0:
        k = k_dark
        factor = 1.0 - factor_light
    else:
        k = k_light
        factor = factor_light
    return k * s * factor


def clipped_branch_rate(x: float, threshold: float, lo: float, hi: float, cap: float) -> float:
    """`_frbss_r`-shaped: if/else assigns 1 local, clipped by a trailing statement."""
    if x <= threshold:
        val = lo
    else:
        val = hi
    return min(val, cap)


def rate_at_exactly_zero(x: float, k_zero: float, k_nonzero: float) -> float:
    return k_zero if x == 0 else k_nonzero


def test_unrequested_readout_dependencies_are_not_computed() -> None:
    """A readout not in ``derived_to_calculate`` must not leak into fluxes/model.

    ``pq_ratio`` depends on ``p_unused``, a parameter no reaction needs.
    Requesting only ``"other"`` must drop both from every function.
    """
    model = (
        KineticModelBuilder()
        .add_variable("v1", initial_value=1.0)
        .add_parameter("p1", value=1.0)
        .add_parameter("p_unused", value=2.0)
        .add_reaction(
            "r1",
            fn=two_arguments,
            stoichiometry={"v1": -1.0},
            args=["v1", "p1"],
        )
        .add_derived("other", fn=one_argument, args=["v1"])
        .add_readout("pq_ratio", fn=one_argument, args=["p_unused"])
    )

    codegen = meta.generate_model_code_jax(
        model,
        free_parameters=["p1"],
        derived_to_calculate=["other"],
    )

    assert "p_unused" not in codegen.derived
    assert "pq_ratio" not in codegen.derived
    assert "p_unused" not in codegen.fluxes
    assert "p_unused" not in codegen.model


def test_derived_term_unused_by_any_reaction_is_dropped_from_fluxes_and_model() -> None:
    """A plain derived term unused by any reaction shouldn't show up outside ``derived``."""
    model = (
        KineticModelBuilder()
        .add_variable("v1", initial_value=1.0)
        .add_parameter("p_unused", value=2.0)
        .add_derived("d_unused", fn=one_argument, args=["p_unused"])
    )

    codegen = meta.generate_model_code_jax(model)

    assert "p_unused" in codegen.derived
    assert "d_unused" in codegen.derived
    assert "p_unused" not in codegen.fluxes
    assert "p_unused" not in codegen.model


def test_functools_import_omitted_when_unused() -> None:
    model = (
        KineticModelBuilder()
        .add_variable("v1", initial_value=1.0)
        .add_parameter("p1", value=1.0)
        .add_reaction(
            "r1",
            fn=two_arguments,
            stoichiometry={"v1": -1.0},
            args=["v1", "p1"],
        )
    )

    codegen = meta.generate_model_code_jax(model)

    assert "functools" not in codegen.imports


def test_functools_import_present_when_reduce_is_emitted() -> None:
    model = (
        KineticModelBuilder()
        .add_variable("v1", initial_value=1.0)
        .add_reaction(
            "r1",
            fn=clip_at_zero,
            stoichiometry={"v1": -1.0},
            args=["v1"],
        )
    )

    codegen = meta.generate_model_code_jax(model)

    assert "functools.reduce" in codegen.fluxes
    assert "import functools" in codegen.imports


def test_time_parameter_resolves_in_generated_source() -> None:
    """A rate law that explicitly depends on ``time`` must not reference an undefined name."""
    model = (
        KineticModelBuilder()
        .add_variable("v1", initial_value=1.0)
        .add_reaction(
            "r1",
            fn=scale_by_time,
            stoichiometry={"v1": -1.0},
            args=["v1", "time"],
        )
    )

    codegen = meta.generate_model_code_jax(model)

    assert "def fluxes(time: jax.Array" in codegen.fluxes
    assert "ts" not in codegen.fluxes

    namespace: dict[str, object] = {}
    exec(codegen.imports, namespace)  # noqa: S102
    exec(codegen.fluxes, namespace)  # noqa: S102

    (r1,) = namespace["fluxes"](jnp.array(3.0), jnp.array([2.0]), jnp.array([]))
    assert r1 == 6.0


def test_if_else_assign_then_combine_survives_codegen() -> None:
    """`k * s * factor` after the if/else must use *both* branch-local variables.

    Before the fix this collapsed to a single branch's value (e.g. always
    `k_light * s * factor_light`, dropping `k`/`factor` entirely).
    """
    model = (
        KineticModelBuilder()
        .add_variable("v1", initial_value=1.0)
        .add_parameter("light", value=0.0)
        .add_parameter("k_light", value=2.0)
        .add_parameter("k_dark", value=5.0)
        .add_parameter("factor_light", value=0.5)
        .add_reaction(
            "r1",
            fn=light_dependent_rate,
            stoichiometry={"v1": -1.0},
            args=["light", "k_light", "k_dark", "factor_light", "v1"],
        )
    )

    codegen = meta.generate_model_code_jax(model, free_parameters=["light"])

    namespace: dict[str, object] = {}
    exec(codegen.imports, namespace)  # noqa: S102
    exec(codegen.fluxes, namespace)  # noqa: S102
    fluxes = namespace["fluxes"]

    # light == 0 -> k = k_dark = 5, factor = 1 - 0.5 = 0.5 -> r1 = 5 * 3 * 0.5
    (r1_dark,) = fluxes(jnp.array(0.0), jnp.array([3.0]), jnp.array([0.0]))
    assert r1_dark == 7.5

    # light != 0 -> k = k_light = 2, factor = 0.5 -> r1 = 2 * 3 * 0.5
    (r1_light,) = fluxes(jnp.array(0.0), jnp.array([3.0]), jnp.array([250.0]))
    assert r1_light == 3.0


def test_if_else_then_trailing_clip_survives_codegen() -> None:
    """`min(val, cap)` after the if/else must survive on both branches."""
    model = (
        KineticModelBuilder()
        .add_variable("v1", initial_value=1.0)
        .add_parameter("x", value=0.0)
        .add_parameter("threshold", value=0.0)
        .add_parameter("lo", value=1.0)
        .add_parameter("hi", value=100.0)
        .add_parameter("cap", value=10.0)
        .add_reaction(
            "r1",
            fn=clipped_branch_rate,
            stoichiometry={"v1": -1.0},
            args=["x", "threshold", "lo", "hi", "cap"],
        )
    )

    codegen = meta.generate_model_code_jax(model, free_parameters=["x"])

    namespace: dict[str, object] = {}
    exec(codegen.imports, namespace)  # noqa: S102
    exec(codegen.fluxes, namespace)  # noqa: S102
    fluxes = namespace["fluxes"]

    # x <= threshold -> val = lo = 1, min(1, 10) = 1
    (r1_lo,) = fluxes(jnp.array(0.0), jnp.array([1.0]), jnp.array([-1.0]))
    assert r1_lo == 1.0

    # x > threshold -> val = hi = 100, min(100, 10) = 10 (clip must survive)
    (r1_hi,) = fluxes(jnp.array(0.0), jnp.array([1.0]), jnp.array([5.0]))
    assert r1_hi == 10.0


def test_equality_condition_creates_real_piecewise_branch() -> None:
    """`if x == 0` must compile to a real branch, not always take the else."""
    model = (
        KineticModelBuilder()
        .add_variable("v1", initial_value=1.0)
        .add_parameter("x", value=0.0)
        .add_parameter("k_zero", value=9.0)
        .add_parameter("k_nonzero", value=3.0)
        .add_reaction(
            "r1",
            fn=rate_at_exactly_zero,
            stoichiometry={"v1": -1.0},
            args=["x", "k_zero", "k_nonzero"],
        )
    )

    codegen = meta.generate_model_code_jax(model, free_parameters=["x"])

    namespace: dict[str, object] = {}
    exec(codegen.imports, namespace)  # noqa: S102
    exec(codegen.fluxes, namespace)  # noqa: S102
    fluxes = namespace["fluxes"]

    (r1_zero,) = fluxes(jnp.array(0.0), jnp.array([1.0]), jnp.array([0.0]))
    assert r1_zero == 9.0

    (r1_nonzero,) = fluxes(jnp.array(0.0), jnp.array([1.0]), jnp.array([5.0]))
    assert r1_nonzero == 3.0
