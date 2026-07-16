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
