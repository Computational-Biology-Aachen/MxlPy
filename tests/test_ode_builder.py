"""Regression coverage for OdeModelBuilder's dx/dt computation.

This class had zero test coverage before this file and, until fixed here,
its core evaluation path (`__call__`/`_get_args`) never actually called a
diff_eq's rate law at all — `ModelCache.dyn_order` only ever included a
diff_eq's own name when that variable's *initial value* happened to be
dynamic (an unrelated case), so `_get_right_hand_side` silently returned
the input concentration unchanged instead of a derivative. A second,
related bug in an intermediate fix attempt: naively evaluating diff_eqs
via the same in-place-mutating `calculate_inpl` pattern `Derived` uses is
wrong for diff_eqs specifically, since one diff_eq's evaluation would
overwrite `args[var]` with *its* derivative — and a second diff_eq that
legitimately needs `var`'s *concentration* as an input would then read the
wrong value. The fix evaluates every diff_eq from one frozen
pre-evaluation snapshot instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mxlpy import OdeModelBuilder, Simulator
from mxlpy.surrogates.abstract_derivative import AbstractDerivativeSurrogate

if TYPE_CHECKING:
    import pandas as pd


@dataclass(kw_only=True)
class _MockDerivativeSurrogate(AbstractDerivativeSurrogate):
    """A minimal, deterministic `AbstractDerivativeSurrogate` for testing `OdeModelBuilder.add_surrogate`."""

    correction: float

    def predict(
        self,
        args: dict[str, float | pd.Series | pd.DataFrame],  # noqa: ARG002, for API compatibility
    ) -> dict[str, float]:
        return dict.fromkeys(self.targets, self.correction)


def test_single_variable_dxdt_is_actually_computed() -> None:
    model = OdeModelBuilder().add_parameter("k", 0.5).add_diff_eq(
        "x", fn=lambda x, k: -k * x, args=["x", "k"], initial_value=2.0
    )
    dxdt = model(0.0, [2.0])
    assert dxdt == (-1.0,)


def test_coupled_system_reads_concentrations_not_derivatives() -> None:
    # dy/dt depends on x's *concentration* — if x's own diff_eq ran first
    # and overwrote args["x"] with x's derivative, dy/dt would silently
    # use the wrong value here.
    model = (
        OdeModelBuilder()
        .add_parameter("k1", 0.5)
        .add_parameter("k2", 0.3)
        .add_diff_eq("x", fn=lambda x, k1: -k1 * x, args=["x", "k1"], initial_value=2.0)
        .add_diff_eq(
            "y",
            fn=lambda x, y, k1, k2: k1 * x - k2 * y,
            args=["x", "y", "k1", "k2"],
            initial_value=1.0,
        )
    )
    dxdt, dydt = model(0.0, [2.0, 1.0])
    assert dxdt == -1.0
    assert dydt == 0.5 * 2.0 - 0.3 * 1.0


def test_diff_eq_depending_on_a_dynamic_derived_quantity() -> None:
    model = (
        OdeModelBuilder()
        .add_parameter("k", 0.5)
        .add_derived("rate", fn=lambda x, k: k * x, args=["x", "k"])
        .add_diff_eq("x", fn=lambda rate: -rate, args=["rate"], initial_value=2.0)
    )
    assert model(0.0, [2.0]) == (-1.0,)


def test_full_simulation_matches_analytic_exponential_decay() -> None:
    model = OdeModelBuilder().add_parameter("k", 0.5).add_diff_eq(
        "x", fn=lambda x, k: -k * x, args=["x", "k"], initial_value=2.0
    )
    # Simulator's type hint is still KineticModelBuilder-only (widening it
    # pulls in other KineticModelBuilder-only code paths inside Simulator
    # itself — to_symbolic_model, get_dependent — that's real, separate
    # scope, not touched here); this works correctly at runtime already.
    result = Simulator(model).simulate(10.0).get_result().unwrap_or_err()  # type: ignore[arg-type]
    df = result.get_variables(
        include_derived_variables=False,
        include_readouts=False,
        include_surrogate_variables=False,
    )
    t_final = df.index[-1]
    analytic = 2.0 * np.exp(-0.5 * t_final)
    np.testing.assert_allclose(df["x"].iloc[-1], analytic, rtol=1e-4)


def test_get_variable_names_matches_get_diff_eq_names() -> None:
    model = OdeModelBuilder().add_diff_eq(
        "x", fn=lambda x: -x, args=["x"], initial_value=1.0
    )
    assert model.get_variable_names() == model.get_diff_eq_names() == ["x"]


def test_add_surrogate_composes_onto_dxdt() -> None:
    model = (
        OdeModelBuilder()
        .add_parameter("k", 0.5)
        .add_diff_eq("x", fn=lambda x, k: -k * x, args=["x", "k"], initial_value=2.0)
        .add_surrogate(
            "corr", _MockDerivativeSurrogate(args=[], targets=["x"], correction=0.1)
        )
    )
    (dxdt,) = model(0.0, [2.0])
    # mechanistic term (-0.5 * 2.0 = -1.0) + surrogate correction (0.1),
    # default mechanism is "additive".
    assert dxdt == pytest.approx(-1.0 + 0.1)


def test_add_surrogate_with_unknown_target_raises_and_does_not_leak_the_name() -> None:
    """A failed `add_surrogate` (bad `targets`) must not permanently reserve `name` in the id namespace.

    Regression test: `add_surrogate` used to call `_insert_id` before
    validating `surrogate.targets`, so a validation failure left `name`
    registered in `self._ids` with no matching entry in
    `self._surrogates` — an unrecoverable half-added state (the name
    became permanently unusable, and `remove_surrogate` couldn't clean it
    up either, since it pops from `self._surrogates`, which never
    contained it).
    """
    model = OdeModelBuilder().add_diff_eq(
        "x", fn=lambda x: -x, args=["x"], initial_value=1.0
    )
    bad_surrogate = _MockDerivativeSurrogate(args=[], targets=["y"], correction=0.0)
    with pytest.raises(KeyError, match="y"):
        model.add_surrogate("corr", bad_surrogate)

    assert "corr" not in model.ids
    assert model.get_raw_surrogates() == {}

    # The name must be fully reusable afterwards.
    good_surrogate = _MockDerivativeSurrogate(args=[], targets=["x"], correction=0.1)
    model.add_surrogate("corr", good_surrogate)
    assert "corr" in model.get_raw_surrogates()


def test_remove_surrogate_removes_correction() -> None:
    model = (
        OdeModelBuilder()
        .add_diff_eq("x", fn=lambda x: -x, args=["x"], initial_value=1.0)
        .add_surrogate(
            "corr", _MockDerivativeSurrogate(args=[], targets=["x"], correction=0.1)
        )
    )
    with_correction = model(0.0, [1.0])
    model.remove_surrogate("corr")
    without_correction = model(0.0, [1.0])

    assert with_correction != without_correction
    assert without_correction == (-1.0,)
    assert model.get_raw_surrogates() == {}
