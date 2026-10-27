# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from __future__ import annotations

import pandas as pd
import pytest

from mxlpy import SteadyStateModelBuilder
from mxlpy.types import InitialAssignment


def chained_model() -> SteadyStateModelBuilder:
    # Derived values are inserted out of dependency order on purpose
    return (
        SteadyStateModelBuilder()
        .add_parameter("k", 2.0)
        .add_parameter("k2", InitialAssignment(fn=lambda k: 3 * k, args=["k"]))
        .add_derived("d2", fn=lambda d1: d1 + 1, args=["d1"])
        .add_derived("d1", fn=lambda k, k2: k * k2, args=["k", "k2"])
    )


def test_evaluate_chained_derived() -> None:
    pd.testing.assert_series_equal(
        chained_model().evaluate(),
        pd.Series({"d2": 13.0, "d1": 12.0}),
    )


def test_evaluate_include_parameters() -> None:
    pd.testing.assert_series_equal(
        chained_model().evaluate(include_parameters=True),
        pd.Series({"k": 2.0, "k2": 6.0, "d2": 13.0, "d1": 12.0}),
    )


def test_evaluate_after_parameter_update() -> None:
    model = chained_model()
    model.evaluate()
    model.update_parameter("k", 1.0)
    pd.testing.assert_series_equal(
        model.evaluate(),
        pd.Series({"d2": 4.0, "d1": 3.0}),
    )


def test_evaluate_raises_on_arity_mismatch() -> None:
    model = (
        SteadyStateModelBuilder()
        .add_parameter("k", 2.0)
        .add_derived("d", fn=lambda k: k, args=["k", "k"])
    )
    with pytest.raises(Exception, match="arity mismatch"):
        model.evaluate()


def test_get_derived_parameters() -> None:
    model = chained_model()
    assert set(model.get_derived_parameters()) == {"d1", "d2"}
    assert model.get_derived_variables() == {}
