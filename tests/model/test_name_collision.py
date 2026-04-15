"""Tests for cross-namespace name collisions in Model."""

from __future__ import annotations

import pytest

from mxlpy import fns
from mxlpy.model import Model


def test_add_variable_then_parameter_same_name() -> None:
    model = Model().add_variable("x", 1.0)
    with pytest.raises(NameError):
        model.add_parameter("x", 2.0)


def test_add_parameter_then_variable_same_name() -> None:
    model = Model().add_parameter("x", 1.0)
    with pytest.raises(NameError):
        model.add_variable("x", 2.0)


def test_add_parameter_then_derived_same_name() -> None:
    model = Model().add_parameter("x", 1.0)
    with pytest.raises(NameError):
        model.add_derived("x", fns.constant, args=["x"])


def test_add_variable_then_derived_same_name() -> None:
    model = Model().add_variable("x", 1.0)
    with pytest.raises(NameError):
        model.add_derived("x", fns.constant, args=["x"])


def test_add_parameter_then_reaction_same_name() -> None:
    model = Model().add_parameter("v1", 1.0).add_variable("S", 1.0)
    with pytest.raises(NameError):
        model.add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["S", "v1"],
            stoichiometry={"S": -1.0},
        )


def test_add_variable_then_reaction_same_name() -> None:
    model = Model().add_variable("v1", 1.0).add_parameter("k1", 1.0)
    with pytest.raises(NameError):
        model.add_reaction(
            "v1",
            fn=fns.mass_action_1s,
            args=["v1", "k1"],
            stoichiometry={"v1": -1.0},
        )


def test_add_derived_then_parameter_same_name() -> None:
    model = Model().add_derived("d1", fns.constant, args=["x"])
    with pytest.raises(NameError):
        model.add_parameter("d1", 1.0)


def test_add_derived_then_variable_same_name() -> None:
    model = Model().add_derived("d1", fns.constant, args=["x"])
    with pytest.raises(NameError):
        model.add_variable("d1", 1.0)
