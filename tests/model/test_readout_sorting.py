"""Readouts are calculated in dependency order, not insertion order."""

from collections.abc import Callable

import pandas as pd
import pytest

from mxlpy import KineticModelBuilder, OdeModelBuilder, fns
from mxlpy._topo import CircularDependencyError, MissingDependenciesError

ModelFactory = Callable[[], KineticModelBuilder] | Callable[[], OdeModelBuilder]


def _kinetic() -> KineticModelBuilder:
    return (
        KineticModelBuilder()
        .add_parameters({"k": 2.0})
        .add_variables({"x": 3.0})
        .add_derived("tot", fns.add, args=["x", "k"])
        .add_reaction("v", fns.mul, args=["k", "x"], stoichiometry={"x": -1})
    )


def _ode() -> OdeModelBuilder:
    return (
        OdeModelBuilder()
        .add_parameters({"k": 2.0})
        .add_diff_eq("x", 3.0, fns.neg, args=["x"])
        .add_derived("tot", fns.add, args=["x", "k"])
    )


@pytest.mark.parametrize("make", [_kinetic, _ode])
def test_readout_added_before_its_dependency(make: ModelFactory) -> None:
    m = (
        make()
        # Added first, but reads "twice_tot", which is added after it
        .add_readout("plus_one", fns.one_div, args=["twice_tot"])
        .add_readout("twice_tot", fns.twice, args=["tot"])
    )
    args = m.get_args(include_readouts=True)
    assert args["twice_tot"] == 10.0
    assert args["plus_one"] == pytest.approx(0.1)


def test_readout_reads_reaction_flux() -> None:
    m = (
        _kinetic()
        .add_readout("scaled", fns.twice, args=["flux2"])
        .add_readout("flux2", fns.twice, args=["v"])
    )
    args = m.get_args(include_readouts=True)
    assert args["flux2"] == 12.0
    assert args["scaled"] == 24.0


@pytest.mark.parametrize("make", [_kinetic, _ode])
def test_readout_order_in_time_course(make: ModelFactory) -> None:
    m = (
        make()
        .add_readout("b", fns.twice, args=["a"])
        .add_readout("a", fns.twice, args=["tot"])
    )
    variables = pd.DataFrame({"x": [1.0, 2.0]}, index=[0.0, 1.0])
    args = m.get_args_time_course(variables, include_readouts=True)
    assert args["b"].tolist() == [12.0, 16.0]


@pytest.mark.parametrize("make", [_kinetic, _ode])
def test_readout_added_after_args_were_cached(make: ModelFactory) -> None:
    m = make()
    m.get_args()
    m.add_readout("late", fns.twice, args=["tot"])
    assert m.get_args(include_readouts=True)["late"] == 10.0
    m.remove_readout("late")
    assert "late" not in m.get_args(include_readouts=True)


@pytest.mark.parametrize("make", [_kinetic, _ode])
def test_readout_missing_dependency(make: ModelFactory) -> None:
    m = make().add_readout("r", fns.twice, args=["ghost"])
    with pytest.raises(MissingDependenciesError):
        m.get_args(include_readouts=True)


@pytest.mark.parametrize("make", [_kinetic, _ode])
def test_readout_circular(make: ModelFactory) -> None:
    m = (
        make()
        .add_readout("r1", fns.twice, args=["r2"])
        .add_readout("r2", fns.twice, args=["r1"])
    )
    with pytest.raises(CircularDependencyError):
        m.get_args(include_readouts=True)
