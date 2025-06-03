import pytest

from mxlpy import Model
from mxlpy.model import CircularDependencyError, MissingDependenciesError


def moiety_1(x1: float, tot: float) -> float:
    return tot - x1


def test_par_missing_parameter() -> None:
    m = (
        Model()
        .add_parameters({"x1": 1})
        # .add_parameters({"xtot": 1})
        .add_derived("x2", moiety_1, args=["x1", "xtot"])
    )
    with pytest.raises(MissingDependenciesError):
        m.get_args()


def test_par_circular() -> None:
    m = (
        Model()
        .add_parameters({"xtot": 1})
        .add_derived("x1", moiety_1, args=["xtot", "x2"])
        .add_derived("x2", moiety_1, args=["xtot", "x1"])
    )
    with pytest.raises(CircularDependencyError):
        m.get_args()


def test_mod_missing_parameter() -> None:
    m = (
        Model()
        .add_variables({"x1": 1})
        .add_derived("x2", moiety_1, args=["x1", "xtot"])
    )
    with pytest.raises(MissingDependenciesError):
        m.get_args()


def test_mod_missing_compound() -> None:
    m = (
        Model()
        .add_parameters({"xtot": 1})
        .add_derived("x2", moiety_1, args=["x1", "xtot"])
    )
    with pytest.raises(MissingDependenciesError):
        m.get_args()


def test_mod_circular() -> None:
    m = (
        Model()
        .add_parameters({"xtot": 1})
        .add_derived("x1", moiety_1, args=["xtot", "x2"])
        .add_derived("x2", moiety_1, args=["xtot", "x1"])
    )
    with pytest.raises(CircularDependencyError):
        m.get_args()
