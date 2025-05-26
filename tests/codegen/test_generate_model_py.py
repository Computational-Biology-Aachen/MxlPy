from __future__ import annotations

from mxlpy.meta import generate_model_code_py
from tests import models


def test_generate_model_code_py_m_0v_1p_0d_0r() -> None:
    assert generate_model_code_py(models.m_0v_1p_0d_0r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from mxlpy.types import Float",
        "",
        "def model(t: Float, variables: Float) -> Iterable[Float]:",
        "    p1 = 1.0",
        "    return ()",
    ]


def test_generate_model_code_py_m_0v_2p_0d_0r() -> None:
    assert generate_model_code_py(models.m_0v_2p_0d_0r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from mxlpy.types import Float",
        "",
        "def model(t: Float, variables: Float) -> Iterable[Float]:",
        "    p1 = 1.0",
        "    p2 = 2.0",
        "    return ()",
    ]


def test_generate_model_code_py_m_1v_0p_0d_0r() -> None:
    assert generate_model_code_py(models.m_1v_0p_0d_0r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from mxlpy.types import Float",
        "",
        "def model(t: Float, variables: Float) -> Iterable[Float]:",
        "    v1 = variables",
        "    return dv1dt",
    ]


def test_generate_model_code_py_m_1v_1p_1d_0r() -> None:
    assert generate_model_code_py(models.m_1v_1p_1d_0r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from mxlpy.types import Float",
        "",
        "def model(t: Float, variables: Float) -> Iterable[Float]:",
        "    v1 = variables",
        "    p1 = 1.0",
        "    d1 = p1 + v1",
        "    return dv1dt",
    ]


def test_generate_model_code_py_m_1v_1p_1d_1r() -> None:
    assert generate_model_code_py(models.m_1v_1p_1d_1r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from mxlpy.types import Float",
        "",
        "def model(t: Float, variables: Float) -> Iterable[Float]:",
        "    v1 = variables",
        "    p1 = 1.0",
        "    d1 = p1 + v1",
        "    r1 = d1*v1",
        "    dv1dt = -r1",
        "    return dv1dt",
    ]


def test_generate_model_code_py_m_2v_0p_0d_0r() -> None:
    assert generate_model_code_py(models.m_2v_0p_0d_0r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from mxlpy.types import Float",
        "",
        "def model(t: Float, variables: Float) -> Iterable[Float]:",
        "    v1, v2 = variables",
        "    return dv1dt, dv2dt",
    ]


def test_generate_model_code_py_m_2v_1p_1d_1r() -> None:
    assert generate_model_code_py(models.m_2v_1p_1d_1r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from mxlpy.types import Float",
        "",
        "def model(t: Float, variables: Float) -> Iterable[Float]:",
        "    v1, v2 = variables",
        "    p1 = 1.0",
        "    d1 = v1 + v2",
        "    r1 = p1*v1",
        "    dv1dt = -r1",
        "    dv2dt = r1",
        "    return dv1dt, dv2dt",
    ]


def test_generate_model_code_py_m_2v_2p_1d_1r() -> None:
    assert generate_model_code_py(models.m_2v_2p_1d_1r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from mxlpy.types import Float",
        "",
        "def model(t: Float, variables: Float) -> Iterable[Float]:",
        "    v1, v2 = variables",
        "    p1 = 1.0",
        "    p2 = 2.0",
        "    d1 = v1 + v2",
        "    r1 = p1*v1",
        "    dv1dt = -r1",
        "    dv2dt = r1",
        "    return dv1dt, dv2dt",
    ]


def test_generate_model_code_py_m_2v_2p_2d_1r() -> None:
    assert generate_model_code_py(models.m_2v_2p_2d_1r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from mxlpy.types import Float",
        "",
        "def model(t: Float, variables: Float) -> Iterable[Float]:",
        "    v1, v2 = variables",
        "    p1 = 1.0",
        "    p2 = 2.0",
        "    d1 = v1 + v2",
        "    d2 = v1*v2",
        "    r1 = p1*v1",
        "    dv1dt = -r1",
        "    dv2dt = r1",
        "    return dv1dt, dv2dt",
    ]


def test_generate_model_code_py_m_2v_2p_2d_2r() -> None:
    assert generate_model_code_py(models.m_2v_2p_2d_2r()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from mxlpy.types import Float",
        "",
        "def model(t: Float, variables: Float) -> Iterable[Float]:",
        "    v1, v2 = variables",
        "    p1 = 1.0",
        "    p2 = 2.0",
        "    d1 = p1 + v1",
        "    d2 = p2*v2",
        "    r1 = d1*v1",
        "    r2 = d2*v2",
        "    dv1dt = -r1 + r2",
        "    dv2dt = r1 - r2",
        "    return dv1dt, dv2dt",
    ]


def test_generate_model_code_py_m_dependent_derived() -> None:
    assert generate_model_code_py(models.m_dependent_derived()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from mxlpy.types import Float",
        "",
        "def model(t: Float, variables: Float) -> Iterable[Float]:",
        "    p1 = 1.0",
        "    d1 = p1",
        "    d2 = d1",
        "    return ()",
    ]


def test_generate_model_code_py_m_derived_stoichiometry() -> None:
    assert generate_model_code_py(models.m_derived_stoichiometry()).split("\n") == [
        "from collections.abc import Iterable",
        "",
        "from mxlpy.types import Float",
        "",
        "def model(t: Float, variables: Float) -> Iterable[Float]:",
        "    v1 = variables",
        "    r1 = v1",
        "    dv1dt = 1.0/v1 * r1",
        "    return dv1dt",
    ]
