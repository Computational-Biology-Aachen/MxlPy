from __future__ import annotations

from mxlpy.meta import generate_model_code_jl
from tests import models


def test_generate_model_code_jl_m_0v_1p_0d_0r() -> None:
    assert generate_model_code_jl(models.m_0v_1p_0d_0r()).split("\n") == [
        "function model(time, variables)",
        "    k = 1.0",
        "    return ()",
        "end",
    ]


def test_generate_model_code_jl_m_0v_2p_0d_0r() -> None:
    assert generate_model_code_jl(models.m_0v_2p_0d_0r()).split("\n") == [
        "function model(time, variables)",
        "    k = 1.0",
        "    k = 2.0",
        "    return ()",
        "end",
    ]


def test_generate_model_code_jl_m_1v_0p_0d_0r() -> None:
    assert generate_model_code_jl(models.m_1v_0p_0d_0r()).split("\n") == [
        "function model(time, variables)",
        "    v1 = *variables",
        "    return ()",
        "end",
    ]


def test_generate_model_code_jl_m_1v_1p_1d_0r() -> None:
    assert generate_model_code_jl(models.m_1v_1p_1d_0r()).split("\n") == [
        "function model(time, variables)",
        "    v1 = *variables",
        "    k = 1.0",
        "    k = p1 + v1",
        "    return ()",
        "end",
    ]


def test_generate_model_code_jl_m_1v_1p_1d_1r() -> None:
    assert generate_model_code_jl(models.m_1v_1p_1d_1r()).split("\n") == [
        "function model(time, variables)",
        "    v1 = *variables",
        "    k = 1.0",
        "    k = p1 + v1",
        "    k = d1 .* v1",
        "    k = -r1",
        "    return dv1dt",
        "end",
    ]


def test_generate_model_code_jl_m_2v_0p_0d_0r() -> None:
    assert generate_model_code_jl(models.m_2v_0p_0d_0r()).split("\n") == [
        "function model(time, variables)",
        "    v1, v2 = *variables",
        "    return ()",
        "end",
    ]


def test_generate_model_code_jl_m_2v_1p_1d_1r() -> None:
    assert generate_model_code_jl(models.m_2v_1p_1d_1r()).split("\n") == [
        "function model(time, variables)",
        "    v1, v2 = *variables",
        "    k = 1.0",
        "    k = v1 + v2",
        "    k = p1 .* v1",
        "    k = -r1",
        "    k = r1",
        "    return dv1dt, dv2dt",
        "end",
    ]


def test_generate_model_code_jl_m_2v_2p_1d_1r() -> None:
    assert generate_model_code_jl(models.m_2v_2p_1d_1r()).split("\n") == [
        "function model(time, variables)",
        "    v1, v2 = *variables",
        "    k = 1.0",
        "    k = 2.0",
        "    k = v1 + v2",
        "    k = p1 .* v1",
        "    k = -r1",
        "    k = r1",
        "    return dv1dt, dv2dt",
        "end",
    ]


def test_generate_model_code_jl_m_2v_2p_2d_1r() -> None:
    assert generate_model_code_jl(models.m_2v_2p_2d_1r()).split("\n") == [
        "function model(time, variables)",
        "    v1, v2 = *variables",
        "    k = 1.0",
        "    k = 2.0",
        "    k = v1 + v2",
        "    k = v1 .* v2",
        "    k = p1 .* v1",
        "    k = -r1",
        "    k = r1",
        "    return dv1dt, dv2dt",
        "end",
    ]


def test_generate_model_code_jl_m_2v_2p_2d_2r() -> None:
    assert generate_model_code_jl(models.m_2v_2p_2d_2r()).split("\n") == [
        "function model(time, variables)",
        "    v1, v2 = *variables",
        "    k = 1.0",
        "    k = 2.0",
        "    k = p1 + v1",
        "    k = p2 .* v2",
        "    k = d1 .* v1",
        "    k = d2 .* v2",
        "    k = -r1 + r2",
        "    k = r1 - r2",
        "    return dv1dt, dv2dt",
        "end",
    ]


def test_generate_model_code_jl_m_dependent_derived() -> None:
    assert generate_model_code_jl(models.m_dependent_derived()).split("\n") == [
        "function model(time, variables)",
        "    k = 1.0",
        "    k = p1",
        "    k = d1",
        "    return ()",
        "end",
    ]


def test_generate_model_code_jl_m_derived_stoichiometry() -> None:
    assert generate_model_code_jl(models.m_derived_stoichiometry()).split("\n") == [
        "function model(time, variables)",
        "    v1 = *variables",
        "    k = v1",
        "    k = r1 ./ v1",
        "    return dv1dt",
        "end",
    ]
