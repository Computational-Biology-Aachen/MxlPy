from __future__ import annotations

from mxlpy.meta import generate_model_code_rs
from tests import models


def test_generate_model_code_py_m_1v_0p_0d_0r() -> None:
    assert generate_model_code_rs(models.m_1v_0p_0d_0r()).split("\n") == [
        "fn model(time: f64, variables: &[f64; 1]) -> [f64; 1] {",
        "    let [v1] = *variables;",
        "    return [()]",
        "}",
    ]


def test_generate_model_code_py_m_2v_0p_0d_0r() -> None:
    assert generate_model_code_rs(models.m_2v_0p_0d_0r()).split("\n") == [
        "fn model(time: f64, variables: &[f64; 2]) -> [f64; 2] {",
        "    let [v1, v2] = *variables;",
        "    return [()]",
        "}",
    ]


def test_generate_model_code_py_m_0v_1p_0d_0r() -> None:
    assert generate_model_code_rs(models.m_0v_1p_0d_0r()).split("\n") == [
        "fn model(time: f64, variables: &[f64; 0]) -> [f64; 0] {",
        "    let p1: f64 = 1.0;",
        "    return [()]",
        "}",
    ]


def test_generate_model_code_py_m_0v_2p_0d_0r() -> None:
    assert generate_model_code_rs(models.m_0v_2p_0d_0r()).split("\n") == [
        "fn model(time: f64, variables: &[f64; 0]) -> [f64; 0] {",
        "    let p1: f64 = 1.0;",
        "    let p2: f64 = 2.0;",
        "    return [()]",
        "}",
    ]


def test_generate_model_code_py_m_1v_1p_1d_0r() -> None:
    assert generate_model_code_rs(models.m_1v_1p_1d_0r()).split("\n") == [
        "fn model(time: f64, variables: &[f64; 1]) -> [f64; 1] {",
        "    let [v1] = *variables;",
        "    let p1: f64 = 1.0;",
        "    let d1: f64 = p1 + v1;",
        "    return [()]",
        "}",
    ]


def test_generate_model_code_py_m_1v_1p_1d_1r() -> None:
    assert generate_model_code_rs(models.m_1v_1p_1d_1r()).split("\n") == [
        "fn model(time: f64, variables: &[f64; 1]) -> [f64; 1] {",
        "    let [v1] = *variables;",
        "    let p1: f64 = 1.0;",
        "    let d1: f64 = p1 + v1;",
        "    let r1: f64 = d1*v1;",
        "    let dv1dt: f64 = -r1;",
        "    return [dv1dt]",
        "}",
    ]


def test_generate_model_code_py_m_2v_1p_1d_1r() -> None:
    assert generate_model_code_rs(models.m_2v_1p_1d_1r()).split("\n") == [
        "fn model(time: f64, variables: &[f64; 2]) -> [f64; 2] {",
        "    let [v1, v2] = *variables;",
        "    let p1: f64 = 1.0;",
        "    let d1: f64 = v1 + v2;",
        "    let r1: f64 = p1*v1;",
        "    let dv1dt: f64 = -r1;",
        "    let dv2dt: f64 = r1;",
        "    return [dv1dt, dv2dt]",
        "}",
    ]


def test_generate_model_code_py_m_2v_2p_1d_1r() -> None:
    assert generate_model_code_rs(models.m_2v_2p_1d_1r()).split("\n") == [
        "fn model(time: f64, variables: &[f64; 2]) -> [f64; 2] {",
        "    let [v1, v2] = *variables;",
        "    let p1: f64 = 1.0;",
        "    let p2: f64 = 2.0;",
        "    let d1: f64 = v1 + v2;",
        "    let r1: f64 = p1*v1;",
        "    let dv1dt: f64 = -r1;",
        "    let dv2dt: f64 = r1;",
        "    return [dv1dt, dv2dt]",
        "}",
    ]


def test_generate_model_code_py_m_2v_2p_2d_1r() -> None:
    assert generate_model_code_rs(models.m_2v_2p_2d_1r()).split("\n") == [
        "fn model(time: f64, variables: &[f64; 2]) -> [f64; 2] {",
        "    let [v1, v2] = *variables;",
        "    let p1: f64 = 1.0;",
        "    let p2: f64 = 2.0;",
        "    let d1: f64 = v1 + v2;",
        "    let d2: f64 = v1*v2;",
        "    let r1: f64 = p1*v1;",
        "    let dv1dt: f64 = -r1;",
        "    let dv2dt: f64 = r1;",
        "    return [dv1dt, dv2dt]",
        "}",
    ]


def test_generate_model_code_py_m_2v_2p_2d_2r() -> None:
    assert generate_model_code_rs(models.m_2v_2p_2d_2r()).split("\n") == [
        "fn model(time: f64, variables: &[f64; 2]) -> [f64; 2] {",
        "    let [v1, v2] = *variables;",
        "    let p1: f64 = 1.0;",
        "    let p2: f64 = 2.0;",
        "    let d1: f64 = p1 + v1;",
        "    let d2: f64 = p2*v2;",
        "    let r1: f64 = d1*v1;",
        "    let r2: f64 = d2*v2;",
        "    let dv1dt: f64 = -r1 + r2;",
        "    let dv2dt: f64 = r1 - r2;",
        "    return [dv1dt, dv2dt]",
        "}",
    ]


def test_generate_model_code_py_m_dependent_derived() -> None:
    assert generate_model_code_rs(models.m_dependent_derived()).split("\n") == [
        "fn model(time: f64, variables: &[f64; 0]) -> [f64; 0] {",
        "    let p1: f64 = 1.0;",
        "    let d1: f64 = p1;",
        "    let d2: f64 = d1;",
        "    return [()]",
        "}",
    ]


def test_generate_model_code_py_m_derived_stoichiometry() -> None:
    assert generate_model_code_rs(models.m_derived_stoichiometry()).split("\n") == [
        "fn model(time: f64, variables: &[f64; 1]) -> [f64; 1] {",
        "    let [v1] = *variables;",
        "    let r1: f64 = v1;",
        "    let dv1dt: f64 = r1/v1;",
        "    return [dv1dt]",
        "}",
    ]
