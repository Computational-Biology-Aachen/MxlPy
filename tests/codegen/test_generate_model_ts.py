from __future__ import annotations

from mxlpy.meta import generate_model_code_ts
from tests import models


def test_generate_model_code_ts_m_1v_0p_0d_0r() -> None:
    assert generate_model_code_ts(models.m_1v_0p_0d_0r()).split("\n") == [
        "function model(time: number, variables: number[]) {",
        "    let [v1] = variables;",
        "    return [()];",
        "};",
    ]


def test_generate_model_code_ts_m_2v_0p_0d_0r() -> None:
    assert generate_model_code_ts(models.m_2v_0p_0d_0r()).split("\n") == [
        "function model(time: number, variables: number[]) {",
        "    let [v1, v2] = variables;",
        "    return [()];",
        "};",
    ]


def test_generate_model_code_ts_m_0v_1p_0d_0r() -> None:
    assert generate_model_code_ts(models.m_0v_1p_0d_0r()).split("\n") == [
        "function model(time: number, variables: number[]) {",
        "    let p1: number = 1.0;",
        "    return [()];",
        "};",
    ]


def test_generate_model_code_ts_m_0v_2p_0d_0r() -> None:
    assert generate_model_code_ts(models.m_0v_2p_0d_0r()).split("\n") == [
        "function model(time: number, variables: number[]) {",
        "    let p1: number = 1.0;",
        "    let p2: number = 2.0;",
        "    return [()];",
        "};",
    ]


def test_generate_model_code_ts_m_1v_1p_1d_0r() -> None:
    assert generate_model_code_ts(models.m_1v_1p_1d_0r()).split("\n") == [
        "function model(time: number, variables: number[]) {",
        "    let [v1] = variables;",
        "    let p1: number = 1.0;",
        "    let d1: number = p1 + v1;",
        "    return [()];",
        "};",
    ]


def test_generate_model_code_ts_m_1v_1p_1d_1r() -> None:
    assert generate_model_code_ts(models.m_1v_1p_1d_1r()).split("\n") == [
        "function model(time: number, variables: number[]) {",
        "    let [v1] = variables;",
        "    let p1: number = 1.0;",
        "    let d1: number = p1 + v1;",
        "    let r1: number = d1*v1;",
        "    let dv1dt: number = -r1;",
        "    return [dv1dt];",
        "};",
    ]


def test_generate_model_code_ts_m_2v_1p_1d_1r() -> None:
    assert generate_model_code_ts(models.m_2v_1p_1d_1r()).split("\n") == [
        "function model(time: number, variables: number[]) {",
        "    let [v1, v2] = variables;",
        "    let p1: number = 1.0;",
        "    let d1: number = v1 + v2;",
        "    let r1: number = p1*v1;",
        "    let dv1dt: number = -r1;",
        "    let dv2dt: number = r1;",
        "    return [dv1dt, dv2dt];",
        "};",
    ]


def test_generate_model_code_ts_m_2v_2p_1d_1r() -> None:
    assert generate_model_code_ts(models.m_2v_2p_1d_1r()).split("\n") == [
        "function model(time: number, variables: number[]) {",
        "    let [v1, v2] = variables;",
        "    let p1: number = 1.0;",
        "    let p2: number = 2.0;",
        "    let d1: number = v1 + v2;",
        "    let r1: number = p1*v1;",
        "    let dv1dt: number = -r1;",
        "    let dv2dt: number = r1;",
        "    return [dv1dt, dv2dt];",
        "};",
    ]


def test_generate_model_code_ts_m_2v_2p_2d_1r() -> None:
    assert generate_model_code_ts(models.m_2v_2p_2d_1r()).split("\n") == [
        "function model(time: number, variables: number[]) {",
        "    let [v1, v2] = variables;",
        "    let p1: number = 1.0;",
        "    let p2: number = 2.0;",
        "    let d1: number = v1 + v2;",
        "    let d2: number = v1*v2;",
        "    let r1: number = p1*v1;",
        "    let dv1dt: number = -r1;",
        "    let dv2dt: number = r1;",
        "    return [dv1dt, dv2dt];",
        "};",
    ]


def test_generate_model_code_ts_m_2v_2p_2d_2r() -> None:
    assert generate_model_code_ts(models.m_2v_2p_2d_2r()).split("\n") == [
        "function model(time: number, variables: number[]) {",
        "    let [v1, v2] = variables;",
        "    let p1: number = 1.0;",
        "    let p2: number = 2.0;",
        "    let d1: number = p1 + v1;",
        "    let d2: number = p2*v2;",
        "    let r1: number = d1*v1;",
        "    let r2: number = d2*v2;",
        "    let dv1dt: number = -r1 + r2;",
        "    let dv2dt: number = r1 - r2;",
        "    return [dv1dt, dv2dt];",
        "};",
    ]


def test_generate_model_code_ts_m_dependent_derived() -> None:
    assert generate_model_code_ts(models.m_dependent_derived()).split("\n") == [
        "function model(time: number, variables: number[]) {",
        "    let p1: number = 1.0;",
        "    let d1: number = p1;",
        "    let d2: number = d1;",
        "    return [()];",
        "};",
    ]


def test_generate_model_code_ts_m_derived_stoichiometry() -> None:
    assert generate_model_code_ts(models.m_derived_stoichiometry()).split("\n") == [
        "function model(time: number, variables: number[]) {",
        "    let [v1] = variables;",
        "    let r1: number = v1;",
        "    let dv1dt: number = r1/v1;",
        "    return [dv1dt];",
        "};",
    ]
