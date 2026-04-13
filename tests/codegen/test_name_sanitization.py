"""Tests for name sanitization in code generation.

Model component names may contain characters that are invalid identifiers
in target languages (e.g. hyphens, spaces, dots). The generators must
sanitize all names before emitting code.
"""

from __future__ import annotations

import pytest

from mxlpy import Model, fns
from mxlpy.meta import (
    generate_model_code_jl,
    generate_model_code_py,
    generate_model_code_rs,
    generate_model_code_ts,
)
from mxlpy.meta.codegen_model import (
    _to_valid_identifier,
    generate_model_code_c,
    generate_model_code_cpp,
    generate_model_code_matlab,
)


# ---------------------------------------------------------------------------
# Unit tests for _to_valid_identifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("my-var", "my_var"),
        ("my.var", "my_var"),
        ("my var", "my_var"),
        ("1abc", "_1abc"),
        ("a+b", "a_b"),
        ("", "_"),
        ("valid_name", "valid_name"),
        ("CamelCase", "CamelCase"),
        ("__private", "__private"),
        ("a[0]", "a_0_"),
        ("k.1", "k_1"),
    ],
)
def test_to_valid_identifier(name: str, expected: str) -> None:
    assert _to_valid_identifier(name) == expected


# ---------------------------------------------------------------------------
# Integration: model with invalid names generates valid code
# ---------------------------------------------------------------------------


def _model_with_invalid_names() -> Model:
    """Model using hyphens and dots in component names."""
    return (
        Model()
        .add_variable("my-var", 1.0)
        .add_parameter("k.1", 2.0)
        .add_derived(
            "d-1",
            fn=fns.add,
            args=["my-var", "k.1"],
        )
        .add_reaction(
            "r-1",
            fn=fns.mass_action_1s,
            args=["my-var", "d-1"],
            stoichiometry={"my-var": -1.0},
        )
    )


def test_name_sanitization_ts() -> None:
    lines = generate_model_code_ts(_model_with_invalid_names()).split("\n")
    assert lines == [
        "function model(time: number, variables: number[]) {",
        "    let [my_var] = variables;",
        "    let k_1: number = 2.0;",
        "    let d_1: number = k_1 + my_var;",
        "    let r_1: number = d_1*my_var;",
        "    let dmy_vardt: number = -r_1;",
        "    return [dmy_vardt];",
        "};",
    ]


def test_name_sanitization_py() -> None:
    code = generate_model_code_py(_model_with_invalid_names())
    assert "my_var" in code
    assert "k_1" in code
    assert "d_1" in code
    assert "r_1" in code
    assert "dmy_vardt" in code
    assert "my-var" not in code
    assert "k.1" not in code
    assert "d-1" not in code
    assert "r-1" not in code


def test_name_sanitization_c() -> None:
    code = generate_model_code_c(_model_with_invalid_names())
    assert "my_var" in code
    assert "k_1" in code
    assert "d_1" in code
    assert "r_1" in code
    assert "dmy_vardt" in code
    # original invalid names must not appear
    assert "my-var" not in code
    assert "k.1" not in code
    assert "d-1" not in code
    assert "r-1" not in code


def test_name_sanitization_cpp() -> None:
    code = generate_model_code_cpp(_model_with_invalid_names())
    assert "my_var" in code
    assert "k_1" in code
    assert "dmy_vardt" in code
    assert "my-var" not in code


def test_name_sanitization_rs() -> None:
    code = generate_model_code_rs(_model_with_invalid_names())
    assert "my_var" in code
    assert "k_1" in code
    assert "dmy_vardt" in code
    assert "my-var" not in code


def test_name_sanitization_jl() -> None:
    code = generate_model_code_jl(_model_with_invalid_names())
    assert "my_var" in code
    assert "k_1" in code
    assert "dmy_vardt" in code
    assert "my-var" not in code


def test_name_sanitization_matlab() -> None:
    code = generate_model_code_matlab(_model_with_invalid_names())
    assert "my_var" in code
    assert "k_1" in code
    assert "dmy_vardt" in code
    assert "my-var" not in code
