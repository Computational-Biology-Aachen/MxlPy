from __future__ import annotations

import pytest

from mxlpy.meta._via_sym_repr import valid_identifier


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("my-var", "my_minus_var"),
        ("my.var", "myvar"),
        ("my var", "my_var"),
        ("1abc", "_1abc"),
        ("a+b", "a_plus_b"),
        ("", "_"),
        ("valid_name", "valid_name"),
        ("CamelCase", "CamelCase"),
        ("__private", "_private"),
        ("a[0]", "a0"),
        ("k.1", "k1"),
    ],
)
def test_to_valid_identifier(name: str, expected: str) -> None:
    assert valid_identifier(name) == expected
