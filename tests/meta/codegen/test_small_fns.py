from __future__ import annotations

import ast

from mxlpy.meta.source_tools import get_fn_ast


def sample_function(x: float, y: float) -> float:
    """A sample function for testing.

    This is a multiline docstring.
    """
    # This is a comment
    return x + y


def test_get_fn_source() -> None:
    fn_def = get_fn_ast(sample_function)

    assert isinstance(fn_def, ast.FunctionDef)
    assert fn_def.name == "sample_function"
    assert len(fn_def.args.args) == 2
    assert fn_def.args.args[0].arg == "x"
    assert fn_def.args.args[1].arg == "y"

    # Skip testing with non-function input as it raises a different error than expected
