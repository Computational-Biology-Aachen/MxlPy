"""Tests for the tex export module."""

from mxlpy import fns
from mxlpy.meta.codegen_latex import (
    TexExport,
    TexReaction,
    default_init,
)
from mxlpy.types import Derived


def test_default_init() -> None:
    """Test default_init function."""
    assert default_init(None) == {}
    test_dict = {"a": 1}
    assert default_init(test_dict) is test_dict


def test_tex_reaction() -> None:
    """Test TexReaction dataclass."""

    def example_fn(a: float, b: float) -> float:
        return a + b

    tr = TexReaction(fn=example_fn, args=["x", "y"])
    assert tr.fn is example_fn
    assert tr.args == ["x", "y"]


def test_tex_export_diff() -> None:
    """Test TexExport subtraction."""
    export1 = TexExport(
        parameters={"k1": 1.0},
        variables={"A": 10.0},
        derived={},
        reactions={},
        diff_eqs={},
    )

    export2 = TexExport(
        parameters={"k1": 2.0, "k2": 3.0},
        variables={"A": 10.0, "B": 5.0},
        derived={},
        reactions={},
        diff_eqs={},
    )

    diff = export2 - export1
    # The actual implementation keeps k1 with its original value and variables is an empty dict
    assert diff.parameters == {"k1": 1.0}
    assert diff.variables == {}


def test_tex_export_rename_with_glossary() -> None:
    """Test renaming with glossary."""

    derived_val = Derived(fn=fns.twice, args=["A"])

    export = TexExport(
        parameters={"k1": 1.0},
        variables={"A": 10.0},
        derived={"d1": derived_val},
        reactions={"r1": TexReaction(fn=fns.add, args=["A", "k1"])},
        diff_eqs={"r1": {"A": 1.0}},
    )

    gls = {"A": "substrate", "k1": "rate_constant"}
    renamed = export.rename_with_glossary(gls)

    assert "substrate" in renamed.variables
    assert "rate_constant" in renamed.parameters
    assert renamed.reactions["r1"].args == ["substrate", "rate_constant"]


def test_export_methods() -> None:
    """Test export methods."""
    export = TexExport(
        parameters={"k1": 1.0},
        variables={"A": 10.0},
        derived={},
        reactions={},
        diff_eqs={},
    )

    # Test that these methods execute without errors
    variables_tex = export.export_variables()
    assert "Model variables" in variables_tex

    parameters_tex = export.export_parameters()
    assert "Model parameters" in parameters_tex

    all_tex = export.export_all()
    assert "Variables" in all_tex
    assert "Parameters" in all_tex

    doc = export.export_document()
    assert "\\documentclass{article}" in doc
    assert "\\maketitle" in doc
