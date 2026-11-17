# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

from pathlib import Path

import pytest

from mxlpy import KineticModelBuilder
from mxlpy.sbml import write


def test_docstringremover_visit_expr() -> None:
    # FIXME: implement this
    assert True


def test_identifierreplacer_visit_name() -> None:
    # FIXME: implement this
    assert True


def test_write(tmp_path: Path) -> None:
    file = write(KineticModelBuilder().add_variable("x", 1.0), tmp_path / "m.xml")
    assert file.read_text().startswith("<?xml")


def test_write_missing_directory_raises(tmp_path: Path) -> None:
    file = tmp_path / "missing" / "m.xml"
    with pytest.raises(OSError, match="Could not write SBML file"):
        write(KineticModelBuilder().add_variable("x", 1.0), file)
