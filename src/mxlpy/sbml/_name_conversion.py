from __future__ import annotations

import keyword
import re

__all__ = [
    "RE_FROM_SBML",
    "RE_KWDS",
    "SBML_DOT",
]

RE_KWDS = re.compile("|".join(f"^{i}$" for i in keyword.kwlist))
SBML_DOT = "__SBML_DOT__"
RE_FROM_SBML = re.compile(r"__(\d+)__")


def _escape_keyword(re_sub: re.Match) -> str:
    return f"{re_sub.group(0)}_"


def _ascii_to_character(re_sub: re.Match) -> str:
    """Convert an escaped non-alphanumeric character."""
    return chr(int(re_sub.group(1)))


def _name_to_py(name: str) -> str:
    name = RE_FROM_SBML.sub(_ascii_to_character, name)
    name = RE_KWDS.sub(_escape_keyword, name)
    name = (
        name.replace(SBML_DOT, ".")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(".", "")
        .replace(",", "")
        .replace(":", "")
        .replace(";", "")
        .replace('"', "")
        .replace("'", "")
        .replace("^", "")
        .replace("|", "")
        .replace("=", "eq")
        .replace(">", "lg")
        .replace("<", "sm")
        .replace("+", "plus")
        .replace("-", "minus")
        .replace("*", "star")
        .replace("/", "div")
        # .lower()
    )
    if not name[0].isalpha():
        return f"_{name}"
    return name
