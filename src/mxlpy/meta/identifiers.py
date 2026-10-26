# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

# ruff: noqa: RUF001

"""Canonical identifiers shared with mxlweb-core.

Model ids are free-form strings (``"CO2 (dissolved)"``), but exporters that turn a
model into code need identifiers. Both mxlpy and mxlweb-core map names through the
same rules (mxl-schemas ``v1/identifiers.json``), so a model has the same ids in
Python and JavaScript. ``RESERVED`` and ``TRANSLITERATE`` are vendored copies of
that file; a drift test compares them against it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "RESERVED",
    "SYMBOLS",
    "TRANSLITERATE",
    "canonical_identifier",
    "canonical_identifier_map",
    "is_canonical_identifier",
]

RESERVED: frozenset[str] = frozenset(
    {
        # Python hard keywords
        "False", "None", "True", "and", "as", "assert", "async", "await", "break",
        "class", "continue", "def", "del", "elif", "else", "except", "finally", "for",
        "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not",
        "or", "pass", "raise", "return", "try", "while", "with", "yield",
        # JavaScript reserved words
        "Infinity", "NaN", "arguments", "case", "catch", "const", "debugger",
        "default", "delete", "do", "enum", "eval", "export", "extends", "false",
        "function", "implements", "instanceof", "interface", "let", "new", "null",
        "package", "private", "protected", "public", "static", "super", "switch",
        "this", "throw", "true", "typeof", "undefined", "var", "void",
        # Names the generated Python / JavaScript code uses itself
        "Math", "all", "all_derived", "derived", "eqx", "get_model", "jax", "jnp",
        "m", "math", "model", "np", "pars", "selected_derived", "time", "variables",
    }
)  # fmt: skip

SYMBOLS: dict[str, str] = {"+": "plus", "*": "star", "/": "per"}

TRANSLITERATE: dict[str, str] = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron", "π": "pi",
    "ρ": "rho", "σ": "sigma", "ς": "sigma", "τ": "tau", "υ": "upsilon",
    "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "Α": "Alpha", "Β": "Beta", "Γ": "Gamma", "Δ": "Delta", "Ε": "Epsilon",
    "Ζ": "Zeta", "Η": "Eta", "Θ": "Theta", "Ι": "Iota", "Κ": "Kappa",
    "Λ": "Lambda", "Μ": "Mu", "Ν": "Nu", "Ξ": "Xi", "Ο": "Omicron", "Π": "Pi",
    "Ρ": "Rho", "Σ": "Sigma", "Τ": "Tau", "Υ": "Upsilon", "Φ": "Phi",
    "Χ": "Chi", "Ψ": "Psi", "Ω": "Omega",
    "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O",
    "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "þ": "th", "Þ": "Th",
}  # fmt: skip

_CANONICAL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# A `-` ending a token is a charge / reduced-state marker (`FA-`), one between
# letters or digits a hyphen (`Light-harvesting`).
_CHARGE_MINUS = re.compile(r"-(?=[^A-Za-z0-9]|$)")
_EDGES = re.compile(r"^[^A-Za-z0-9_]+|[^A-Za-z0-9_]+$")
_INVALID_RUN = re.compile(r"[^A-Za-z0-9]*[^A-Za-z0-9_][^A-Za-z0-9]*")


def is_canonical_identifier(name: str) -> bool:
    """Whether `name` is already canonical (and thus maps to itself)."""
    return _CANONICAL.match(name) is not None and name not in RESERVED


def canonical_identifier(name: str) -> str:
    """Map `name` to its canonical identifier.

    Examples
    --------
        >>> canonical_identifier("CO2 (dissolved)")
        'CO2_dissolved'
        >>> canonical_identifier("P700+FA-")
        'P700_plus_FA_minus'
        >>> canonical_identifier("Δψ")
        'Deltapsi'
        >>> canonical_identifier("time")
        'time_'

    """
    if is_canonical_identifier(name):
        return name
    s = "".join(
        c
        for c in unicodedata.normalize("NFKD", name)
        if not unicodedata.category(c).startswith("M")
    )
    s = "".join(TRANSLITERATE.get(c, c) for c in s)
    s = _CHARGE_MINUS.sub(" minus ", s)
    s = "".join(f" {SYMBOLS[c]} " if c in SYMBOLS else c for c in s)
    s = _INVALID_RUN.sub("_", _EDGES.sub("", s)) or "_"
    if s[0].isdigit():
        s = f"_{s}"
    if s in RESERVED:
        s = f"{s}_"
    return s


def canonical_identifier_map(names: Iterable[str]) -> dict[str, str]:
    """Map every name of one model to its canonical identifier.

    Raises
    ------
    ValueError
        If two names map to the same identifier. Rename one of them at the source.

    """
    out: dict[str, str] = {}
    owner: dict[str, str] = {}
    for name in names:
        ident = canonical_identifier(name)
        if (other := owner.get(ident)) is not None:
            msg = f"{other!r} and {name!r} both map to the identifier {ident!r}; rename one of them"
            raise ValueError(msg)
        owner[ident] = name
        out[name] = ident
    return out
