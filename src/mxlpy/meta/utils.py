import re

__all__ = ["valid_identifier", "valid_tex_identifier"]


def valid_identifier(name: str) -> str:
    """Convert an arbitrary string to a valid identifier.

    Uses C99 identifier rules ([a-zA-Z_][a-zA-Z0-9_]*), which are the
    strictest common denominator across all supported target languages
    (C, C++, Rust, TypeScript, Julia, MATLAB/Octave, Python).

    Parameters
    ----------
    name
        Original component name from the model

    Returns
    -------
    str
        Sanitized identifier safe for use in all target languages

    """
    sanitized = (
        name.replace(" ", "_")
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
        .replace("=", "_eq_")
        .replace(">", "_lg_")
        .replace("<", "_sm_")
        .replace("+", "_plus_")
        .replace("-", "_minus_")
        .replace("*", "_star_")
        .replace("/", "_div_")
    )
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.rstrip("_")
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "_"


def valid_tex_identifier(k: str) -> str:
    return k.replace("_", r"\_")
