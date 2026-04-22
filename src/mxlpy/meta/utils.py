import re


def _to_valid_identifier(name: str) -> str:
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
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "_"
