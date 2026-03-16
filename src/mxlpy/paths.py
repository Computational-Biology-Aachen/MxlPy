"""Shared paths between the mxlpy package."""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = [
    "default_tmp_dir",
]


def default_tmp_dir(tmp_dir: Path | None, *, remove_old_cache: bool) -> Path:
    """Returns the default temporary directory path.

    If `tmp_dir` is None, it defaults to the user's home directory under ".cache/mxlpy".
    Optionally removes old cache if specified.

    Parameters
    ----------
    tmp_dir : Path | None
        The temporary directory path. If None, defaults to
        Path.home() / ".cache" / "mxlpy".
    remove_old_cache : bool
        If True, removes the old cache directory if it exists.
        Defaults to False.

    Returns
    -------
    Path
        The path to the temporary directory.

    """
    if tmp_dir is None:
        tmp_dir = Path.home() / ".cache" / "mxlpy"

    if tmp_dir.exists() and remove_old_cache:
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(exist_ok=True, parents=True)
    return tmp_dir
