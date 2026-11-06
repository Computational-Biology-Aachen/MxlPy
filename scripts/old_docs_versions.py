# Copyright 2026 Marvin van Aalst, Anna Matuszyńska

"""Print the docs versions to delete so only the newest KEEP remain.

Reads `mike list --json` from stdin and prints the versions beyond the newest KEEP
(argv[1]) as one space-separated line. Versions with an alias (e.g. `latest`) and
names that aren't release numbers (e.g. a hand-deployed `dev`) are never deleted.
Used by .github/workflows/release.yml.
"""

from __future__ import annotations

import json
import sys

from packaging.version import InvalidVersion, Version


def _release(version: str) -> Version | None:
    try:
        return Version(version)
    except InvalidVersion:
        return None


def main() -> None:
    """Print the versions to delete."""
    keep = int(sys.argv[1])
    releases = [
        (v, e)
        for e in json.load(sys.stdin)
        if (v := _release(e["version"])) is not None
    ]
    releases.sort(key=lambda ve: ve[0], reverse=True)
    old = [e["version"] for _, e in releases[keep:] if not e["aliases"]]
    sys.stdout.write(" ".join(old) + "\n")


if __name__ == "__main__":
    main()
