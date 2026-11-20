# Copyright 2026 Marvin van Aalst, Anna Matuszyńska
# ruff: noqa: T201  (a command line script reports with print)

"""Fail if `import mxlpy` takes more private memory than the budget.

Every worker process of a parallel routine imports mxlpy, so its import cost is
paid once per CPU core. Heavy dependencies only some routines need are imported
lazily (see `mxlpy.__init__`); this guards against eager imports creeping back.

Measures the private (anonymous) resident memory a fresh interpreter gains by
importing mxlpy, which is what every additional process costs, unlike shared
library pages. Runs outside pytest, so the measurement comes from a clean
interpreter. Linux only, as it reads /proc; CI runs it on ubuntu.

On failure, prints how much each third-party package contributed and which
module imported it first.

Usage: python scripts/check_import_memory.py [--budget MIB]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# 122 MiB with CPython 3.13 on Linux in 2026-11 (175 MiB before the analysis
# subpackages became lazy)
BUDGET_MIB = 150

_PROBE = r"""
import builtins, json, sys

def anon():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("no RssAnon in /proc/self/status")

original = builtins.__import__
stack, packages = [], {}

def attributing_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.partition(".")[0] if level == 0 else None
    fresh = top is not None and top != "mxlpy" and top not in sys.modules
    if fresh:
        importer = (globals or {}).get("__name__", "?")
        start = anon()
        stack.append(top)
    try:
        return original(name, globals, locals, fromlist, level)
    finally:
        if fresh:
            stack.pop()
            if not stack:  # only the outermost third-party import
                packages[top] = (anon() - start, importer)

before = anon()
builtins.__import__ = attributing_import
import mxlpy
builtins.__import__ = original
print(json.dumps({"total": anon() - before, "packages": packages}))
"""


def main() -> int:
    """Measure the import and compare it with the budget."""
    parser = argparse.ArgumentParser(description="Check the import mxlpy memory")
    parser.add_argument("--budget", type=float, default=BUDGET_MIB, help="MiB")
    budget = parser.parse_args().budget

    if not Path("/proc/self/status").exists():
        print("Skipping the import memory check, it needs Linux's /proc")
        return 0

    res = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True
    )
    measured = json.loads(res.stdout.strip().splitlines()[-1])
    total = measured["total"]
    print(f"import mxlpy: {total:.1f} MiB private memory (budget {budget:.0f} MiB)")
    if total <= budget:
        return 0

    print("\nLargest third-party packages (MiB, first imported by):")
    packages = sorted(measured["packages"].items(), key=lambda kv: -kv[1][0])
    for name, (size, importer) in packages[:15]:
        print(f"  {name:24s} {size:6.1f}  {importer}")
    attributed = sum(size for size, _ in measured["packages"].values())
    print(f"  {'mxlpy itself / other':24s} {total - attributed:6.1f}")
    print(
        "\nImport heavy dependencies lazily (e.g. inside the functions using them,"
        " or via a module __getattr__ like mxlpy.__init__), or raise BUDGET_MIB"
        " in scripts/check_import_memory.py if the growth is intended."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
