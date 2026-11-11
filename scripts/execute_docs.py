# Copyright 2026 Marvin van Aalst, Anna Matuszyńska
# ruff: noqa: T201  (a command line script reports with print)

"""Execute the docs notebooks in place, reusing cached results of expensive cells.

The docs are built once per release, and every release changes the library, so
caching whole notebooks keyed on the library would never hit. Instead, only
cells tagged `cache` (cell metadata `"tags": ["cache"]`) are cached: the
variables they assign and their outputs are stored, and on a hit the variables
are restored in the kernel and the outputs pasted in, without executing the
cell. All other cells always run, so the examples are still checked against the
current library.

A cached cell is recomputed when its source, any preceding code cell of its
notebook or `CACHE_VERSION` changes. A tagged cell should only assign names
(no in-place changes to existing objects), since only assigned names are
restored. Cells whose names can't be pickled are executed and not cached.

Run before `mkdocs build` with `MKDOCS_EXECUTE=false`, so mkdocs renders the
executed notebooks instead of executing them again:

    uv run python scripts/execute_docs.py
    MKDOCS_EXECUTE=false uv run mkdocs build

The notebooks are overwritten with their outputs, so only run this in CI or
discard the changes afterwards.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbclient.util import run_sync

# Bump to invalidate every cached cell, e.g. after a change in the library that
# alters the results shown in the docs
CACHE_VERSION = 1
CACHE_TAG = "cache"


def _assigned_names(source: str) -> list[str]:
    """Top-level names a cell (re)binds."""
    names: list[str] = []

    def add_target(target: ast.expr) -> None:
        names.extend(
            node.id
            for node in ast.walk(target)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )

    for stmt in ast.parse(source).body:
        match stmt:
            case ast.Assign(targets=targets):
                for t in targets:
                    add_target(t)
            case ast.AnnAssign(target=t) | ast.AugAssign(target=t):
                add_target(t)
            case ast.For(target=t) | ast.AsyncFor(target=t):
                add_target(t)
            case ast.With(items=items) | ast.AsyncWith(items=items):
                for item in items:
                    if item.optional_vars is not None:
                        add_target(item.optional_vars)
            case (
                ast.FunctionDef(name=n)
                | ast.AsyncFunctionDef(name=n)
                | ast.ClassDef(name=n)
            ):
                names.append(n)
            case ast.Import(names=aliases) | ast.ImportFrom(names=aliases):
                names.extend((a.asname or a.name).split(".")[0] for a in aliases)
            case _:
                pass
    return list(dict.fromkeys(n for n in names if n != "_"))


def _key(previous: list[str], source: str) -> str:
    h = hashlib.sha256(f"v{CACHE_VERSION}".encode())
    for s in [*previous, source]:
        h.update(hashlib.sha256(s.encode()).digest())
    return h.hexdigest()[:32]


def _run(client: NotebookClient, code: str) -> bool:
    """Run code in the kernel, outside the notebook; whether it succeeded."""
    reply = run_sync(client.kc.execute_interactive)(
        code, store_history=False, output_hook=lambda _: None, timeout=600
    )
    return reply["content"]["status"] == "ok"


def _save_names(client: NotebookClient, names: list[str], file: Path) -> bool:
    code = (
        "import cloudpickle as __cp\n"
        f"with open({str(file)!r}, 'wb') as __f:\n"
        f"    __cp.dump({{__n: globals()[__n] for __n in {names!r}}}, __f)\n"
        "del __cp, __f"
    )
    return _run(client, code)


def _load_names(client: NotebookClient, file: Path) -> bool:
    code = (
        "import cloudpickle as __cp\n"
        f"with open({str(file)!r}, 'rb') as __f:\n"
        "    globals().update(__cp.load(__f))\n"
        "del __cp, __f"
    )
    return _run(client, code)


def execute(nb_path: Path, cache_dir: Path, used: set[str]) -> tuple[float, int, int]:
    """Execute one notebook in place; return (seconds, cache hits, cached cells)."""
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=3600,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    start = time.perf_counter()
    hits = tagged = 0
    previous: list[str] = []
    with client.setup_kernel():
        for index, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            source = "".join(cell.source)
            if CACHE_TAG not in cell.metadata.get("tags", []):
                client.execute_cell(cell, index)
                previous.append(source)
                continue

            tagged += 1
            key = _key(previous, source)
            used.add(key)
            pickle_file = cache_dir / f"{key}.pkl"
            outputs_file = cache_dir / f"{key}.json"
            if (
                pickle_file.exists()
                and outputs_file.exists()
                and _load_names(client, pickle_file)
            ):
                cell.outputs = [
                    nbformat.from_dict(o) for o in json.loads(outputs_file.read_text())
                ]
                cell.execution_count = None
                hits += 1
            else:
                client.execute_cell(cell, index)
                if _save_names(client, _assigned_names(source), pickle_file):
                    outputs_file.write_text(json.dumps(cell.outputs))
                else:
                    pickle_file.unlink(missing_ok=True)
                    print(f"  {nb_path.name} cell {index}: not picklable, not cached")
            previous.append(source)
    nbformat.write(nb, nb_path)
    return time.perf_counter() - start, hits, tagged


def main() -> int:
    """Execute the docs notebooks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".docs-cache/cells"))
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete cache entries no executed cell used (only with all notebooks)",
    )
    args = parser.parse_args()

    cache_dir: Path = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    notebooks = args.notebooks or sorted(args.docs.glob("*.ipynb"))
    used: set[str] = set()
    failed: list[str] = []
    total = time.perf_counter()
    for nb_path in notebooks:
        try:
            seconds, hits, tagged = execute(nb_path, cache_dir, used)
        except CellExecutionError as e:
            failed.append(nb_path.name)
            print(f"FAILED {nb_path.name}\n{str(e)[-2000:]}", file=sys.stderr)
            continue
        cached = f" (cached cells: {hits}/{tagged} hit)" if tagged else ""
        print(f"{seconds:7.1f}s {nb_path.name}{cached}", flush=True)
    print(f"{time.perf_counter() - total:7.1f}s total")

    if args.prune and not args.notebooks:
        for file in cache_dir.iterdir():
            if file.stem not in used:
                file.unlink()
    if failed:
        print(f"Notebooks failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
