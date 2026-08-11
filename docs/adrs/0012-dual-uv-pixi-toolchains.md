# ADR 0012: Dual `uv`/`pixi` Toolchains

**Status:** Implemented
**Scope:** `pyproject.toml`, `pixi.toml`/`pixi.lock`, `uv.lock`

---

## 1. Context

mxlpy can be installed two ways:

```bash
uv sync --all-extras --all-groups          # PyPI-only, no assimulo
pixi install --frozen                      # conda-forge, includes assimulo solver
```

## 2. Decision

Support both `uv` (PyPI) and `pixi` (conda-forge) as first-class installation paths,
rather than picking one.

## 3. Rationale

The `assimulo` ODE solver backend is **not available on PyPI** — it's only distributable
via conda-forge. `pixi` is the only supported path that gets a user `assimulo` (and any
other conda-forge-only scientific dependency). `uv` remains the faster, simpler default
for everyone who doesn't need that specific solver. This isn't tooling indecision — it's
a direct consequence of `assimulo`'s packaging constraints; there is no single tool that
covers both dependency universes today.

## 4. Consequences

- Both lock files (`uv.lock`, `pixi.lock`) must be kept in sync with `pyproject.toml`
  (pre-commit hooks handle this).
- If `assimulo` ever becomes PyPI-installable, or is dropped as a supported integrator
  backend, this dual-toolchain requirement should be revisited — it exists specifically
  because of that one dependency.
