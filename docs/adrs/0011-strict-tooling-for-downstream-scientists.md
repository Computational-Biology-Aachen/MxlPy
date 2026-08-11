# ADR 0011: `Ruff ALL` + Bandit + Pyright Strict

**Status:** Implemented
**Scope:** `pyproject.toml` (`[tool.ruff]`, `[tool.pyright]`, `[tool.bandit]`)

---

## 1. Context

mxlpy uses Ruff with `select = ["ALL"]` (a curated ignore list on top, not a lax
default), Bandit for security linting, and Pyright in strict-ish mode across the entire
source tree — a notably heavier tooling posture than many research-code Python projects.

## 2. Decision

Keep the rigorous configuration rather than relaxing it for convenience.

## 3. Rationale

This is deliberate rigor, not personal preference carried over by habit: mxlpy is a
**library other scientists build directly on top of** (mxlbricks, downstream research
repos pinning it as a submodule). Code quality issues here don't just cost the author
time — they propagate into every downstream model built with it. The cost of strict
linting/typing is paid once, centrally, by whoever touches mxlpy's source; the cost of
*not* having it would be paid repeatedly, by every downstream user hitting a bug or
ambiguity mxlpy could have caught statically.

## 4. Consequences

- New contributors should not treat `# noqa`/ignore additions to `pyproject.toml` as a
  routine way to unblock a PR — the bar for loosening these rules should stay high,
  proportional to mxlpy's role as shared infrastructure for other scientists' work.
- When Ruff/Pyright/Bandit genuinely conflict with a specific pattern that's correct and
  intentional, prefer a scoped, local suppression over a project-wide rule relaxation.
