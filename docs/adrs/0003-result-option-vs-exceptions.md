# ADR 0003: `Result`/`Option` for Expected Failures, Exceptions for the Rest

**Status:** Implemented
**Scope:** `src/mxlpy/types.py` (`Result[T]`, `Option[T]`), `scan.py`, `fit/`, `mc.py`,
`simulator.py`, `integrators/`

---

## 1. Context

Scientific batch workflows (parameter scans over thousands of points, Monte Carlo
ensembles, multi-start fitting) *expect* a fraction of individual runs to fail — a
steady state that doesn't exist for a given parameter set, an integrator that can't
converge, a fit that doesn't reach an acceptable loss. These are not bugs; they're data
the caller wants to collect and keep working around, not have crash the whole batch.

Python has no compiler-enforced notion of "this function might throw" the way checked
exceptions or `Result`-returning languages do — a caller has no static way to know
whether a given call can raise, or what.

## 2. Decision

- `Result[T]` (wraps `T | Exception`) and `Option[T]` (wraps `T | None`) are used for
  **expected, routine failure modes** — anything a caller is expected to branch on
  programmatically as part of normal control flow (a scan point failing, a fit not
  converging, a steady state not found). `.unwrap_or_err()` / `.default(fn)` extract or
  raise/substitute.
- Custom exceptions (`IntegrationFailure`, `NoSteadyState`, `FitFailure`) are used for
  the lower-level, single-call APIs, and are caught and wrapped into `Result` by the
  batch-oriented outer layers (`scan.py`, `mc.py`, `fit/`).
- Exceptions are also, unavoidably, still used where:
  - the failure originates at the **C level of a dependency** (e.g. certain solver
    backends) and cannot be intercepted and turned into a `Result` any other way, or
  - wrapping in `Result` would add ceremony disproportionate to the value gained, and an
    exception was kept **deliberately, for convenience**.

## 3. Rationale

`Result`/`Option` make the possibility of failure visible in the type signature, which
is the closest Python gets to checked exceptions — the caller's type checker (Pyright)
will flag an unhandled `Result[T]` sitting unused. This is valuable specifically at the
boundary where failure is *routine*, because that's where a caller is most likely to
forget to handle it if it's just an exception.

This is deliberately **not** a blanket "no exceptions" rule, and not applied
religiously. Railway-oriented programming (threading `Result` through every call) can
get verbose in Python, which lacks native `?`-operator-style propagation sugar. The
guiding balance is: use `Result`/`Option` where the type system earns its keep (routine,
programmatically-branched-on failures), and accept plain exceptions where Python's lack
of checked-exception tracking makes the `Result` wrapping not worth it (C-level
dependency failures, deliberate convenience cases).

## 4. Consequences

- A new contributor adding a function that can fail should ask: "is this failure routine
  and something callers are expected to branch on in bulk workflows?" If yes, return
  `Result`/`Option`. If it's a genuine misuse of the API, or the failure surfaces from
  outside Python's control (C-level), a plain exception is fine and expected.
- Don't try to convert every remaining exception in the codebase to `Result` on
  principle — check whether it falls into one of the two "stays an exception"
  categories above first.
