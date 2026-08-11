# ADR 0004: Named Functions Only for Reactions/Derived Quantities, Never Lambdas

**Status:** Implemented
**Scope:** `src/mxlpy/fns.py`, `Model.add_reaction`/`add_derived`, `meta/`,
`symbolic/`, `parallel.py`

---

## 1. Context

Lambdas are used freely elsewhere in the codebase for internal glue (e.g. `scan.py`,
`simulator.py`'s jacobian wrapper). But any function passed as the rate/derived function
to `Model.add_reaction()` / `add_derived()` — the functions that actually define a
model's kinetics — must be a `def`-named, importable, top-level function, never a
lambda or closure.

## 2. Decision

Reaction and derived-quantity functions registered on a `Model` must be named function
references (e.g. `mass_action_1s`, `michaelis_menten_2s` from `fns.py`), not lambdas or
closures over mutable state.

## 3. Rationale

This single constraint pays off in four places at once:

1. **Codegen introspection (`meta/`).** Generating Python/Julia/Rust/TypeScript/LaTeX
   from a model requires getting a function's source (`inspect.getsource`) and a stable
   name to emit. Lambdas have no reliable `__name__` and their source is much harder to
   extract cleanly (especially multi-line or closure-capturing ones) than a `def`. See
   [ADR 0008](0008-meta-codegen-single-source-of-truth.md).
2. **Symbolic analysis (`symbolic/`).** Converting a model to a `SymbolicModel` for
   identifiability analysis walks the same source/AST path as codegen and has the same
   requirement. See [ADR 0010](0010-symbolic-module-scope.md).
3. **Serialization for `parallel.py`.** Distributing model evaluation across a process
   pool requires the function to serialize cleanly and reconstruct identically in the
   worker process. Named top-level functions are unambiguous; lambdas and closures are
   fragile to pickle/introspect correctly, causing metaprogramming and pickling issues
   in practice.
4. **Reusability and readability.** Named functions in `fns.py` form a shared,
   greppable, importable vocabulary (`mass_action_1s`, `michaelis_menten_2s`, ...) that
   other models reuse, rather than every model reinventing an anonymous inline
   expression. Code is read far more often than it's written — naming a function after
   its biochemical intent, even at the cost of a couple of extra lines, pays for itself
   every time someone else (or future-you) reads the model definition.

## 4. Consequences

- New rate functions should be added to `fns.py` (or an equivalent named, importable
  location) rather than inlined as lambdas, even for one-off/model-specific kinetics.
- This constraint does **not** extend to internal, non-model-facing glue code — lambdas
  remain fine there.
