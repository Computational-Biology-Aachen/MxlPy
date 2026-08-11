# ADR 0002: Fluent `Model` Builder with Wholesale Cache Invalidation

**Status:** Implemented
**Scope:** `src/mxlpy/_kinetic_builder.py` (`KineticModelBuilder`, exported as `Model`)

---

## 1. Context

`Model` is built incrementally, typically in a Jupyter notebook, by chaining calls:

```python
model = (
    Model()
    .add_variables({"x": 1.0, "y": 1.0})
    .add_parameters({"k1": 1.0, "k2": 2.0})
    .add_reaction("v1", constant, stoichiometry={"x": 1}, args=["k1"])
    .add_reaction("v2", mass_action_1s, stoichiometry={"x": -1, "y": 1}, args=["k2", "x"])
)
```

Every mutating method returns `Self`. `Model` also holds a `_cache: ModelCache | None` —
compiled dependency graph, topological order, and generated evaluation functions — that
is expensive-ish to build but cheap to use. Some cached artifact needs to be recomputed
whenever the model's structure changes.

## 2. Decision

**2.1 — Fluent, `Self`-returning builder.** Every `add_*`/`remove_*`/`update_*` method is
decorated with `@_invalidate_cache` and returns `Self`.

**2.2 — Cache invalidation is wholesale, not incremental.** `@_invalidate_cache` simply
sets `self._cache = None` on any mutating call. There is no attempt to reason about
*which* cached artifacts a given mutation actually invalidates (e.g. "renaming a
parameter only affects X, not Y"). The cache is lazily rebuilt in full on next access.

## 3. Rationale

**Why fluent/chainable:** model construction in this domain is inherently incremental
and interactive — a scientist adds variables, then parameters, then reactions, often
iterating in a notebook cell by cell. Chaining reads as a specification of the model
(closer to how it would be described in a paper) rather than a sequence of imperative
mutation statements, and it composes well with `.copy()`-then-branch workflows common in
scanning/comparison analyses.

**Why wholesale cache invalidation:** incremental (fine-grained) cache invalidation was
tried in earlier, related projects and consistently became a source of hard-to-find
correctness bugs — a code path that mutates the model in a way the incremental
invalidation logic didn't anticipate silently leaves a stale cached artifact in place.
Given that:

- model *construction* happens once per session, not in a hot loop — the cost of
  rebuilding the cache is paid rarely,
- the cache is cheap to rebuild at the scale of models mxlpy models are built for,

the correctness guarantee of "any mutation, however exotic, always fully invalidates"
was judged more valuable than the performance gain of fine-grained invalidation. This is
a deliberate trade of a small constant-factor performance cost for eliminating an entire
class of subtle bugs.

## 4. Consequences

- Any future contributor adding a new mutating method only needs to remember one rule
  (`@_invalidate_cache`), not reason about which cached fields their change affects.
- If model sizes or construction patterns change dramatically (e.g. models built
  programmatically in a hot loop, at a much larger scale), this trade-off should be
  revisited — but doing so should be approached cautiously given the prior-project
  history of incremental invalidation bugs.
