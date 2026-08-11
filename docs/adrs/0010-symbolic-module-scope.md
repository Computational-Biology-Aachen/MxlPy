# ADR 0010: `symbolic/` Scope, and Its Relationship to `meta/`

**Status:** Implemented
**Scope:** `src/mxlpy/symbolic/` (`symbolic_model.py`, `strikepy.py`)

---

## 1. Context

Both `meta/` (see [ADR 0008](0008-meta-codegen-single-source-of-truth.md)) and
`symbolic/` build a sympy representation of a `Model`. This looks redundant — why two
sympy-touching modules instead of one "symbolic" package?

## 2. Decision

Keep them separate. They exist for two unrelated reasons:

- **`meta/`** uses sympy purely as an **intermediate representation for code
  generation** — build the expression tree once, hand it to a target-language printer.
- **`symbolic/`** uses sympy for **actual mathematical analysis of model structure**.
  Today this is structural identifiability analysis (`strikepy.py`, a reimplementation
  of STRIKE-PY: can these parameters be uniquely determined from the chosen observables,
  independent of any specific dataset?) — a scientific question answered by symbolic
  manipulation itself, not a translation step.

## 3. Why the Two Modules Need Different Forms of the Symbolic Model

`meta/`'s codegen wants every derived/reaction/surrogate expression to refer only to its
own immediate arguments by name — exactly what per-language, per-statement codegen
wants (one line of code per expression, each referencing the previous ones by name).

Structural analyses (e.g. the Jacobian in `symbolic_model.py`) instead need **one fully
expanded closed-form expression per differential equation**. `to_symbolic_model()`
therefore walks the model's components in dependency order and substitutes each
already-expanded value into its dependents, until only bare variables/parameters/data/
time remain (`symbolic_model.py:102-107` documents this substitution explicitly). This
expansion is unnecessary — and would be wasted work — for codegen's per-statement form.

## 4. Historical Note and Naming

`symbolic/` predates `meta/`. It was later reimplemented on top of `meta/`'s symbolic
reconstruction rather than maintaining its own from scratch. The current split is
**historic and minor**, not a deliberate long-term architectural boundary — a future
maintainer might reasonably fold `symbolic/` into `meta/` or rename it to something that
better reflects "structural analysis" rather than "symbolic," since `meta/` is now also
fundamentally a symbolic-representation consumer.

## 5. Future Directions (currently out of scope, not rejected on principle)

Conservation-law detection/elimination and symbolic steady-state solving were evaluated
(see `mxlpy-research/features/rejected/conservation_law_reduction.md`) and are **not
currently planned**, not because they're a bad fit, but because:

- conservation laws are currently treated as an **explicit design decision made by the
  modeller** (they choose how to represent conserved pools), and
- the models mxlpy is typically used to build are not yet at a scale (state count,
  stiffness) where automatic reduction would pay for itself.

If model scale grows, this should be revisited — the rejection is about current
cost/benefit, not a structural incompatibility.

## 6. Consequences

- Don't assume `symbolic/` and `meta/`'s reconstructions are interchangeable — check
  which form (per-statement vs. fully-expanded) a new analysis actually needs before
  reusing either.
- A rename/merge of `symbolic/` into `meta/` is a reasonable future cleanup, not a
  currently-planned one.
