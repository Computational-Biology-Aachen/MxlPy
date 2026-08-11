# ADR 0006: Split `nn/` / `surrogates/` / `npe/` by Scientific Task, Not by Layer

**Status:** Implemented
**Scope:** `src/mxlpy/nn/`, `src/mxlpy/surrogates/`, `src/mxlpy/npe/`

---

## 1. Context

Three packages provide ML capability, each with backend variants for
torch/keras/equinox (`_torch.py`/`_keras.py`/`_equinox.py`). They could plausibly have
been one "ml/" module.

## 2. Decision

Keep them as three separate packages, split by the *scientific problem* they solve
rather than by implementation layer:

- **`nn/`** — backend-specific network architecture primitives. The shared low-level
  zoo that the other two packages build on.
- **`surrogates/`** — *forward* substitution: replace an expensive or unknown
  reaction/derived-quantity submodel inside a `Model` with a trained (or polynomial/QSS)
  stand-in, so the rest of the mechanistic model runs unchanged around it.
- **`npe/`** — *inverse* problem: neural posterior/parameter estimation
  (simulation-based inference) — going from observed data back to a parameter
  distribution. A fundamentally different task from substituting a model component.

## 3. Rationale

The split tracks what a scientist is trying to *do* (approximate a piece of the forward
model vs. infer parameters from data going backward), not an implementation detail. This
keeps each package's public API scoped to one mental model, rather than one "ml/" module
where users have to figure out which class does which task. `npe/` and `surrogates/` are
independent consumers of `nn/`'s primitives — neither depends on the other.

## 4. Consequences

- New ML capability should be evaluated against this task-based split before adding a
  new top-level package: does it approximate a forward component (`surrogates/`), infer
  parameters from data (`npe/`), or is it a new architecture primitive both could use
  (`nn/`)?
