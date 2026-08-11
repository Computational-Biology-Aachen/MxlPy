# ADR 0005: `jax/` as an Orthogonal Subsystem, Not an Integrator Plugin

**Status:** Implemented
**Scope:** `src/mxlpy/jax/` (`models.py`, `train.py`, `simulation.py`, `ensemble.py`,
`io.py`), `src/mxlpy/integrators/`

---

## 1. Context

`integrators/` provides swappable numerical solver backends (scipy, assimulo, diffrax)
for the *same* `Model` object — one mechanistic model, several interchangeable ODE
solvers behind a common `AbstractIntegrator`/`IntegratorProtocol` interface.

`jax/` looks superficially similar (it also runs simulations) but has grown into
something structurally different: its own `JaxModel` protocol and `eqx.Module`
hierarchy (`Base`, `Ode`, `FluxOde`, `Node`, several `LatentMapper` variants), plus a
full gradient-based training apparatus (`train.py`: curriculum learning, UDE/NDE
training, protocol fitting) with no equivalent anywhere in `integrators/`.

## 2. Decision

`jax/` is **not** implemented as another `AbstractIntegrator` backend plugged into the
same `Model` abstraction. It defines its own model representation (`JaxModel`) and its
own training/simulation apparatus, orthogonal to `integrators/` and `Model`.

## 3. Rationale

**Why it couldn't be just another integrator backend:** JAX's `jit`/`grad`/`vmap`
transformations require the *entire* computational path — including the ODE
right-hand side, and any embedded neural components in a UDE/NDE — to be JAX-native and
traceable ahead of time. The general `Model` (fluent mutation, sympy-based derived
quantities, a mutable `_cache`) is not structurally built for that; training a UDE/NDE
needed a model representation designed for differentiable programming from the start,
not a `Model` retrofitted to be traceable.

**How the two representations stay in sync:** the `JaxModel` right-hand side is not
hand-written per model either — it's mechanically generated from the same `Model`
definition via `meta/`'s `generate_model_code_jax`/`sympy_to_inline_jax` (see
[ADR 0008](0008-meta-codegen-single-source-of-truth.md)). This avoids maintaining two
parallel, hand-synchronized implementations of the same model.

**Historical note:** this was not the original design intent. `jax/` began as *just*
another integrator backend alongside scipy/assimulo — a straightforward addition. It
grew into a fully orthogonal subsystem (its own model class, its own training loop) only
as JAX-specific training/UDE use cases were added on top.

## 4. Future Directions (not currently recommended)

Two natural extensions exist and were discussed but are **not currently recommended**:

- **Additional `JaxModelBuilder` classes** covering more model shapes/latent-mapper
  variants — plausible, incremental growth of the existing pattern.
- **Making `jax` the default simulation backend** instead of an opt-in orthogonal
  subsystem — explicitly **not recommended**: JAX errors are notoriously hard to debug
  (opaque tracing errors, shape mismatches surfacing far from their cause), which makes
  it a poor default for the general (non-ML-training) use case that `integrators/`
  serves well today.

This note is left for the next maintainer to weigh, not as a directive either way.
