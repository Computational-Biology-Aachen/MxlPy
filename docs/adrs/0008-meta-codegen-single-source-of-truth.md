# ADR 0008: `meta/` Codegen — One Model, Mechanically Projected to Every Target

**Status:** Implemented
**Scope:** `src/mxlpy/meta/` (`codegen_mxlpy.py`, `codegen_latex.py`, `sympy_tools.py`,
`_via_sym_repr.py`, `source_tools.py`, `_mathml/`)

---

## 1. Context

A model is authored once, in mxlpy's fluent builder. Several other artifacts need to
represent the *same* model:

- LaTeX equations for a paper/report,
- Julia or Rust source for performance-critical or foreign-ecosystem consumers,
- a TypeScript AST for MxlWeb's independent browser-based simulator (which, per the
  meta-repo's tool-family architecture, deliberately does **not** depend on mxlpy at
  runtime and has its own AST/integrator tree in TypeScript/Rust/WASM),
- a `JaxModel` right-hand side for training (see
  [ADR 0005](0005-jax-as-orthogonal-subsystem.md)),
- a native mxlpy `Model` reconstructed from an imported SBML file (see
  [ADR 0009](0009-sbml-dual-use-bridge.md)).

## 2. Decision

All of these are **derived**, not hand-transcribed, from the same symbolic
representation of the model, built once via `meta/`'s sympy reconstruction
(`sympy_tools.py`, `_via_sym_repr.py`) and projected outward by target-specific
generators:

- `codegen_mxlpy.py` — regenerate mxlpy Python source (used by SBML import).
- `codegen_latex.py` — LaTeX equations/report tables.
- `sympy_tools.py`'s `sympy_to_inline_julia`/`sympy_to_inline_rust`/`sympy_to_inline_jax`
  — inline expressions for Julia, Rust, and JAX-native Python.
- `_via_sym_repr.py`'s sympy-expression → MxlWeb TypeScript AST converter —
  fully custom logic, since no sympy printer exists for MxlWeb's AST.
- `_via_sym_repr.py`'s `generate_model_code_jax` — a complete JAX-native model module,
  consumed to build `JaxModel` instances.

## 3. Rationale

**Single source of truth, mechanically projected outward.** This eliminates the classic
failure mode in computational-biology tooling where a manuscript's equations, a web
demo, and the "real" simulation code silently drift apart because each was hand-written
separately. Every downstream representation is generated from the same model
definition, so they can't drift.

**Why sympy as the intermediate representation.** sympy already has robust code
printers for Rust/Julia/etc., so mxlpy only needs to reconstruct the symbolic expression
tree once and get most target languages "for free." Only the MxlWeb TS-AST path needed
fully custom logic, since sympy has no printer for it.

**Why codegen produces source text, not just in-memory objects** (relevant to both SBML
import and, historically, JAX model generation): an auditable, diffable, hand-editable
source file is valuable on its own — an imported/generated model reads like a normal
mxlpy model a scientist can inspect, tweak, and check into their own repo, rather than
an opaque object graph assembled by a black-box importer. It also sidesteps
metaprogramming/pickling failures that occur when there's no source to introspect (the
same underlying concern as [ADR 0004](0004-named-functions-no-lambdas.md)'s
no-lambdas rule).

## 4. Consequences

- Any new export target (a new language, a new downstream tool) should go through the
  same sympy reconstruction rather than writing a bespoke `Model`-to-text converter.
- `jax/`'s model representation is not a maintenance burden separate from `Model` —
  changes to a model's kinetics regenerate its `JaxModel` form; they are not maintained
  as two independent hand-written implementations.
