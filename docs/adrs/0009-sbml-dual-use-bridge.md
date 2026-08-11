# ADR 0009: `sbml/` as mxlpy's Consumer of the (Now-Independent) `pysbml`

**Status:** Implemented
**Scope:** `src/mxlpy/sbml/` (`_import.py`, `_export.py`, `_data.py`,
`_name_conversion.py`)

---

## 1. Context

`sbml/_import.py` reads a file via `pysbml.transform.data.Model` and then runs
`_codegen` to produce mxlpy source text, ending in a `KineticModelBuilder`. `pysbml` is
a separate, standalone repo/submodule in the tool family, also serving as the reference
implementation for MxlWeb's own SBML layer.

## 2. Decision and History

`pysbml` originated **inside** mxlpy. It was extracted into its own standalone project
once it proved general and reusable enough for other consumers (mxlpy itself, and
MxlWeb's SBML reference implementation) — it was not designed as a separate project
from day one.

`mxlpy/sbml/` is the mxlpy-specific consumer: it takes `pysbml`'s parsed/transformed
representation and turns it into a native, editable `Model` — via source-text codegen
(see [ADR 0008](0008-meta-codegen-single-source-of-truth.md)), not by constructing a
`KineticModelBuilder` object graph directly in memory.

## 3. Rationale

**Why extract `pysbml`:** once it was clearly reusable beyond mxlpy's own needs
(and useful as a shared reference for MxlWeb's SBML implementation), keeping it
in-tree would have coupled an independently-useful SBML layer to mxlpy's release cycle
and dependencies for no benefit.

**Why codegen (source text) rather than direct in-memory construction**, specifically
for SBML import — two reasons:
1. **Auditability.** An imported SBML model becomes ordinary, readable mxlpy source a
   scientist can inspect and hand-edit, not an opaque object built by an importer.
2. **Metaprogramming/pickling correctness.** Constructing model functions
   programmatically without generating real source text runs into the same
   introspection/pickling failure modes that motivate
   [ADR 0004](0004-named-functions-no-lambdas.md) — without source to inspect, later
   codegen, symbolic analysis, or process-pool serialization of the imported model can
   break in hard-to-diagnose ways.

## 4. Consequences

- If `pysbml` needs a breaking change for mxlpy's sake, remember it now has other
  consumers (MxlWeb's reference implementation) and is no longer mxlpy-internal.
- SBML import should continue to go through codegen, not be "optimized" into direct
  object construction — that would reintroduce the auditability/introspection problems
  this design avoided.
