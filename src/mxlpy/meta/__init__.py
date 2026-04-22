"""Metaprogramming facilities.

Requirements
------------
- IDs need to be renamed (target is code)
  - generate_latex_code      | yes
  - generate_model_code      | yes
  - generate_mxlpy_code      | no
  - generate_mxlpy_code_raw  | no
  - generate_mxlweb_page     | yes
- Requires `fn` to be parse-able by SymPy
  - generate_latex_code      | yes
  - generate_model_code      | yes
  - generate_mxlpy_code      | yes
  - generate_mxlpy_code_raw  | no
  - generate_mxlweb_page     | yes
- Requires `fn` to be subset of MathMl nodes
  - generate_latex_code      | no
  - generate_model_code      | no
  - generate_mxlpy_code      | no
  - generate_mxlpy_code_raw  | no
  - generate_mxlweb_page     | yes

Why are there so many versions of this?
---------------------------------------
There are different use cases.
The `generate_model_code_*` functions are mostly used for exporting a model to other
packages or runtimes.

The `generate_mxlpy_code` and `generate_mxlpy_code_raw` are mostly used to move models
between our different packages (e.g. between mxlbricks and mxlmodels). They both encode
the same models, but with a different structure - again, for different use cases.
`mxlbricks` is supposed to be used to *build* larger models by composing reactions, while
`mxlmodels` is supposed to be used to *read* and inspect these models in a flat, easier
to digest way.
The reason for introducing the `_raw` version was to avoid parsing the model fns with
sympy, which is impossible for e.g. surrogates. So this just quite literally copy-pastes
code around.

The `generate_mxlweb_page` function is directly used to generate the initial version of
mxlweb. The function should be considered volatile and subject to change.

"""

from __future__ import annotations

from .codegen_latex import generate_latex_code, to_tex_export
from .codegen_model import (
    generate_model_code_c,
    generate_model_code_cpp,
    generate_model_code_jl,
    generate_model_code_matlab,
    generate_model_code_py,
    generate_model_code_rs,
    generate_model_code_ts,
)
from .codegen_mxlpy import generate_mxlpy_code
from .codegen_mxlpy_raw import generate_mxlpy_code_raw
from .codegen_mxlweb import generate_mode_code_mxlweb, generate_mxlweb_page

__all__ = [
    "generate_latex_code",
    "generate_mode_code_mxlweb",
    "generate_model_code_c",
    "generate_model_code_cpp",
    "generate_model_code_jl",
    "generate_model_code_matlab",
    "generate_model_code_py",
    "generate_model_code_rs",
    "generate_model_code_ts",
    "generate_mxlpy_code",
    "generate_mxlpy_code_raw",
    "generate_mxlweb_page",
    "to_tex_export",
]
