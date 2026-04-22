"""Codegen test suite.

Component x codegen-function coverage matrix
=============================================

Legend
------
to-check    : not yet tested
in-progress : test being written
implemented : test exists and passes

Components
----------
vars        : variable with float initial value
pars        : parameter with float value
derived     : derived computed value
reactions   : reaction with stoichiometry
surrogates  : QSS surrogate
readouts    : readout (post-ODE observable)
ia_var      : variable with InitialAssignment
ia_par      : parameter with InitialAssignment
vars_unit   : variable with unit annotation
pars_unit   : parameter with unit annotation
derived_unit: derived with unit annotation
rxn_unit    : reaction with unit annotation
readout_unit: readout with unit annotation

Notes
-----
surrogates/latex     : not exported (FIXME in to_tex_export); no crash
surrogates/mxlpy     : warned and skipped; no crash
readouts/jl+py+rs+ts : not part of ODE RHS; no crash, name absent
readouts/mxlpy+raw   : not exported; no crash, name absent
readouts/mxlweb      : not exported; no crash, name absent
ia_par/latex+jl+...  : InitialAssignment params excluded from
                       get_parameter_values(); p1 absent from output
ia_par/mxlpy+raw     : uses get_raw_parameters(); InitialAssignment preserved
vars_unit/mxlpy      : unit=units.X preserved; units import added
pars_unit/mxlpy      : unit=units.X preserved; units import added
derived_unit/mxlpy   : unit NOT preserved (SymbolicFn has no unit field)
rxn_unit/mxlpy       : unit NOT preserved (SymbolicFn has no unit field)
*/mxlpy_raw          : units NOT preserved (codegen_mxlpy_raw limitation)
*/latex+mxlweb+ode   : units not part of output format

| component    | latex       | jl          | py          | rs          | ts          | mxlpy       | mxlpy_raw   | mxlweb      |
|--------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| vars         | implemented | implemented | implemented | implemented | implemented | implemented | implemented | implemented |
| pars         | implemented | implemented | implemented | implemented | implemented | implemented | implemented | implemented |
| derived      | implemented | implemented | implemented | implemented | implemented | implemented | implemented | implemented |
| reactions    | implemented | implemented | implemented | implemented | implemented | implemented | implemented | implemented |
| surrogates   | implemented | implemented | implemented | implemented | implemented | implemented | implemented | implemented |
| readouts     | implemented | implemented | implemented | implemented | implemented | implemented | implemented | implemented |
| ia_var       | implemented | implemented | implemented | implemented | implemented | implemented | implemented | implemented |
| ia_par       | implemented | implemented | implemented | implemented | implemented | implemented | implemented | implemented |
| vars_unit    | implemented | implemented | implemented | implemented | implemented | implemented | implemented | implemented |
| pars_unit    | implemented | implemented | implemented | implemented | implemented | implemented | implemented | implemented |
| derived_unit | implemented | implemented | implemented | implemented | implemented | implemented | implemented | implemented |
| rxn_unit     | implemented | implemented | implemented | implemented | implemented | implemented | implemented | implemented |
| readout_unit | implemented | implemented | implemented | implemented | implemented | implemented | implemented | implemented |

"""
