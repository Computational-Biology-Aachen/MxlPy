`radau5.c`, `dc_decsol.c`, `decsol.c`, `f2c.h` and `libf2c_stubs.c` are copied from `pkg-js/mxlweb-core/src-c` at 279349a, plus its then-uncommitted `radau5.c` changes: non-finite steps are rejected, and `radau5_jac_growth` limits how far the step can grow over a Jacobian's lifetime. They are Hairer and Wanner's solvers (see `HAIRER-LICENCE.txt`).

`mxn_radau.c` is written for MxlPy. It is compiled unchanged into both the native wheel and `radau.wasm`.
