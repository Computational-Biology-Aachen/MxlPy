# mxlpy-native

The compiled Radau and LSODA solvers behind MxlPy's `Native` integrator. Install it with `pip install "mxlpy[native]"`, not on its own.

`c/vendor/` is copied from [mxl-solvers](https://github.com/Computational-Biology-Aachen/mxl-solvers) by its `scripts/vendor.sh`. `vendor/VENDORED` records the commit and file hashes, which `tests/integrators/test_native_vendor.py` checks. Fix solvers there, not here. `c/mxn_solvers.c` is MxlPy's own glue, compiled unchanged into both the wheel and `solvers.wasm`.
