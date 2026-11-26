#!/usr/bin/env bash
# Rebuild the WASM solvers shipped inside MxlPy from an mxlpy-native checkout.
# Needs emcc on PATH. MXLPY_NATIVE_DIR defaults to the sibling checkout in the
# cpbl-admin meta-repo (pkg-py/mxlpy-native).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
src="${MXLPY_NATIVE_DIR:-$here/../mxlpy-native}"
[ -f "$src/c/mxn_solvers.c" ] || {
    echo "no mxlpy-native checkout at $src; clone" \
        "https://github.com/Computational-Biology-Aachen/mxlpy-native or set MXLPY_NATIVE_DIR" >&2
    exit 1
}
command -v emcc >/dev/null || { echo "emcc not found; install and activate emsdk first" >&2; exit 1; }
cd "$src/c"
MATH=(exp log sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh pow)
EXPORTS='"_mxn_abi_version","_mxn_set_model","_mxn_radau","_mxn_lsoda","_mxn_cvode","_mxn_eval_rows","_malloc","_free"'
for f in set_fns init chunk nfev residual_norm params predictions jacobian free; do EXPORTS+=",\"_mxn_fit_$f\""; done
for f in set_fns vjp; do EXPORTS+=",\"_mxn_adjoint_$f\""; done
for f in forward backward replay free tape_steps tape_bytes; do EXPORTS+=",\"_mxn_da_$f\""; done
for f in "${MATH[@]}"; do EXPORTS+=",\"_mxn_$f\""; done
V=vendor
# The Levenberg-Marquardt fit loop (mxn_fit_*), continuous (mxn_adjoint_*) and discrete
# (mxn_da_*) adjoint gradients and the solver layer they run on.
FIT=($V/dop853.c $V/dopri5.c $V/radau5_wrapper.c $V/solver_common.c $V/fit_experiments.c
  $V/jacobian_wrapper.c $V/adjoint_wrapper.c $V/discrete_adjoint.c
  $V/minpack/{lmder,dpmpar,enorm,lmpar,qrfac,qrsolv}.c)
# Every SUNDIALS file mxl-solvers vendors is needed (CVODE only).
mapfile -t SUNDIALS < <(find $V/sundials/src -name '*.c' | LC_ALL=C sort)
emcc -O2 -w -DMXN_WASM -I$V -I$V/sundials/include -I$V/sundials/src/sundials \
  mxn_solvers.c $V/radau5.c $V/dc_decsol.c $V/decsol.c $V/libf2c_stubs.c \
  $V/lsoda.c $V/mxs_lapack.c "${FIT[@]}" "${SUNDIALS[@]}" \
  -o "$here/src/mxlpy/integrators/_native/assets/solvers.wasm" \
  --no-entry -s STANDALONE_WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s ALLOW_TABLE_GROWTH=1 \
  -s EXPORTED_FUNCTIONS="[$EXPORTS]" -Wno-deprecated-non-prototype
# solvers.wasm contains these solvers, so their licences ship next to it (and in the wheel's
# metadata, see license-files in pyproject.toml).
cp $V/*LICENCE*.txt $V/*NOTICE*.txt "$here/src/mxlpy/integrators/_native/assets/"
cp $V/minpack/COPYRIGHT.txt "$here/src/mxlpy/integrators/_native/assets/MINPACK-COPYRIGHT.txt"
echo "built solvers.wasm from mxlpy-native $(git -C "$src" rev-parse --short HEAD)"
