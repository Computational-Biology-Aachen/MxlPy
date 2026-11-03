#!/usr/bin/env bash
# Rebuild the WASM solvers shipped inside MxlPy. Needs emcc on PATH.
set -euo pipefail
cd "$(dirname "$0")/../native/c"
MATH=(exp log sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh pow)
EXPORTS='"_mxn_set_model","_mxn_radau","_mxn_lsoda","_mxn_eval_rows","_malloc","_free"'
for f in "${MATH[@]}"; do EXPORTS+=",\"_mxn_$f\""; done
V=vendor
emcc -O2 -w -DMXN_WASM -I$V mxn_solvers.c $V/radau5.c $V/dc_decsol.c $V/decsol.c $V/libf2c_stubs.c \
  $V/lsoda.c $V/mxs_lapack.c \
  -o ../../src/mxlpy/integrators/_native/assets/solvers.wasm \
  --no-entry -s STANDALONE_WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s ALLOW_TABLE_GROWTH=1 \
  -s EXPORTED_FUNCTIONS="[$EXPORTS]" -Wno-deprecated-non-prototype
