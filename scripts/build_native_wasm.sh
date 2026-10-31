#!/usr/bin/env bash
# Rebuild the WASM solver shipped inside MxlPy. Needs emcc on PATH.
set -euo pipefail
cd "$(dirname "$0")/../native/c"
MATH=(exp log sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh pow)
EXPORTS='"_mxn_set_model","_mxn_radau","_malloc","_free"'
for f in "${MATH[@]}"; do EXPORTS+=",\"_mxn_$f\""; done
emcc -O2 -w -DMXN_WASM -I. mxn_radau.c radau5.c dc_decsol.c decsol.c libf2c_stubs.c \
  -o ../../src/mxlpy/integrators/_native/assets/radau.wasm \
  --no-entry -s STANDALONE_WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s ALLOW_TABLE_GROWTH=1 \
  -s EXPORTED_FUNCTIONS="[$EXPORTS]" -Wno-deprecated-non-prototype
