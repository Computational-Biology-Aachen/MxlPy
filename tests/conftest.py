# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import os

# Collection imports every test module in every xdist worker, so keras's
# default TensorFlow backend would cost each worker ~500 MB. The jax backend
# is loaded anyway and adds ~40 MB. An explicit KERAS_BACKEND still wins.
os.environ.setdefault("KERAS_BACKEND", "jax")
