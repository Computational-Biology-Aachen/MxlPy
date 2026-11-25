# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import os

# Collection imports every test module in every xdist worker, so keras's
# default TensorFlow backend would cost each worker ~500 MB. The jax backend
# is loaded anyway and adds ~40 MB. An explicit KERAS_BACKEND still wins.
os.environ.setdefault("KERAS_BACKEND", "jax")
# No GUI windows (and figures drawn from server threads, e.g. the lab's report)
os.environ.setdefault("MPLBACKEND", "Agg")

# Every xdist worker starts its own process pools in parallel scans and fits,
# so share the CPUs between them instead of each starting one worker per CPU
if (n_workers := os.environ.get("PYTEST_XDIST_WORKER_COUNT")) is not None:
    os.environ.setdefault(
        "LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 1) // int(n_workers)))
    )
