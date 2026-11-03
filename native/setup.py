# Copyright 2025 Marvin van Aalst, Anna Matuszyńska  # noqa: INP001

"""Build the solver as one stable-ABI extension, so a wheel covers all Python versions."""

import sys

from setuptools import Extension, setup

vendored = ["radau5.c", "dc_decsol.c", "decsol.c", "libf2c_stubs.c", "lsoda.c", "mxs_lapack.c"]
setup(
    ext_modules=[
        Extension(
            "mxlpy_native._solvers",
            sources=["c/mxn_solvers.c", *(f"c/vendor/{s}" for s in vendored)],
            include_dirs=["c/vendor"],
            py_limited_api=True,
            define_macros=[("Py_LIMITED_API", "0x030C0000")],
            # lsoda.c needs C11 (restrict, compound literals), which MSVC only enables on request.
            extra_compile_args=["/std:c11"]
            if sys.platform == "win32"
            else ["-std=gnu17", "-w", "-O2"],
            libraries=[] if sys.platform == "win32" else ["m"],
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
