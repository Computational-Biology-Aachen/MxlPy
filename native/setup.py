# Copyright 2025 Marvin van Aalst, Anna Matuszyńska  # noqa: INP001

"""Build the solver as one stable-ABI extension, so a wheel covers all Python versions."""

import sys

from setuptools import Extension, setup

sources = ["mxn_radau.c", "radau5.c", "dc_decsol.c", "decsol.c", "libf2c_stubs.c"]
setup(
    ext_modules=[
        Extension(
            "mxlpy_native._radau",
            sources=[f"c/{s}" for s in sources],
            include_dirs=["c"],
            py_limited_api=True,
            define_macros=[("Py_LIMITED_API", "0x030C0000")],
            extra_compile_args=[]
            if sys.platform == "win32"
            else ["-std=gnu17", "-w", "-O2"],
            libraries=[] if sys.platform == "win32" else ["m"],
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
