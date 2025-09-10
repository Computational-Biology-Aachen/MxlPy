"""Neural Process Estimation (NPE) module.

This module provides classes and functions for estimating metabolic processes using
neural networks. It includes functionality for both steady-state and time-course data.

Classes:
    TorchSteadyState: Class for steady-state neural network estimation.
    TorchSteadyStateTrainer: Class for training steady-state neural networks.
    TorchTimeCourse: Class for time-course neural network estimation.
    TorchTimeCourseTrainer: Class for training time-course neural networks.

Functions:
    train_torch_steady_state: Train a PyTorch steady-state neural network.
    train_torch_time_course: Train a PyTorch time-course neural network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import contextlib

    with contextlib.suppress(ImportError):
        from . import _equinox as equinox
        from . import _keras as keras
        from . import _torch as torch
else:
    from lazy_import import lazy_module

    equinox = lazy_module(
        "mxlpy.npe._equinox",
        error_strings={"module": "equinox", "install_name": "mxlpy[equinox]"},
    )
    keras = lazy_module(
        "mxlpy.npe._keras",
        error_strings={"module": "keras", "install_name": "mxlpy[tf]"},
    )
    torch = lazy_module(
        "mxlpy.npe._torch",
        error_strings={"module": "torch", "install_name": "mxlpy[torch]"},
    )


__all__ = [
    "keras",
    "torch",
]
