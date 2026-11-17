# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

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
    from mxlpy._lazy import backend_getattr

    __getattr__ = backend_getattr(
        __name__,
        {
            "equinox": ("_equinox", "jax"),
            "keras": ("_keras", "keras"),
            "torch": ("_torch", "torch"),
        },
    )


__all__ = [
    "equinox",
    "keras",
    "torch",
]
