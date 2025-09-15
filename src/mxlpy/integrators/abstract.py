"""Integrator Interface."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from mxlpy.types import Array, ArrayLike, Rhs

__all__ = ["IntegratorProtocol", "IntegratorType"]


class IntegratorProtocol(Protocol):
    """Protocol for numerical integrators."""

    def __init__(
        self,
        rhs: Rhs,
        y0: tuple[float, ...],
        jacobian: Callable | None = None,
    ) -> None:
        """Initialise the integrator."""
        ...

    def reset(self) -> None:
        """Reset the integrator."""
        ...

    def integrate(
        self,
        *,
        t_end: float,
        steps: int | None = None,
    ) -> tuple[Array | None, ArrayLike | None]:
        """Integrate the system."""
        ...

    def integrate_time_course(
        self, *, time_points: ArrayLike
    ) -> tuple[Array | None, ArrayLike | None]:
        """Integrate the system over a time course."""
        ...

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
    ) -> tuple[float | None, ArrayLike | None]:
        """Integrate the system to steady state."""
        ...


type IntegratorType = Callable[
    [
        Rhs,  # model
        tuple[float, ...],  # y0
        Callable | None,  # jacobian
    ],
    IntegratorProtocol,
]
