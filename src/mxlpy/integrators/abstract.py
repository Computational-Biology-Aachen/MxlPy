"""Integrator Interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from mxlpy.types import Array, ArrayLike, Result, Rhs

__all__ = [
    "AbstractIntegrator",
    "IntegratorProtocol",
    "IntegratorType",
    "MockIntegrator",
    "TimeCourse",
    "TimePoint",
]


@dataclass(slots=True)
class TimeCourse:
    """FIXME: Something I will fill out."""

    time: Array
    values: ArrayLike


@dataclass(slots=True)
class TimePoint:
    """FIXME: Something I will fill out."""

    time: float
    values: Array


class AbstractIntegrator(ABC):
    """Protocol for numerical integrators."""

    @abstractmethod
    def __init__(
        self,
        rhs: Rhs,
        y0: tuple[float, ...],
        jacobian: Callable | None = None,
    ) -> None:
        """Initialise the integrator."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the integrator."""
        ...

    @abstractmethod
    def integrate(
        self,
        *,
        t_end: float,
        steps: int | None = None,
    ) -> Result[TimeCourse]:
        """Integrate the system."""
        ...

    @abstractmethod
    def integrate_time_course(
        self,
        *,
        time_points: ArrayLike,
    ) -> Result[TimeCourse]:
        """Integrate the system over a time course."""
        ...

    @abstractmethod
    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
    ) -> Result[TimeCourse]:
        """Integrate the system to steady state."""
        ...


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
    ) -> Result[TimeCourse]:
        """Integrate the system."""
        ...

    def integrate_time_course(
        self,
        *,
        time_points: ArrayLike,
    ) -> Result[TimeCourse]:
        """Integrate the system over a time course."""
        ...

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,
        rel_norm: bool,
    ) -> Result[TimeCourse]:
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


class MockIntegrator(AbstractIntegrator):
    """FIXME: Something I will fill out."""

    def __init__(
        self,
        rhs: Callable,  # noqa: ARG002
        y0: tuple[float, ...],
        jacobian: Callable | None = None,  # noqa: ARG002
    ) -> None:
        """FIXME: Something I will fill out."""
        self.y0 = y0

    def reset(self) -> None:
        """FIXME: Something I will fill out."""
        return

    def integrate(
        self,
        *,
        t_end: float,  # noqa: ARG002
        steps: int | None = None,  # noqa: ARG002
    ) -> Result[TimeCourse]:
        """FIXME: Something I will fill out."""
        return Result(
            TimeCourse(
                time=np.array([0.0]),
                values=np.ones((1, len(self.y0))),
            )
        )

    def integrate_time_course(
        self,
        *,
        time_points: ArrayLike | None = None,  # noqa: ARG002
    ) -> Result[TimeCourse]:
        """FIXME: Something I will fill out."""
        return Result(
            TimeCourse(
                time=np.array([0.0]),
                values=np.ones((1, len(self.y0))),
            )
        )

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,  # noqa: ARG002
        rel_norm: bool,  # noqa: ARG002
    ) -> Result[TimeCourse]:
        """FIXME: Something I will fill out."""
        return Result(
            TimeCourse(
                time=np.array([0.0], dtype=float),
                values=np.ones((1, len(self.y0))),
            )
        )
