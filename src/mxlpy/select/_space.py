# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Model spaces: a base model and switchable features."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "BASE",
    "ModelSpace",
    "Move",
    "Switch",
    "free_parameter",
]

type _Model = KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder

type Modification = Callable[[Any], object]
"""Changes a model in place, or returns the changed model."""

type Move = Literal["forward", "backward", "swap"]
"""Neighbours of a model: one switch more, one less, or one exchanged."""

BASE = "base"
"""Id of the model with all switches off."""


@dataclass(frozen=True, kw_only=True)
class Switch:
    """A model feature that is either on or off.

    Attributes
    ----------
    on
        Applied to the model when the switch is on, e.g. adding a reaction.
        Changes the model in place or returns the changed model
    off
        Applied when the switch is off
    fix_on
        Values of parameters (or initial values) that are fixed, i.e. not
        estimated, when the switch is on
    fix_off
        Values fixed when the switch is off, e.g. a feedback strength of 0

    """

    on: Modification | None = None
    off: Modification | None = None
    fix_on: Mapping[str, float] = field(default_factory=dict)
    fix_off: Mapping[str, float] = field(default_factory=dict)


def free_parameter(name: str, fixed_value: float = 0.0) -> Switch:
    """Switch estimating a parameter when on and fixing it when off.

    Fixing a parameter to the value at which a feature vanishes (e.g. a
    feedback strength of 0) keeps the models nested, so likelihood-ratio
    tests apply, and the larger model is warm-started from the smaller one's
    optimum with the parameter at `fixed_value`.

    Parameters
    ----------
    name
        Parameter (or variable whose initial value is fitted)
    fixed_value
        Its value when the switch is off

    """
    return Switch(fix_off={name: fixed_value})


def _apply(model: _Model, fn: Modification | None) -> _Model:
    if fn is None:
        return model
    res = fn(model)
    if isinstance(res, KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder):
        return res
    return model


@dataclass
class ModelSpace[M: _Model]:
    """Candidate models: a base model and switchable features.

    A candidate is a set of switches that are on. Its model is a copy of
    `base` with the `on` (or `off`) modification of every switch applied in
    the order of `switches`, and the parameters in the `fix_on` (`fix_off`)
    values fixed. The structure search moves between candidates that differ
    by one switch.

    Attributes
    ----------
    base
        The model with all switches off (before their `off` modifications)
    switches
        Switch per name. A plain function is a switch that applies it when
        on. Names can't contain `+` or be `"base"`, as they form the model ids
    allowed
        Excludes combinations of switches, e.g. alternatives of which at most
        one can be on: `lambda on: len(on & {"mm", "hill"}) <= 1`

    Examples
    --------
        >>> space = select.ModelSpace(
        ...     base=model,
        ...     switches={
        ...         "feedback": select.free_parameter("k_fb", fixed_value=0.0),
        ...         "degradation": lambda m: m.add_reaction("deg", ...),
        ...     },
        ... )

    """

    base: M
    switches: Mapping[str, Switch | Modification]
    allowed: Callable[[frozenset[str]], bool] | None = None
    _switches: dict[str, Switch] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalise the switches and check their names."""
        for name in self.switches:
            if not name or "+" in name or name == BASE:
                msg = (
                    f"Switch names can't be empty, contain '+' or be {BASE!r}: {name!r}"
                )
                raise ValueError(msg)
        self._switches = {
            k: v if isinstance(v, Switch) else Switch(on=v)
            for k, v in self.switches.items()
        }

    @property
    def names(self) -> list[str]:
        """Switch names, in the order they are applied."""
        return list(self._switches)

    def _check(self, on: Iterable[str]) -> frozenset[str]:
        on = frozenset(on)
        if unknown := sorted(on - set(self._switches)):
            msg = f"Unknown switches {unknown}"
            raise ValueError(msg)
        return on

    def id(self, on: Iterable[str]) -> str:
        """Id of a candidate: its switches that are on joined by `+`, or `base`."""
        on = self._check(on)
        return "+".join(k for k in self._switches if k in on) or BASE

    def is_allowed(self, on: Iterable[str]) -> bool:
        """Whether a combination of switches is a candidate."""
        on = self._check(on)
        return self.allowed is None or bool(self.allowed(on))

    def build(self, on: Iterable[str]) -> tuple[M, dict[str, float]]:
        """Model of a candidate and the values it fixes.

        Fixed parameters are set in the model. Fixed variables (initial
        values) are returned only, as fits take them as `y0`.
        """
        on = self._check(on)
        model: Any = deepcopy(self.base)
        fixed: dict[str, float] = {}
        for name, switch in self._switches.items():
            active = name in on
            model = _apply(model, switch.on if active else switch.off)
            fixed.update(switch.fix_on if active else switch.fix_off)
        parameters = set(model.get_parameter_names())
        for k, v in fixed.items():
            if k in parameters:
                model.update_parameter(k, v)
        return model, fixed

    def neighbours(self, on: Iterable[str], move: Move) -> list[frozenset[str]]:
        """Allowed candidates one move away from `on`.

        `forward` turns one switch on, `backward` one off and `swap` turns
        one on and another off at the same time.
        """
        on = self._check(on)
        off = [k for k in self._switches if k not in on]
        match move:
            case "forward":
                out = [on | {k} for k in off]
            case "backward":
                out = [on - {k} for k in self._switches if k in on]
            case "swap":
                out = [(on - {a}) | {b} for a in self._switches if a in on for b in off]
        return [c for c in out if self.is_allowed(c)]

    def candidates(self) -> list[frozenset[str]]:
        """All allowed candidates, by number of switches that are on."""
        names: Sequence[str] = self.names
        return [
            frozenset(c)
            for r in range(len(names) + 1)
            for c in itertools.combinations(names, r)
            if self.is_allowed(c)
        ]
