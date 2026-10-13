# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Base classes, protocols and type aliases shared by every JAX model.

:class:`Base` implements the whole integration API once: ``integrate``,
``integrate_protocol``, ``integrate_to_steady_state``,
``integrate_protocol_from_steady_state`` and their ``simulate_*``
counterparts returning named :class:`~mxlpy.jax.simulation.Simulation`
results. A concrete model only supplies its right-hand side
(``__call__``) and :meth:`Base.readouts`. Latent models additionally
override the ``_encode``/``_decode`` hooks, which map observations into
the ODE state and back and are the identity here.

:class:`FluxBase` extends this for models whose dynamics are
``nv(fluxes)``: it adds the abstract ``fluxes`` and returns
:class:`~mxlpy.jax.simulation.FluxSimulation` results carrying the fluxes
(plus any per-component fluxes a model reports via ``_flux_parts``).

Also defined here: the structural :class:`JaxModel`/:class:`FluxModel`
protocols that training and ensemble code accept, the ``Rhs``/``Nv``/
``ReadoutFn`` callable types, the :data:`CombineOp` names used by the UDEs,
and :func:`boundaries_to_ts` for building protocol time grids.
"""

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal, Protocol, cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import PyTree

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy.jax.simulation import FluxSimulation, Simulation
from mxlpy.meta._via_sym_repr import _get_extended_returns

type Method = Callable[[], diffrax.AbstractRungeKutta]
type Rhs = Callable[[PyTree, PyTree, PyTree], jax.Array]
type Nv = Callable[[PyTree], PyTree]
type CombineOp = Literal["+", "-", "*", "/", "rel", "exp", "pow"]


class ReadoutFn(Protocol):
    """Protocol for a model's readout function ``(t, y, args) -> readouts``."""

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Calculate readouts."""
        ...


class JaxModel(Protocol):
    """Protocol for JAX-compatible ODE models.

    The surface shared by every model in this module, and what the
    training utilities in ``train.py`` rely on.
    """

    @property
    def variable_names(self) -> tuple[str, ...]:
        """Names of the observed variables."""
        ...

    @property
    def readout_names(self) -> tuple[str, ...]:
        """Names of the values returned by :meth:`readouts`."""
        ...

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Evaluate the ODE right-hand side at time ``t`` and ODE state ``y``."""
        ...

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Readouts at time ``t`` for the observations ``y``."""
        ...

    def integrate(
        self,
        ts: jax.Array,
        y0: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
        dt0: jax.Array | float | None = None,
        alpha: float | None = None,
    ) -> jax.Array:
        """Integrate from ``ts[0]`` to ``ts[-1]`` and return observations."""
        ...

    def integrate_protocol(
        self,
        ts: list[jax.Array],
        y0: jax.Array,
        protocol: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None = None,
        t0: float = 0.0,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
        dt0: jax.Array | float | None = None,
        alpha: float | None = None,
    ) -> jax.Array:
        """Integrate through a sequence of protocol steps and return observations."""
        ...

    def integrate_to_steady_state(
        self,
        y0: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None = None,
        t1: float = 1e6,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        steady_state_rtol: float | None = None,
        steady_state_atol: float | None = None,
        method: Method = diffrax.Tsit5,
        dt0: jax.Array | float | None = None,
        alpha: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Integrate until a steady state is reached."""
        ...

    def integrate_protocol_from_steady_state(
        self,
        ts: list[jax.Array],
        y0: jax.Array,
        protocol: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None = None,
        preeq_args: jax.Array | None = None,
        t0: float = 0.0,
        t1: float = 1e6,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        steady_state_rtol: float | None = None,
        steady_state_atol: float | None = None,
        method: Method = diffrax.Tsit5,
        dt0: jax.Array | float | None = None,
        alpha: float | None = None,
        freeze_preeq_gradient: bool = False,
    ) -> jax.Array:
        """Pre-equilibrate ``y0`` to steady state, then integrate a protocol."""
        ...

    def simulate_time_course(
        self,
        ts: jax.Array,
        y0: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
        dt0: jax.Array | float | None = None,
        alpha: float | None = None,
    ) -> Simulation:
        """Integrate and return a simulation result."""
        ...

    def simulate_protocol_time_course(
        self,
        ts: list[jax.Array],
        y0: jax.Array,
        protocol: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None = None,
        t0: float = 0.0,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
        dt0: jax.Array | float | None = None,
        alpha: float | None = None,
    ) -> Simulation:
        """Integrate through a protocol and return a simulation result."""
        ...

    def simulate_to_steady_state(
        self,
        y0: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None = None,
        t1: float = 1e6,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        steady_state_rtol: float | None = None,
        steady_state_atol: float | None = None,
        method: Method = diffrax.Tsit5,
        dt0: jax.Array | float | None = None,
        alpha: float | None = None,
    ) -> Simulation:
        """Integrate to steady state and return a 1-row simulation result."""
        ...


class FluxModel(Protocol):
    pars: PyTree | None
    readout_scale: PyTree
    readout_fn: ReadoutFn


def _no_readouts(
    time: PyTree,  # noqa: ARG001
    y: PyTree,  # noqa: ARG001
    args: PyTree,  # noqa: ARG001
) -> jax.Array:
    return jnp.array([])


def _resolve_readouts(
    readout_fn: ReadoutFn | None, readout_names: tuple[str, ...] | None
) -> tuple[ReadoutFn, tuple[str, ...]]:
    if readout_fn is None:
        return _no_readouts, ()
    if readout_names is None:
        msg = "readout_names is required when readout_fn is given"
        raise ValueError(msg)
    return readout_fn, readout_names


def _readout_names(
    model: KineticModelBuilder | OdeModelBuilder,
    readouts_to_calculate: list[str] | None,
) -> tuple[str, ...]:
    """Names of the generated readout function's outputs, in its return order."""
    return tuple(_get_extended_returns(model, readouts_to_calculate))


def _readouts_with_pars(
    model: FluxModel, t: PyTree, y: PyTree, args: PyTree
) -> jax.Array:
    if model.pars is not None:
        args = jnp.concat((args, model.pars))
    return model.readout_scale * model.readout_fn(t, y, args)


@functools.cache
def _steady_state_event(rtol: float | None, atol: float | None) -> diffrax.Event:
    """One shared :class:`diffrax.Event` per tolerance pair.

    ``diffrax.steady_state_event`` returns a fresh closure on every call, and
    the event is a static argument of ``diffeqsolve`` -- building it per call
    made every steady-state solve recompile the whole solver.
    """
    return diffrax.Event(diffrax.steady_state_event(rtol=rtol, atol=atol))


def boundaries_to_ts(boundaries: jax.Array) -> list[jax.Array]:
    """Convert a flat per-step boundary-time array to ``integrate_protocol``'s ``ts`` format.

    ``integrate_protocol``/``integrate_protocol_from_steady_state`` take
    ``ts`` as one array of save points per protocol step; the common case
    is a single save point per step -- that step's own transition time.
    Hand-writing that as ``[boundaries[i : i + 1] for i in ...]`` is easy to
    get subtly wrong (``boundaries[i]`` is a scalar, not the required
    one-dimensional per-step array), so this is the convenience constructor
    for it.

    Parameters
    ----------
    boundaries : jax.Array
        Ascending, one-dimensional array of per-step transition times,
        shape ``(n_steps,)``. Does **not** include the protocol's start
        time ``t0``.

    Returns
    -------
    list[jax.Array]
        One length-1 array per step, each holding that step's own
        transition time -- directly usable as ``integrate_protocol``'s
        ``ts``.
    """
    return [boundaries[i : i + 1] for i in range(boundaries.shape[0])]


def _protocol_args_per_row(
    ts: list[jax.Array], protocol: jax.Array, args: jax.Array | None
) -> jax.Array:
    """Arguments active at every row of a ``simulate_protocol_time_course`` result.

    Uses the same combination rule as :meth:`Base._integrate_protocol_raw`
    (constant ``args`` first, then the step's ``protocol`` row); the
    prepended ``t0`` row gets the first step's arguments.
    """
    combined = [
        step_args if args is None else jnp.concat((args, step_args))
        for step_args in protocol
    ]
    return jnp.concatenate(
        [combined[0][None, :]]
        + [
            jnp.broadcast_to(a, (step_ts.shape[0], a.shape[0]))
            for a, step_ts in zip(combined, ts, strict=True)
        ],
        axis=0,
    )


###############################################################################
# ABC
###############################################################################


class Base[SimT: Simulation](eqx.Module, ABC):
    """Abstract base class for equinox ODE models.

    Implements every ``integrate*`` method once, in the model's ODE state:
    ``y0`` is mapped into it by :meth:`_encode` and saved trajectories are
    mapped back to observations by :meth:`_decode`. Both are the identity
    here; models with a latent space override just those two.

    Solver failures (e.g. ``max_steps`` reached) raise
    ``equinox.EquinoxRuntimeError``, which :func:`mxlpy.jax.train.train`
    catches to retry; ``simulate_*(throw=False)`` reports them via
    ``success`` instead. A steady state not reached by ``t1`` never raises
    and returns a zeroed-out state.
    """

    variable_names: eqx.AbstractVar[tuple[str, ...]]
    readout_names: eqx.AbstractVar[tuple[str, ...]]

    @abstractmethod
    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Evaluate the ODE right-hand side at time ``t`` and ODE state ``y``."""
        ...

    @abstractmethod
    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Readouts at time ``t`` for the observations ``y``, in ``readout_names`` order."""
        ...

    def _encode(self, y0: jax.Array) -> jax.Array:
        """Map an observation into the ODE state; the identity by default."""
        return y0

    def _decode(
        self,
        zs: jax.Array,
        *,
        alpha: float | None,  # noqa: ARG002
    ) -> jax.Array:
        """Map ODE-state trajectories ``(T, n_state)`` to observations; the identity by default."""
        return zs

    def _decode_point(
        self,
        z: jax.Array,
        *,
        alpha: float | None,  # noqa: ARG002
    ) -> jax.Array:
        """:meth:`_decode` for a single state ``(n_state,)``; the identity by default.

        Separate from :meth:`_decode` so non-latent models don't pay for the
        eager reshape/index ops (each compiles its own tiny kernel).
        """
        return z

    def _solve(  # noqa: PLR0917
        self,
        t0: jax.Array,
        t1: jax.Array,
        y0: jax.Array,
        args: jax.Array | None,
        saveat_ts: jax.Array,
        max_steps: int,
        rtol: float,
        atol: float,
        method: Method,
        dt0: jax.Array | float | None = None,
        *,
        throw: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        """Integrate the ODE state from ``t0`` to ``t1``, saving at ``saveat_ts``.

        ``args=None`` is translated to an empty array, since every model's
        ``__call__`` concatenates ``args`` with its own inputs. ``dt0=None``
        lets diffrax pick the initial step. ``throw=True`` raises on solver
        failure; with ``throw=False`` failed solves return zeros instead of
        propagating NaN values.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            Saved ODE states, shape ``(len(saveat_ts), n_state)``, and
            whether the solve succeeded (boolean scalar).
        """
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self),
            method(),
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=jnp.zeros((0,)) if args is None else args,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            saveat=diffrax.SaveAt(ts=saveat_ts),
            max_steps=max_steps,
            throw=throw,
        )
        ys = cast(jax.Array, sol.ys)
        ok = sol.result == diffrax.RESULTS.successful
        # jnp.where rather than lax.cond over lambdas: outside jit, every
        # cond call traces fresh closures and compiles (and caches) a new
        # executable, leaking memory linearly in the number of integrations.
        return jnp.where(
            ok, ys, jnp.nan_to_num(ys, nan=0.0, posinf=0.0, neginf=0.0)
        ), ok

    def _integrate_latent(
        self,
        ts: jax.Array,
        y0: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None,
        rtol: float,
        atol: float,
        method: Method,
        dt0: jax.Array | float | None,
        throw: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        """:meth:`integrate` without the final decode: ODE states ``(T, n_state)`` and success.

        Anything evaluated on the model's state (``__call__``, ``fluxes``)
        must use these: re-encoding a decoded observation would replace a
        latent model's hidden dimensions with their encoding of ``y``
        instead of the hidden state the ODE actually reached.
        """
        return self._solve(
            t0=ts[0],
            t1=ts[-1],
            y0=self._encode(y0),
            args=args,
            saveat_ts=ts,
            max_steps=max_steps,
            rtol=rtol,
            atol=atol,
            method=method,
            dt0=dt0,
            throw=throw,
        )

    def integrate(
        self,
        ts: jax.Array,
        y0: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
        dt0: jax.Array | float | None = None,
        alpha: float | None = None,
    ) -> jax.Array:
        """Integrate from ``ts[0]`` to ``ts[-1]`` and return saved observations.

        Parameters
        ----------
        ts : jax.Array
            Sorted time points at which to save the solution, shape ``(T,)``.
        y0 : jax.Array
            Initial observation, shape ``(n_obs,)``.
        max_steps : int
            Maximum number of adaptive solver steps before aborting.
        args : jax.Array or None
            External arguments, held constant across the trajectory.
        rtol, atol : float
            Tolerances for the PID step-size controller.
        method : Method
            Diffrax Runge-Kutta solver class.
        dt0 : jax.Array, float, or None
            Initial step size; ``None`` lets diffrax choose.
        alpha : float or None
            Forwarded to the latent mapper's ``decode``; only meaningful
            for an annealed mapper (e.g. :class:`MixedLatentMapper`,
            :class:`TrainableZ0LatentMapper`), ignored otherwise.

        Returns
        -------
        jax.Array
            Saved observations, shape ``(T, n_obs)``.
        """
        zs, _ = self._integrate_latent(
            ts,
            y0,
            max_steps=max_steps,
            args=args,
            rtol=rtol,
            atol=atol,
            method=method,
            dt0=dt0,
        )
        return self._decode(zs, alpha=alpha)

    @eqx.filter_jit
    def _integrate_protocol_raw(  # noqa: PLR0917
        self,
        z0: jax.Array,
        ts: list[jax.Array],
        protocol: jax.Array,
        t0: jax.Array | np.ndarray,
        max_steps: int,
        args: jax.Array | None,
        rtol: float,
        atol: float,
        method: Method,
        dt0: jax.Array | float | None,
        *,
        throw: bool,
    ) -> tuple[jax.Array, jax.Array]:
        """Run the per-step solve loop in the ODE state, starting at ``(t0, z0)``.

        Returns the saved states and whether every step's solve succeeded.

        Implemented via ``lax.scan`` rather than a Python loop, so this
        compiles once regardless of the number of protocol steps. ``ts``'s
        per-step save-point counts may differ (ragged); each step's array is
        padded up to the longest step's length by repeating its own final
        timestamp -- diffrax accepts repeated ``SaveAt`` entries -- so every
        step has a uniform shape for ``scan``'s ``xs``, and the padded rows
        are sliced back off again before concatenating.
        """
        step_lengths = [t.shape[0] for t in ts]
        max_len = max(step_lengths)
        ts_padded = jnp.stack(
            [jnp.concatenate([t, jnp.full((max_len - t.shape[0],), t[-1])]) for t in ts]
        )

        def _step(
            carry: tuple[jax.Array, jax.Array], xs: tuple[jax.Array, jax.Array]
        ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
            y, t_start = carry
            step_ts, step_args = xs
            combined_args = step_args if args is None else jnp.concat((args, step_args))
            ys, ok = self._solve(
                t0=t_start,
                t1=step_ts[-1],
                y0=y,
                args=combined_args,
                saveat_ts=step_ts,
                max_steps=max_steps,
                rtol=rtol,
                atol=atol,
                method=method,
                dt0=dt0,
                throw=throw,
            )
            return (ys[-1], step_ts[-1]), (ys, ok)

        _, (ys_padded, oks) = lax.scan(
            _step, (z0, jnp.asarray(t0)), (ts_padded, protocol)
        )
        ys = jnp.concatenate([ys_padded[i, :n] for i, n in enumerate(step_lengths)])
        return ys, jnp.all(oks)

    def _integrate_protocol_latent(
        self,
        ts: list[jax.Array],
        y0: jax.Array,
        protocol: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None,
        t0: float,
        rtol: float,
        atol: float,
        method: Method,
        dt0: jax.Array | float | None,
        throw: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        """:meth:`integrate_protocol` without the final decode; also returns success."""
        return self._integrate_protocol_raw(
            self._encode(y0),
            ts,
            protocol,
            # An array, not a Python float: filter_jit treats floats as
            # static, so every distinct t0 would compile its own program.
            # NumPy, not jnp: creating a jnp scalar compiles a convert kernel.
            np.asarray(t0, dtype=ts[0].dtype),
            max_steps,
            args,
            rtol,
            atol,
            method,
            dt0,
            throw=throw,
        )

    def integrate_protocol(
        self,
        ts: list[jax.Array],
        y0: jax.Array,
        protocol: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None = None,
        t0: float = 0.0,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
        dt0: jax.Array | float | None = None,
        alpha: float | None = None,
    ) -> jax.Array:
        """Integrate through a sequence of protocol steps and return saved observations.

        Mirrors :meth:`mxlpy.simulator.Simulator.simulate_protocol_time_course`:
        each protocol step supplies a piecewise-constant external argument
        that is active for the duration of that step, and the trajectory is
        continuous across step boundaries.

        Parameters
        ----------
        ts : list[jax.Array]
            Per-step, ascending, absolute time points at which to save the
            solution -- one array per protocol step.  The **last** entry of
            each array must be that step's transition time, since it both
            bounds the integration and seeds the next step's initial
            condition. Build the common one-save-point-per-step case from a
            flat boundary array via :func:`boundaries_to_ts`.
        y0 : jax.Array
            Initial observation at ``t0``.
        protocol : jax.Array
            Per-step external argument, shape ``(len(ts), n_args)``.
            ``protocol[i]`` is held constant while integrating ``ts[i]``.
        max_steps : int
            Maximum number of adaptive solver steps per step.
        args : jax.Array or None
            Additional arguments held constant across all steps and
            concatenated before ``protocol[i]``.
        t0 : float
            Time the first step starts at.
        rtol, atol : float
            Tolerances for the PID step-size controller.
        method : Method
            Diffrax Runge-Kutta solver class.
        dt0 : jax.Array, float, or None
            Initial step size of every step; ``None`` lets diffrax choose.
        alpha : float or None
            Forwarded to the latent decode; see :meth:`integrate`.

        Returns
        -------
        jax.Array
            Saved observations concatenated across all steps, shape
            ``(sum(len(t) for t in ts), n_obs)``.  Does **not** include
            ``y0`` itself.
        """
        zs, _ = self._integrate_protocol_latent(
            ts,
            y0,
            protocol,
            max_steps=max_steps,
            args=args,
            t0=t0,
            rtol=rtol,
            atol=atol,
            method=method,
            dt0=dt0,
        )
        return self._decode(zs, alpha=alpha)

    def _integrate_to_steady_state_raw(
        self,
        z0: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None,
        t1: float,
        rtol: float,
        atol: float,
        steady_state_rtol: float | None,
        steady_state_atol: float | None,
        method: Method,
        dt0: jax.Array | float | None,
        throw: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Run the steady-state event solve in the ODE state.

        Returns the convergence time, the final state and whether the
        steady-state event was reached.
        """
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self),
            method(),
            t0=0.0,
            t1=t1,
            dt0=dt0,
            y0=z0,
            args=jnp.zeros((0,)) if args is None else args,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            saveat=diffrax.SaveAt(t1=True),
            event=_steady_state_event(steady_state_rtol, steady_state_atol),
            max_steps=max_steps,
            throw=throw,
        )
        t = cast(jax.Array, sol.ts)[-1]
        y = cast(jax.Array, sol.ys)[-1]
        ok = sol.result == diffrax.RESULTS.event_occurred
        # jnp.where, not lax.cond over lambdas: see _solve.
        return (
            t,
            jnp.where(ok, y, jnp.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)),
            ok,
        )

    def integrate_to_steady_state(
        self,
        y0: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None = None,
        t1: float = 1e6,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        steady_state_rtol: float | None = None,
        steady_state_atol: float | None = None,
        method: Method = diffrax.Tsit5,
        dt0: jax.Array | float | None = None,
        alpha: float | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Integrate until a steady state is reached and return the final observation.

        Analogous to
        :meth:`mxlpy.simulator.Simulator.simulate_to_steady_state`, but uses
        diffrax's event mechanism, since JAX requires a statically-shaped
        return value. Oscillation detection is not supported: oscillating
        systems run until ``t1`` or ``max_steps`` and return whatever state
        they end on.

        Parameters
        ----------
        y0 : jax.Array
            Initial observation at ``t=0``.
        max_steps : int
            Maximum number of adaptive solver steps before aborting.
        args : jax.Array or None
            Additional arguments forwarded to the RHS.
        t1 : float
            Upper bound on integration time; reaching it without triggering
            the steady-state event counts as a failed solve.
        rtol, atol : float
            Tolerances for the PID step-size controller.
        steady_state_rtol, steady_state_atol : float or None
            Tolerances of the steady-state event; default to ``rtol``/``atol``.
        method : Method
            Diffrax Runge-Kutta solver class.
        dt0 : jax.Array, float, or None
            Initial step size; ``None`` lets diffrax choose.
        alpha : float or None
            Forwarded to the latent decode; see :meth:`integrate`.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            Convergence time (scalar) and steady-state observation, shape
            ``(n_obs,)``.  If no steady state is reached by ``t1``, the real
            time paired with a zeroed-out state.
        """
        t, z, _ = self._integrate_to_steady_state_raw(
            self._encode(y0),
            max_steps=max_steps,
            args=args,
            t1=t1,
            rtol=rtol,
            atol=atol,
            steady_state_rtol=steady_state_rtol,
            steady_state_atol=steady_state_atol,
            method=method,
            dt0=dt0,
        )
        return t, self._decode_point(z, alpha=alpha)

    def _integrate_protocol_from_steady_state_latent(
        self,
        ts: list[jax.Array],
        y0: jax.Array,
        protocol: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None,
        preeq_args: jax.Array | None,
        t0: float,
        t1: float,
        rtol: float,
        atol: float,
        steady_state_rtol: float | None,
        steady_state_atol: float | None,
        method: Method,
        dt0: jax.Array | float | None,
        freeze_preeq_gradient: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """:meth:`integrate_protocol_from_steady_state` without the final decode.

        Also returns whether the steady state and the protocol succeeded --
        separately, as combining them is an eager op callers may not need.

        The steady state is handed to the protocol in the ODE state as-is --
        decoding and re-encoding it in between would reset a latent model's
        hidden dimensions, so the protocol would no longer start at the
        steady state.
        """
        if freeze_preeq_gradient:
            params, static = eqx.partition(self, eqx.is_inexact_array)
            preeq_model = eqx.combine(jax.tree.map(lax.stop_gradient, params), static)
        else:
            preeq_model = self
        _, z_ss, ok_ss = preeq_model._integrate_to_steady_state_raw(
            self._encode(y0),
            max_steps=max_steps,
            args=protocol[0] if preeq_args is None else preeq_args,
            t1=t1,
            rtol=rtol,
            atol=atol,
            steady_state_rtol=steady_state_rtol,
            steady_state_atol=steady_state_atol,
            method=method,
            dt0=dt0,
        )
        zs, ok = self._integrate_protocol_raw(
            z_ss,
            ts,
            protocol,
            np.asarray(t0, dtype=ts[0].dtype),
            max_steps,
            args,
            rtol,
            atol,
            method,
            dt0,
            throw=True,
        )
        return zs, ok_ss, ok

    def integrate_protocol_from_steady_state(
        self,
        ts: list[jax.Array],
        y0: jax.Array,
        protocol: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None = None,
        preeq_args: jax.Array | None = None,
        t0: float = 0.0,
        t1: float = 1e6,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        steady_state_rtol: float | None = None,
        steady_state_atol: float | None = None,
        method: Method = diffrax.Tsit5,
        dt0: jax.Array | float | None = None,
        alpha: float | None = None,
        freeze_preeq_gradient: bool = False,
    ) -> jax.Array:
        """Pre-equilibrate ``y0`` to steady state, then integrate a protocol from there.

        Parameters
        ----------
        ts, protocol, args, t0
            See :meth:`integrate_protocol`.
        y0 : jax.Array
            Initial observation to pre-equilibrate from.
        max_steps : int
            Maximum number of adaptive solver steps, both for the
            pre-equilibration and for each protocol step.
        preeq_args : jax.Array or None
            Arguments held constant during pre-equilibration. Defaults to
            ``protocol[0]``, the condition the protocol starts from.
        t1 : float
            Upper bound on pre-equilibration time; see
            :meth:`integrate_to_steady_state`.
        rtol, atol, steady_state_rtol, steady_state_atol, method, dt0
            Shared by the pre-equilibration and the protocol steps; see
            :meth:`integrate_to_steady_state`.
        alpha : float or None
            Forwarded to the latent decode; see :meth:`integrate`.
        freeze_preeq_gradient : bool
            Detach every inexact-array leaf of the model before the
            pre-equilibration solve, so it is skipped during
            differentiation. Differentiating through the stiff
            steady-state solve can hit singular linear solves for models
            with extreme eigenvalue spread, and a stop-gradient on the
            *output* alone does not help, because the failure happens
            inside the solve. The protocol stays fully differentiable.

        Returns
        -------
        jax.Array
            Saved observations across the protocol, shape
            ``(sum(len(t) for t in ts), n_obs)``.  Does **not** include the
            steady state itself.
        """
        zs, _, _ = self._integrate_protocol_from_steady_state_latent(
            ts,
            y0,
            protocol,
            max_steps=max_steps,
            args=args,
            preeq_args=preeq_args,
            t0=t0,
            t1=t1,
            rtol=rtol,
            atol=atol,
            steady_state_rtol=steady_state_rtol,
            steady_state_atol=steady_state_atol,
            method=method,
            dt0=dt0,
            freeze_preeq_gradient=freeze_preeq_gradient,
        )
        return self._decode(zs, alpha=alpha)

    # --- named simulation results -------------------------------------------

    def _result(
        self,
        ts: jax.Array,
        zs: jax.Array,
        ys: jax.Array,
        args: jax.Array,
        success: jax.Array,
    ) -> SimT:
        """Build the ``simulate_*`` result from its saved trajectory.

        ``zs`` is the ODE-state trajectory and ``ys`` the observed one;
        ``args`` holds the arguments active at every row. Readouts are
        evaluated on the observations. Only called inside the jitted
        assemblers below, so its vmaps and concatenations compile once per
        simulate path instead of once per eager operation.
        """
        return cast(
            SimT,
            Simulation(
                ts=ts,
                ys=ys,
                args=args,
                readouts=jax.vmap(self.readouts)(ts, ys, args),
                success=success,
                variable_names=self.variable_names,
                readout_names=self.readout_names,
                _latent_ys=zs,
            ),
        )

    @eqx.filter_jit
    def _time_course_result(
        self,
        ts: jax.Array,
        zs: jax.Array,
        args: jax.Array | None,
        success: jax.Array,
        alpha: float | None,
    ) -> SimT:
        row_args = jnp.zeros((0,)) if args is None else args
        return self._result(
            ts,
            zs,
            self._decode(zs, alpha=alpha),
            jnp.broadcast_to(row_args, (ts.shape[0], row_args.shape[0])),
            success,
        )

    @eqx.filter_jit
    def _protocol_result(  # noqa: PLR0917
        self,
        ts: list[jax.Array],
        t0: jax.Array | np.ndarray,
        y0: jax.Array,
        zs: jax.Array,
        protocol: jax.Array,
        args: jax.Array | None,
        success: jax.Array,
        alpha: float | None,
    ) -> SimT:
        return self._result(
            jnp.concatenate((jnp.asarray(t0)[None], *ts)),
            jnp.concatenate((self._encode(y0)[None, :], zs)),
            jnp.concatenate((y0[None, :], self._decode(zs, alpha=alpha))),
            _protocol_args_per_row(ts, protocol, args),
            success,
        )

    @eqx.filter_jit
    def _steady_state_result(
        self,
        t: jax.Array,
        z: jax.Array,
        args: jax.Array | None,
        success: jax.Array,
        alpha: float | None,
    ) -> SimT:
        row_args = jnp.zeros((0,)) if args is None else args
        return self._result(
            t[None],
            z[None, :],
            self._decode(z[None, :], alpha=alpha),
            row_args[None, :],
            success,
        )

    def simulate_time_course(
        self,
        ts: jax.Array,
        y0: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
        dt0: jax.Array | float | None = None,
        alpha: float | None = None,
        throw: bool = True,
    ) -> SimT:
        """Integrate and return a simulation result; see :meth:`integrate`.

        ``throw=False`` reports solver failures via the result's ``success``
        instead of raising.
        """
        zs, ok = self._integrate_latent(
            ts,
            y0,
            max_steps=max_steps,
            args=args,
            rtol=rtol,
            atol=atol,
            method=method,
            dt0=dt0,
            throw=throw,
        )
        return self._time_course_result(ts, zs, args, ok, alpha)

    def simulate_protocol_time_course(
        self,
        ts: list[jax.Array],
        y0: jax.Array,
        protocol: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None = None,
        t0: float = 0.0,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
        dt0: jax.Array | float | None = None,
        alpha: float | None = None,
        throw: bool = True,
    ) -> SimT:
        """Integrate through a protocol and return a simulation result.

        See :meth:`integrate_protocol`. Unlike it, the result includes the
        initial ``(t0, y0)`` row, consistent with
        :meth:`simulate_time_course`.
        """
        zs, ok = self._integrate_protocol_latent(
            ts,
            y0,
            protocol,
            max_steps=max_steps,
            args=args,
            t0=t0,
            rtol=rtol,
            atol=atol,
            method=method,
            dt0=dt0,
            throw=throw,
        )
        return self._protocol_result(
            ts, np.asarray(t0, dtype=ts[0].dtype), y0, zs, protocol, args, ok, alpha
        )

    def simulate_to_steady_state(
        self,
        y0: jax.Array,
        *,
        max_steps: int,
        args: jax.Array | None = None,
        t1: float = 1e6,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        steady_state_rtol: float | None = None,
        steady_state_atol: float | None = None,
        method: Method = diffrax.Tsit5,
        dt0: jax.Array | float | None = None,
        alpha: float | None = None,
        throw: bool = True,
    ) -> SimT:
        """Integrate to steady state and return a 1-row simulation result.

        See :meth:`integrate_to_steady_state`; the row is indexed by the
        actual convergence time.
        """
        t, z, ok = self._integrate_to_steady_state_raw(
            self._encode(y0),
            max_steps=max_steps,
            args=args,
            t1=t1,
            rtol=rtol,
            atol=atol,
            steady_state_rtol=steady_state_rtol,
            steady_state_atol=steady_state_atol,
            method=method,
            dt0=dt0,
            throw=throw,
        )
        return self._steady_state_result(t, z, args, ok, alpha)


class FluxBase(Base[FluxSimulation]):
    """Base class for models whose dynamics are ``nv(fluxes)``.

    Adds the per-point :meth:`fluxes`; ``simulate_*`` return a
    :class:`~mxlpy.jax.simulation.FluxSimulation`, whose fluxes are
    evaluated on the ODE-state trajectory.
    """

    flux_names: eqx.AbstractVar[tuple[str, ...]]

    @abstractmethod
    def fluxes(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Fluxes driving the ODE at time ``t`` and ODE state ``y``, in ``flux_names`` order."""
        ...

    def _flux_parts(
        self,
        ts: jax.Array,  # noqa: ARG002
        zs: jax.Array,  # noqa: ARG002
        args: jax.Array,  # noqa: ARG002
    ) -> dict[str, jax.Array]:
        """Extra per-row flux arrays for the result (see :class:`FluxUde`)."""
        return {}

    def _result(
        self,
        ts: jax.Array,
        zs: jax.Array,
        ys: jax.Array,
        args: jax.Array,
        success: jax.Array,
    ) -> FluxSimulation:
        return FluxSimulation(
            ts=ts,
            ys=ys,
            args=args,
            readouts=jax.vmap(self.readouts)(ts, ys, args),
            success=success,
            variable_names=self.variable_names,
            readout_names=self.readout_names,
            _latent_ys=zs,
            fluxes=jax.vmap(self.fluxes)(ts, zs, args),
            flux_names=self.flux_names,
            **self._flux_parts(ts, zs, args),
        )
