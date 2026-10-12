"""JAX model classes for mechanistic and neural ODE components.

Conventions shared by every model here:

* Every model integrates in its own *ODE state*. For most models that's
  simply the observed variables; :class:`Anode`/:class:`FluxAnode` integrate
  in a latent space of ``n_obs + n_hidden`` dimensions instead, mapped to and
  from observations by a :class:`LatentMapper` (see :meth:`Base._encode`/
  :meth:`Base._decode`). ``integrate*`` always takes and returns
  observations; ``alpha`` is forwarded to the latent decode and ignored by
  models without a latent space.
* ``__call__`` (the RHS) and ``fluxes`` take the ODE state; ``readouts``
  take observations.
* ``variable_names`` (and ``flux_names`` for flux models) define the model's
  dimensions. ``readout_names`` label :meth:`Base.readouts`'s output.
* Solver options after the leading data arguments are keyword-only.
"""

import functools
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal, Protocol, Self, cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import PRNGKeyArray, PyTree

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy.jax.simulation import FluxSimulation, Simulation
from mxlpy.meta._via_sym_repr import _get_extended_returns, generate_model_code_jax

__all__ = [
    "Anode",
    "Base",
    "CombineOp",
    "FluxAnode",
    "FluxBase",
    "FluxNode",
    "FluxOde",
    "FluxUde",
    "FullLatentMapper",
    "HardLatentMapper",
    "JaxModel",
    "LatentMapper",
    "LatentMapperFn",
    "Method",
    "MixedLatentMapper",
    "Node",
    "Nv",
    "Ode",
    "ReadoutFn",
    "Rhs",
    "SoftLatentMapper",
    "TrainableZ0LatentMapper",
    "Ude",
    "boundaries_to_ts",
]

###############################################################################
# Types
###############################################################################

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

    def _solve(
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
    def _integrate_protocol_raw(
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
    def _protocol_result(
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


###############################################################################
# Latent mapper
###############################################################################


class LatentMapper(Protocol):
    """Protocol for encode/decode mappings between observation and latent space."""

    def __init__(self, n_obs: int, n_hidden: int, key: PRNGKeyArray) -> None: ...

    def encode(self, y0: jax.Array) -> jax.Array:
        """Encode an observation vector into the latent space."""
        ...

    def decode(self, zs: jax.Array, *, alpha: float | None = None) -> jax.Array:
        """Decode latent trajectories back to observation space.

        ``alpha`` is ignored by every implementation except
        :class:`MixedLatentMapper`/:class:`TrainableZ0LatentMapper`, whose
        decode is an annealed hard/learned blend; see their docstrings. Kept
        on the shared Protocol (rather than only on those two classes) so
        callers that thread an annealing schedule through
        :meth:`Anode.integrate`/:meth:`FluxAnode.integrate` don't need to
        know which concrete mapper is in use.
        """
        ...


type LatentMapperFn = Callable[[int, int, PRNGKeyArray], LatentMapper]


class SoftLatentMapper(eqx.Module):
    """Maps observed states to and from a soft latent representation.

    The encoder appends learned hidden features to the raw observation;
    the decoder maps the full latent vector back to observation space.

    z0 <- concat((y0, linear(y0)))
    ys <- linear(zs)

    Parameters
    ----------
    n_obs : int
        Number of observed state dimensions.
    n_hidden : int
        Number of additional learned latent dimensions.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    """

    n_obs: int
    n_hidden: int
    n_latent: int
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(
        self,
        n_obs: int,
        n_hidden: int,
        key: PRNGKeyArray,
    ) -> None:
        self.n_obs = n_obs
        self.n_hidden = n_hidden
        self.n_latent = n_obs + n_hidden

        self.encoder = eqx.nn.Linear(
            in_features=self.n_obs,
            out_features=self.n_hidden,
            key=key,
        )
        self.decoder = eqx.nn.Linear(
            in_features=self.n_latent,
            out_features=self.n_obs,
            key=key,
        )

    def encode(self, y0: jax.Array) -> jax.Array:
        """Encode observation into latent space by appending learned features.

        Parameters
        ----------
        y0 : jax.Array
            Observation vector, shape ``(n_obs,)``.

        Returns
        -------
        jax.Array
            Latent vector, shape ``(n_latent,)``.
        """
        return jnp.concat((y0, self.encoder(y0)))

    def decode(self, zs: jax.Array, *, alpha: float | None = None) -> jax.Array:  # noqa: ARG002
        """Decode latent trajectories to observation space.

        Parameters
        ----------
        zs : jax.Array
            Latent trajectories, shape ``(T, n_latent)``.

        Returns
        -------
        jax.Array
            Decoded observations, shape ``(T, n_obs)``.
        """
        return jax.vmap(self.decoder)(zs)


class FullLatentMapper(eqx.Module):
    """Maps observed states to and from a fully learned latent representation.

    No part of the raw observation is preserved verbatim; both encoder
    and decoder are learned linear maps.

    z0 <- linear(y0)
    ys <- linear(zs)

    Parameters
    ----------
    n_obs : int
        Number of observed state dimensions.
    n_hidden : int
        Number of additional latent dimensions (n_latent = n_obs + n_hidden).
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    """

    n_obs: int
    n_hidden: int
    n_latent: int
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(
        self,
        n_obs: int,
        n_hidden: int,
        key: PRNGKeyArray,
    ) -> None:
        self.n_obs = n_obs
        self.n_hidden = n_hidden
        self.n_latent = n_obs + n_hidden

        self.encoder = eqx.nn.Linear(
            in_features=self.n_obs,
            out_features=self.n_latent,
            key=key,
        )
        self.decoder = eqx.nn.Linear(
            in_features=self.n_latent,
            out_features=self.n_obs,
            key=key,
        )

    def encode(self, y0: jax.Array) -> jax.Array:
        """Encode observation into latent space via a learned linear map.

        Parameters
        ----------
        y0 : jax.Array
            Observation vector, shape ``(n_obs,)``.

        Returns
        -------
        jax.Array
            Latent vector, shape ``(n_latent,)``.
        """
        return self.encoder(y0)

    def decode(self, zs: jax.Array, *, alpha: float | None = None) -> jax.Array:  # noqa: ARG002
        """Decode latent trajectories to observation space.

        Parameters
        ----------
        zs : jax.Array
            Latent trajectories, shape ``(T, n_latent)``.

        Returns
        -------
        jax.Array
            Decoded observations, shape ``(T, n_obs)``.
        """
        return jax.vmap(self.decoder)(zs)


class HardLatentMapper(eqx.Module):
    """Maps observed states to and from a latent space with a hard (slice) decoder.

    The encoder appends learned features; the decoder is a fixed slice
    that returns the first ``n_obs`` latent dimensions without learning.

    z0 <- concat((y0, linear(y0)))
    ys <- zs[:, :n_obs]

    Parameters
    ----------
    n_obs : int
        Number of observed state dimensions.
    n_hidden : int
        Number of additional learned latent dimensions.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    """

    n_obs: int
    n_hidden: int
    n_latent: int
    encoder: eqx.nn.Linear

    def __init__(self, n_obs: int, n_hidden: int, key: PRNGKeyArray) -> None:
        self.n_obs = n_obs
        self.n_hidden = n_hidden
        self.n_latent = n_obs + n_hidden

        self.encoder = eqx.nn.Linear(
            in_features=self.n_obs,
            out_features=self.n_hidden,
            key=key,
        )

    def encode(self, y0: jax.Array) -> jax.Array:
        """Encode observation into latent space by appending learned features.

        Parameters
        ----------
        y0 : jax.Array
            Observation vector, shape ``(n_obs,)``.

        Returns
        -------
        jax.Array
            Latent vector, shape ``(n_latent,)``.
        """
        return jnp.concat((y0, self.encoder(y0)))

    def decode(self, zs: jax.Array, *, alpha: float | None = None) -> jax.Array:  # noqa: ARG002
        """Decode latent trajectories by slicing the first ``n_obs`` dimensions.

        Parameters
        ----------
        zs : jax.Array
            Latent trajectories, shape ``(T, n_latent)``.

        Returns
        -------
        jax.Array
            Observed states, shape ``(T, n_obs)``.
        """
        return zs[:, : self.n_obs]


def _annealed_decode(
    zs: jax.Array,
    decoder: eqx.nn.Linear,
    n_obs: int,
    alpha: float | None,
) -> jax.Array:
    """Shared decode body for :class:`MixedLatentMapper`/:class:`TrainableZ0LatentMapper`.

    ``(1 - alpha) * hard + alpha * learned``, where ``hard`` is a plain slice
    of the first ``n_obs`` latent dimensions and ``learned`` is
    ``decoder`` applied to the full latent vector.

    Parameters
    ----------
    zs : jax.Array
        Latent trajectories, shape ``(T, n_latent)``.
    decoder : eqx.nn.Linear
        Learned latent-to-observation map.
    n_obs : int
        Number of observed state dimensions.
    alpha : float or None
        Mixing coefficient. ``0`` returns pure hard (slice) decoding; ``1``
        returns pure learned decoding. Required (not actually optional)
        despite the ``| None`` -- kept nullable only so the signature
        matches :class:`LatentMapper`'s shared Protocol, which defaults
        ``alpha`` to ``None`` for mappers that ignore it entirely; a caller
        using an annealed mapper without threading a real ``alpha`` through
        is a bug, not a case to silently paper over with a guessed default.

    Returns
    -------
    jax.Array
        Decoded observations, shape ``(T, n_obs)``.
    """
    if alpha is None:
        msg = (
            "alpha is required for an annealed hard/learned decode "
            "(MixedLatentMapper/TrainableZ0LatentMapper) -- thread it "
            "through the model's integrate*(alpha=...) "
            "parameter."
        )
        raise ValueError(msg)
    hard = zs[:, :n_obs]
    learned = jax.vmap(decoder)(zs)
    return (1 - alpha) * hard + alpha * learned


class MixedLatentMapper(eqx.Module):
    """Maps observed states to and from a latent space with an annealed decoder.

    The decoder interpolates between a hard projection and a learned map
    via ``alpha``; annealing ``alpha`` toward zero recovers the hard decoder.

    z0 <- concat((y0, linear(y0)))
    ys <- (1 - alpha) * zs[:, :n_obs] + alpha * linear(zs)

    Parameters
    ----------
    n_obs : int
        Number of observed state dimensions.
    n_hidden : int
        Number of additional learned latent dimensions.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    """

    n_obs: int
    n_hidden: int
    n_latent: int
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(self, n_obs: int, n_hidden: int, key: PRNGKeyArray) -> None:
        self.n_obs = n_obs
        self.n_hidden = n_hidden
        self.n_latent = n_obs + n_hidden

        self.encoder = eqx.nn.Linear(
            in_features=self.n_obs,
            out_features=self.n_hidden,
            key=key,
        )
        self.decoder = eqx.nn.Linear(
            in_features=self.n_latent,
            out_features=self.n_obs,
            key=key,
        )

    def encode(self, y0: jax.Array) -> jax.Array:
        """Encode observation into latent space by appending learned features.

        Parameters
        ----------
        y0 : jax.Array
            Observation vector, shape ``(n_obs,)``.

        Returns
        -------
        jax.Array
            Latent vector, shape ``(n_latent,)``.
        """
        return jnp.concat((y0, self.encoder(y0)))

    def decode(self, zs: jax.Array, *, alpha: float | None = None) -> jax.Array:
        """Decode latent trajectories using an annealed mixture of hard and learned maps.

        See :func:`_annealed_decode`.
        """
        return _annealed_decode(zs, self.decoder, self.n_obs, alpha)


class TrainableZ0LatentMapper(eqx.Module):
    """Maps observed states to and from a latent space with a free, trainable hidden init.

    Unlike every other :class:`LatentMapper`, the hidden dimensions of the
    initial latent state are not a function of ``y0`` at all: they're a
    single trainable vector shared across every trajectory the model is
    ever trained or evaluated on. The observed dimensions are still carried
    through ``y0`` verbatim, and the decoder is the same annealed
    hard/learned blend as :class:`MixedLatentMapper` (see
    :func:`_annealed_decode`) -- the two share that decode logic via a
    module-level helper rather than inheritance, since inheriting from
    MixedLatentMapper would carry along its (here unused) ``encoder`` field.

    z0 <- concat((y0, self.z0))
    ys <- (1 - alpha) * zs[:, :n_obs] + alpha * linear(zs)

    Parameters
    ----------
    n_obs : int
        Number of observed state dimensions.
    n_hidden : int
        Number of additional trainable latent dimensions.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    z0_minval : float
        Lower bound for ``z0``'s uniform initialisation. Defaults to 0.0.
    z0_maxval : float
        Upper bound for ``z0``'s uniform initialisation. Defaults to 1.0.
    """

    n_obs: int
    n_hidden: int
    n_latent: int
    z0: jax.Array
    decoder: eqx.nn.Linear

    def __init__(
        self,
        n_obs: int,
        n_hidden: int,
        key: PRNGKeyArray,
        z0_minval: float = 0.0,
        z0_maxval: float = 1.0,
    ) -> None:
        self.n_obs = n_obs
        self.n_hidden = n_hidden
        self.n_latent = n_obs + n_hidden

        z0_key, decoder_key = jax.random.split(key)
        self.z0 = jax.random.uniform(
            key=z0_key,
            shape=(n_hidden,),
            minval=z0_minval,
            maxval=z0_maxval,
        )
        self.decoder = eqx.nn.Linear(
            in_features=self.n_latent,
            out_features=self.n_obs,
            key=decoder_key,
        )

    def encode(self, y0: jax.Array) -> jax.Array:
        """Encode observation into latent space, appending the trainable ``z0``.

        Parameters
        ----------
        y0 : jax.Array
            Observation vector, shape ``(n_obs,)``.

        Returns
        -------
        jax.Array
            Latent vector, shape ``(n_latent,)``.
        """
        return jnp.concat((y0, self.z0))

    def decode(self, zs: jax.Array, *, alpha: float | None = None) -> jax.Array:
        """Decode latent trajectories using an annealed mixture of hard and learned maps.

        See :func:`_annealed_decode`.
        """
        return _annealed_decode(zs, self.decoder, self.n_obs, alpha)


###############################################################################
# Odes
###############################################################################


class Ode(Base[Simulation]):
    """ODE model with trainable parameter vector.

    Trainable parameters are appended to ``args`` at each evaluation, so
    ``rhs`` receives ``concat(args, pars)`` as its third argument -- the
    argument layout :func:`generate_model_code_jax` emits (free/runtime
    parameters first, trainable parameters last).

    Parameters
    ----------
    rhs : Rhs
        Right-hand-side function ``(t, y, args) -> dy/dt``.
    pars : jax.Array
        Trainable parameter vector.
    variable_names : tuple[str, ...]
        Names of the state variables, in ``rhs``'s order.
    readout_fn : ReadoutFn or None
        Readout function ``(t, y, concat(args, pars)) -> readouts``.
    readout_names : tuple[str, ...] or None
        Names of ``readout_fn``'s outputs; required with ``readout_fn``.
    readout_scale : jax.Array or float or None
        Multiplier for the readouts. A plain float (the default, 1.0) is
        not trained; pass a ``jax.Array`` to learn it.
    out_scale : jax.Array or None
        Multiplier for ``dy/dt``.  Defaults to 1.0.
    """

    rhs: Rhs
    pars: jax.Array
    out_scale: jax.Array
    readout_fn: ReadoutFn
    readout_scale: jax.Array | float
    variable_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        rhs: Rhs,
        pars: jax.Array,
        variable_names: tuple[str, ...],
        *,
        readout_fn: ReadoutFn | None = None,
        readout_names: tuple[str, ...] | None = None,
        readout_scale: jax.Array | float | None = None,
        out_scale: jax.Array | None = None,
    ) -> None:
        self.rhs = rhs
        self.pars = pars
        self.variable_names = variable_names
        self.readout_fn, self.readout_names = _resolve_readouts(
            readout_fn, readout_names
        )
        self.readout_scale = 1.0 if readout_scale is None else readout_scale
        self.out_scale = jnp.array([1.0]) if out_scale is None else out_scale

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``out_scale * rhs(t, y, concat(args, pars))``."""
        return self.out_scale * self.rhs(time, y, jnp.concat((args, self.pars)))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``readout_scale * readout_fn(t, y, concat(args, pars))``."""
        return self.readout_scale * self.readout_fn(t, y, jnp.concat((args, self.pars)))

    @classmethod
    def from_mxlpy(
        cls,
        mxlpy_model: KineticModelBuilder | OdeModelBuilder,
        parameters_to_fit: list[str] | None = None,
        free_parameters: list[str] | None = None,
        readouts_to_calculate: list[str] | None = None,
    ) -> Self:
        """Construct an :class:`Ode` from an mxlpy ``Model`` or ``OdeModelBuilder``.

        Generates JAX code for the RHS and the readouts via symbolic code
        generation and initialises ``pars`` from the model's current
        parameter values.

        Parameters
        ----------
        mxlpy_model : Model or OdeModelBuilder
            The mechanistic model to convert.
        parameters_to_fit : list[str] or None
            Parameters that become trainable (stored in ``pars``).
        free_parameters : list[str] or None
            Parameters passed as external ``args`` at call time rather than
            baked into the RHS.
        readouts_to_calculate : list[str] or None
            Subset of the model's derived parameters, derived variables,
            reactions, surrogate outputs and readouts that :meth:`readouts`
            returns. ``None`` returns all of them.

        Returns
        -------
        Self
            A new :class:`Ode`.
        """
        parameters_to_fit = [] if parameters_to_fit is None else parameters_to_fit
        free_parameters = [] if free_parameters is None else free_parameters
        codegen = generate_model_code_jax(
            mxlpy_model,
            parameters_to_fit=parameters_to_fit,
            free_parameters=free_parameters,
            derived_to_calculate=readouts_to_calculate,
        )

        # A single shared namespace, so names bound by ``codegen.imports``
        # (e.g. ``jnp``, ``functools``) are visible to the generated code
        # regardless of what this module happens to import at the top level.
        generated: dict[str, object] = {}

        exec(codegen.imports, generated)  # noqa: S102
        exec(codegen.model, generated)  # noqa: S102
        exec(codegen.derived, generated)  # noqa: S102

        model_pars = mxlpy_model.get_parameter_values()
        return cls(
            rhs=cast(Rhs, generated["model"]),
            pars=jnp.array([model_pars[k] for k in parameters_to_fit]),
            variable_names=tuple(mxlpy_model.get_initial_conditions()),
            readout_fn=cast(ReadoutFn, generated["derived"]),
            readout_names=_readout_names(mxlpy_model, readouts_to_calculate),
        )


class FluxOde(FluxBase):
    """ODE model defined via reaction fluxes and a stoichiometric map.

    ``dy/dt = nv(fluxes(t, y, args))``, with
    ``fluxes(t, y, args) = flux_scale * flux_fn(t, y, concat(args, pars))``.

    Parameters
    ----------
    flux_fn : Rhs
        Raw flux function ``(t, y, concat(args, pars)) -> flux_vector``.
    nv : Nv
        Stoichiometric map ``flux_vector -> dy/dt``.
    pars : jax.Array
        Trainable parameter vector, appended to ``args``.
    variable_names : tuple[str, ...]
        Names of the state variables, in the order ``flux_fn``/``nv`` expect
        and return them.
    flux_names : tuple[str, ...]
        Names of the fluxes, in ``flux_fn``'s output order.
    readout_fn, readout_names, readout_scale
        See :class:`Ode`.
    flux_scale : jax.Array or None
        Multiplier for the fluxes.  Defaults to 1.0.
    """

    flux_fn: Rhs
    nv: Nv
    pars: jax.Array
    flux_scale: jax.Array
    readout_fn: ReadoutFn
    readout_scale: jax.Array | float
    variable_names: tuple[str, ...] = eqx.field(static=True)
    flux_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        flux_fn: Rhs,
        nv: Nv,
        pars: jax.Array,
        variable_names: tuple[str, ...],
        flux_names: tuple[str, ...],
        *,
        readout_fn: ReadoutFn | None = None,
        readout_names: tuple[str, ...] | None = None,
        readout_scale: jax.Array | float | None = None,
        flux_scale: jax.Array | None = None,
    ) -> None:
        self.flux_fn = flux_fn
        self.nv = nv
        self.pars = pars
        self.variable_names = variable_names
        self.flux_names = flux_names
        self.flux_scale = jnp.array([1.0]) if flux_scale is None else flux_scale
        self.readout_fn, self.readout_names = _resolve_readouts(
            readout_fn, readout_names
        )
        self.readout_scale = 1.0 if readout_scale is None else readout_scale

    def fluxes(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``flux_scale * flux_fn(t, y, concat(args, pars))``."""
        return self.flux_scale * self.flux_fn(t, y, jnp.concat((args, self.pars)))

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``nv(fluxes(t, y, args))``."""
        return self.nv(self.fluxes(time, y, args))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``readout_scale * readout_fn(t, y, concat(args, pars))``."""
        return self.readout_scale * self.readout_fn(t, y, jnp.concat((args, self.pars)))

    @classmethod
    def from_mxlpy(
        cls,
        mxlpy_model: KineticModelBuilder,
        parameters_to_fit: list[str] | None = None,
        free_parameters: list[str] | None = None,
        readouts_to_calculate: list[str] | None = None,
    ) -> Self:
        """Construct a :class:`FluxOde` from an mxlpy ``Model``.

        See :meth:`Ode.from_mxlpy` for the parameters; additionally
        generates the flux function and the stoichiometric map. By default
        the readouts exclude the fluxes, which :meth:`fluxes` already
        reports (scaled by ``flux_scale``).
        """
        parameters_to_fit = [] if parameters_to_fit is None else parameters_to_fit
        free_parameters = [] if free_parameters is None else free_parameters
        flux_names = tuple(
            mxlpy_model.get_arg_names(
                include_time=False,
                include_variables=False,
                include_parameters=False,
                include_derived_parameters=False,
                include_derived_variables=False,
                include_reactions=True,
                include_surrogate_variables=False,
                include_surrogate_fluxes=True,
                include_readouts=False,
            )
        )
        if readouts_to_calculate is None:
            readouts_to_calculate = [
                name
                for name in _readout_names(mxlpy_model, None)
                if name not in flux_names
            ]
        codegen = generate_model_code_jax(
            mxlpy_model,
            parameters_to_fit=parameters_to_fit,
            free_parameters=free_parameters,
            derived_to_calculate=readouts_to_calculate,
        )

        # See Ode.from_mxlpy for the shared namespace.
        generated: dict[str, object] = {}

        exec(codegen.imports, generated)  # noqa: S102
        exec(codegen.fluxes, generated)  # noqa: S102
        exec(codegen.nv, generated)  # noqa: S102
        exec(codegen.derived, generated)  # noqa: S102

        model_pars = mxlpy_model.get_parameter_values()
        return cls(
            flux_fn=cast(Rhs, generated["fluxes"]),
            nv=cast(Nv, generated["nv"]),
            pars=jnp.array([model_pars[k] for k in parameters_to_fit]),
            variable_names=tuple(mxlpy_model.get_initial_conditions()),
            flux_names=flux_names,
            readout_fn=cast(ReadoutFn, generated["derived"]),
            readout_names=_readout_names(mxlpy_model, readouts_to_calculate),
        )


###############################################################################
# Nodes
###############################################################################


class Node(Base[Simulation]):
    """Neural ODE: an MLP maps state (and args) to its derivative.

    Output is scaled by ``out_scale`` (default 0.1) to keep initial
    dynamics small and aid early training stability.

    Parameters
    ----------
    variable_names : tuple[str, ...]
        Names of the state variables; their count is the MLP's state size.
    width : int
        Hidden layer width.
    depth : int
        Number of hidden layers.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    n_args : int
        Number of external arguments the MLP additionally consumes.
    out_scale : jax.Array or None
        Multiplier for the MLP output.  Defaults to 0.1.
    readout_fn, readout_names, readout_scale
        See :class:`Ode`; ``readout_fn`` gets ``(t, y, args)``.
    """

    out_scale: jax.Array
    nn: eqx.nn.MLP
    readout_fn: ReadoutFn
    readout_scale: jax.Array | float
    variable_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        variable_names: tuple[str, ...],
        width: int,
        depth: int,
        key: PRNGKeyArray,
        *,
        n_args: int = 0,
        out_scale: jax.Array | None = None,
        readout_fn: ReadoutFn | None = None,
        readout_names: tuple[str, ...] | None = None,
        readout_scale: jax.Array | float | None = None,
    ) -> None:
        n_obs = len(variable_names)
        self.variable_names = variable_names
        self.out_scale = jnp.array([0.1]) if out_scale is None else out_scale
        self.nn = eqx.nn.MLP(
            in_size=n_obs + n_args,
            out_size=n_obs,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )
        self.readout_fn, self.readout_names = _resolve_readouts(
            readout_fn, readout_names
        )
        self.readout_scale = 1.0 if readout_scale is None else readout_scale

    def __call__(
        self,
        time: PyTree,  # noqa: ARG002 ; for API stability
        y: PyTree,
        args: PyTree,
    ) -> jax.Array:
        """``out_scale * nn(concat(y, args))``."""
        return self.out_scale * self.nn(jnp.concat((y, args)))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``readout_scale * readout_fn(t, y, args)``."""
        return self.readout_scale * self.readout_fn(t, y, args)


def _readouts_with_pars(
    model: "FluxNode | FluxAnode", t: PyTree, y: PyTree, args: PyTree
) -> jax.Array:
    if model.pars is not None:
        args = jnp.concat((args, model.pars))
    return model.readout_scale * model.readout_fn(t, y, args)


class FluxNode(FluxBase):
    """Neural ODE with a flux-based formulation and stoichiometric map.

    An MLP predicts reaction fluxes; ``nv`` maps them to state derivatives.

    Parameters
    ----------
    variable_names : tuple[str, ...]
        Names of the state variables (MLP state input).
    flux_names : tuple[str, ...]
        Names of the fluxes (MLP output).
    width : int
        Hidden layer width.
    depth : int
        Number of hidden layers.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    nv : Nv
        Stoichiometric map ``flux_vector -> dy/dt``.
    n_args : int
        Number of external arguments the MLP additionally consumes.
    flux_scale : jax.Array or None
        Multiplier for the MLP's fluxes.  Defaults to 0.1.
    readout_fn, readout_names, readout_scale
        See :class:`Ode`.
    pars : jax.Array or None
        Parameters appended to ``args`` in :meth:`readouts` only (the MLP
        never sees them); :meth:`from_flux_ode` sets them to the source
        ``FluxOde``'s ``pars``, which its ``readout_fn`` expects.
    """

    flux_scale: jax.Array
    flux_nn: eqx.nn.MLP
    nv: Nv
    readout_fn: ReadoutFn
    readout_scale: jax.Array | float
    pars: jax.Array | None
    variable_names: tuple[str, ...] = eqx.field(static=True)
    flux_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        variable_names: tuple[str, ...],
        flux_names: tuple[str, ...],
        width: int,
        depth: int,
        key: PRNGKeyArray,
        nv: Nv,
        *,
        n_args: int = 0,
        flux_scale: jax.Array | None = None,
        readout_fn: ReadoutFn | None = None,
        readout_names: tuple[str, ...] | None = None,
        readout_scale: jax.Array | float | None = None,
        pars: jax.Array | None = None,
    ) -> None:
        self.variable_names = variable_names
        self.flux_names = flux_names
        self.pars = pars
        self.flux_scale = jnp.array([0.1]) if flux_scale is None else flux_scale
        self.flux_nn = eqx.nn.MLP(
            in_size=len(variable_names) + n_args,
            out_size=len(flux_names),
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )
        self.nv = nv
        self.readout_fn, self.readout_names = _resolve_readouts(
            readout_fn, readout_names
        )
        self.readout_scale = 1.0 if readout_scale is None else readout_scale

    @classmethod
    def from_flux_ode(
        cls,
        flux_ode: FluxOde,
        width: int,
        depth: int,
        key: PRNGKeyArray,
        *,
        n_args: int = 0,
        flux_scale: jax.Array | None = None,
        readout_scale: jax.Array | float | None = None,
    ) -> Self:
        """Construct a :class:`FluxNode` sharing ``flux_ode``'s names, ``nv`` and readouts."""
        return cls(
            variable_names=flux_ode.variable_names,
            flux_names=flux_ode.flux_names,
            width=width,
            depth=depth,
            key=key,
            nv=flux_ode.nv,
            n_args=n_args,
            flux_scale=flux_scale,
            readout_fn=flux_ode.readout_fn,
            readout_names=flux_ode.readout_names,
            readout_scale=readout_scale,
            pars=flux_ode.pars,
        )

    def fluxes(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:  # noqa: ARG002
        """``flux_scale * flux_nn(concat(y, args))``."""
        return self.flux_scale * self.flux_nn(jnp.concat((y, args)))

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``nv(fluxes(t, y, args))``."""
        return self.nv(self.fluxes(time, y, args))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``readout_scale * readout_fn(t, y, concat(args, pars))``."""
        return _readouts_with_pars(self, t, y, args)


###############################################################################
# Anodes
###############################################################################


class _LatentMixin:
    """Encode/decode through ``self.latent_mapper``; shared by the latent models."""

    latent_mapper: LatentMapper

    def _encode(self, y0: jax.Array) -> jax.Array:
        return self.latent_mapper.encode(y0)

    def _decode(self, zs: jax.Array, *, alpha: float | None) -> jax.Array:
        return self.latent_mapper.decode(zs, alpha=alpha)

    def _decode_point(self, z: jax.Array, *, alpha: float | None) -> jax.Array:
        return self.latent_mapper.decode(z[None, :], alpha=alpha)[0]


class Anode(_LatentMixin, Base[Simulation]):
    """Augmented Neural ODE: integrates in latent space, decodes to observations.

    The initial observation is encoded into an ``n_obs + n_hidden``
    dimensional latent space via ``latent_mapper``; the ODE runs there and
    the trajectory is decoded back to observation space.

    Parameters
    ----------
    variable_names : tuple[str, ...]
        Names of the observed variables.
    n_hidden : int
        Number of additional latent dimensions.
    width : int
        MLP hidden layer width.
    depth : int
        Number of MLP hidden layers.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    n_args : int
        Number of external arguments the MLP additionally consumes.
    out_scale : jax.Array or None
        Multiplier for the MLP output.  Defaults to 0.1.
    latent_mapper : LatentMapperFn or None
        Factory that constructs the latent mapper.  Defaults to
        :class:`SoftLatentMapper`.
    readout_fn, readout_names, readout_scale
        See :class:`Ode`; ``readout_fn`` gets decoded observations.
    """

    n_obs: int
    n_args: int
    out_scale: jax.Array
    nn: eqx.nn.MLP
    latent_mapper: LatentMapper
    readout_fn: ReadoutFn
    readout_scale: jax.Array | float
    variable_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        variable_names: tuple[str, ...],
        n_hidden: int,
        width: int,
        depth: int,
        key: PRNGKeyArray,
        *,
        n_args: int = 0,
        out_scale: jax.Array | None = None,
        latent_mapper: LatentMapperFn | None = None,
        readout_fn: ReadoutFn | None = None,
        readout_names: tuple[str, ...] | None = None,
        readout_scale: jax.Array | float | None = None,
    ) -> None:
        self.variable_names = variable_names
        self.n_obs = len(variable_names)
        self.n_args = n_args
        self.out_scale = jnp.array([0.1]) if out_scale is None else out_scale
        self.nn = eqx.nn.MLP(
            in_size=self.n_obs + n_hidden + n_args,
            out_size=self.n_obs + n_hidden,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )
        mapper = SoftLatentMapper if latent_mapper is None else latent_mapper
        self.latent_mapper = mapper(self.n_obs, n_hidden, key)
        self.readout_fn, self.readout_names = _resolve_readouts(
            readout_fn, readout_names
        )
        self.readout_scale = 1.0 if readout_scale is None else readout_scale

    def __call__(
        self,
        time: PyTree,  # noqa: ARG002 ; for API stability
        y: PyTree,
        args: PyTree,
    ) -> jax.Array:
        """``out_scale * nn(concat(z, args))`` on the latent state ``z``."""
        return self.out_scale * self.nn(jnp.concat((y, args)))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``readout_scale * readout_fn(t, y, args)`` on observations ``y``."""
        return self.readout_scale * self.readout_fn(t, y, args)


class FluxAnode(_LatentMixin, FluxBase):
    """Augmented flux Neural ODE with a Markov term for the hidden dimensions.

    Integrates in latent space using two MLPs: ``flux_nn`` predicts the
    reaction fluxes driving the observed dimensions through ``nv``,
    ``markov_nn`` the dynamics of the hidden dimensions.

    Parameters
    ----------
    variable_names : tuple[str, ...]
        Names of the observed variables.
    flux_names : tuple[str, ...]
        Names of the fluxes (``flux_nn``'s output).
    n_hidden : int
        Number of additional latent dimensions.
    flux_width, flux_depth : int
        Hidden layer width and count of the flux MLP.
    markov_width, markov_depth : int
        Hidden layer width and count of the Markov MLP.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    nv : Nv
        Stoichiometric map ``flux_vector -> dy/dt``.
    n_args : int
        Number of external arguments both MLPs additionally consume.
    flux_scale : jax.Array or None
        Multiplier for the fluxes.  Defaults to 0.1.
    markov_scale : jax.Array or None
        Multiplier for the Markov MLP output.  Defaults to 0.1.
    latent_mapper : LatentMapperFn or None
        Factory that constructs the latent mapper.  Defaults to
        :class:`HardLatentMapper`.
    readout_fn, readout_names, readout_scale, pars
        See :class:`FluxNode`; ``readout_fn`` gets decoded observations.
    """

    n_obs: int
    n_args: int
    nv: Nv
    flux_nn: eqx.nn.MLP
    markov_nn: eqx.nn.MLP
    flux_scale: jax.Array
    markov_scale: jax.Array
    latent_mapper: LatentMapper
    readout_fn: ReadoutFn
    readout_scale: jax.Array | float
    pars: jax.Array | None
    variable_names: tuple[str, ...] = eqx.field(static=True)
    flux_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        variable_names: tuple[str, ...],
        flux_names: tuple[str, ...],
        n_hidden: int,
        flux_width: int,
        flux_depth: int,
        markov_width: int,
        markov_depth: int,
        key: PRNGKeyArray,
        nv: Nv,
        *,
        n_args: int = 0,
        flux_scale: jax.Array | None = None,
        markov_scale: jax.Array | None = None,
        latent_mapper: LatentMapperFn | None = None,
        readout_fn: ReadoutFn | None = None,
        readout_names: tuple[str, ...] | None = None,
        readout_scale: jax.Array | float | None = None,
        pars: jax.Array | None = None,
    ) -> None:
        self.variable_names = variable_names
        self.flux_names = flux_names
        self.n_obs = len(variable_names)
        self.n_args = n_args
        self.nv = nv
        self.pars = pars
        self.flux_scale = jnp.array([0.1]) if flux_scale is None else flux_scale
        self.markov_scale = jnp.array([0.1]) if markov_scale is None else markov_scale
        n_latent = self.n_obs + n_hidden
        self.flux_nn = eqx.nn.MLP(
            in_size=n_latent + n_args,
            out_size=len(flux_names),
            width_size=flux_width,
            depth=flux_depth,
            activation=jax.nn.softplus,
            key=key,
        )
        self.markov_nn = eqx.nn.MLP(
            in_size=n_latent + n_args,
            out_size=n_hidden,
            width_size=markov_width,
            depth=markov_depth,
            activation=jax.nn.softplus,
            key=key,
        )
        mapper = HardLatentMapper if latent_mapper is None else latent_mapper
        self.latent_mapper = mapper(self.n_obs, n_hidden, key)
        self.readout_fn, self.readout_names = _resolve_readouts(
            readout_fn, readout_names
        )
        self.readout_scale = 1.0 if readout_scale is None else readout_scale

    @classmethod
    def from_flux_ode(
        cls,
        flux_ode: FluxOde,
        n_hidden: int,
        key: PRNGKeyArray,
        *,
        n_args: int = 0,
        flux_width: int | None = None,
        flux_depth: int = 2,
        markov_width: int | None = None,
        markov_depth: int = 2,
        flux_scale: jax.Array | None = None,
        markov_scale: jax.Array | None = None,
        readout_scale: jax.Array | float | None = None,
        latent_mapper: LatentMapperFn | None = None,
    ) -> Self:
        """Construct a :class:`FluxAnode` sharing ``flux_ode``'s names, ``nv`` and readouts.

        ``flux_width`` defaults to ``n_flux + n_args``, ``markov_width`` to
        ``n_obs + n_hidden + n_args``.
        """
        n_obs, n_flux = len(flux_ode.variable_names), len(flux_ode.flux_names)
        return cls(
            variable_names=flux_ode.variable_names,
            flux_names=flux_ode.flux_names,
            n_hidden=n_hidden,
            flux_width=n_flux + n_args if flux_width is None else flux_width,
            flux_depth=flux_depth,
            markov_width=(
                n_obs + n_hidden + n_args if markov_width is None else markov_width
            ),
            markov_depth=markov_depth,
            key=key,
            nv=flux_ode.nv,
            n_args=n_args,
            flux_scale=flux_scale,
            markov_scale=markov_scale,
            latent_mapper=latent_mapper,
            readout_fn=flux_ode.readout_fn,
            readout_names=flux_ode.readout_names,
            readout_scale=readout_scale,
            pars=flux_ode.pars,
        )

    def fluxes(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:  # noqa: ARG002
        """``flux_scale * flux_nn(concat(z, args))`` on the latent state ``z``."""
        return self.flux_scale * self.flux_nn(jnp.concat((y, args)))

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``concat(nv(fluxes), markov_scale * markov_nn(concat(z, args)))``."""
        return jnp.concat(
            (
                self.nv(self.fluxes(time, y, args)),
                self.markov_scale * self.markov_nn(jnp.concat((y, args))),
            )
        )

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``readout_scale * readout_fn(t, y, concat(args, pars))`` on observations ``y``."""
        return _readouts_with_pars(self, t, y, args)


###############################################################################
# UDEs
###############################################################################


def _rel(ode: jax.Array, nn: jax.Array) -> jax.Array:
    return ode * (1 + nn)


def _exp(ode: jax.Array, nn: jax.Array) -> jax.Array:
    return ode * jnp.exp(nn)


def _pow(ode: jax.Array, nn: jax.Array) -> jax.Array:
    return ode**nn


_OPS: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "rel": _rel,
    "exp": _exp,
    "pow": _pow,
}


def _resolve_op(
    op: CombineOp | Callable[[jax.Array, jax.Array], jax.Array],
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    if not isinstance(op, str):
        return op
    if op not in _OPS:
        msg = f"unknown op {op!r}; expected one of {sorted(_OPS)} or a callable"
        raise ValueError(msg)
    return _OPS[op]


class Ude(Base[Simulation]):
    """Universal Differential Equation: an ODE combined with a neural ODE.

    ``dy/dt = op(ode(t, y, args), nn(t, y, args))``, element-wise.

    Parameters
    ----------
    ode : Ode
        Mechanistic component.
    nn : Node
        Neural component; must have the same ``variable_names``.
    op : CombineOp or Callable
        ``"+"``, ``"-"``, ``"*"``, ``"/"``; ``"rel"``: ``ode * (1 + nn)``;
        ``"exp"``: ``ode * exp(nn)``; ``"pow"``: ``ode ** nn``; or any
        ``(ode, nn) -> combined`` function.
    """

    ode: Ode
    nn: Node
    op: Callable[[jax.Array, jax.Array], jax.Array]
    variable_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        ode: Ode,
        nn: Node,
        op: CombineOp | Callable[[jax.Array, jax.Array], jax.Array],
    ) -> None:
        if nn.variable_names != ode.variable_names:
            msg = (
                f"nn.variable_names {nn.variable_names} differ from "
                f"ode.variable_names {ode.variable_names}"
            )
            raise ValueError(msg)
        self.ode = ode
        self.nn = nn
        self.op = _resolve_op(op)
        self.variable_names = ode.variable_names
        self.readout_names = ode.readout_names

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``op(ode(t, y, args), nn(t, y, args))``."""
        return self.op(self.ode(time, y, args), self.nn(time, y, args))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """The mechanistic component's readouts."""
        return self.ode.readouts(t, y, args)


class FluxUde(FluxBase):
    """Flux-based Universal Differential Equation.

    Combines the mechanistic and the neural *fluxes*:
    ``fluxes = op(flux_ode.fluxes, flux_nn.fluxes)`` and
    ``dy/dt = flux_ode.nv(fluxes)``.

    Parameters
    ----------
    flux_ode : FluxOde
        Mechanistic component.
    flux_nn : FluxNode
        Neural component; must have the same ``variable_names`` and
        ``flux_names`` (e.g. built via :meth:`FluxNode.from_flux_ode`).
    op : CombineOp or Callable
        See :class:`Ude`.
    """

    flux_ode: FluxOde
    flux_nn: FluxNode
    op: Callable[[jax.Array, jax.Array], jax.Array]
    variable_names: tuple[str, ...] = eqx.field(static=True)
    flux_names: tuple[str, ...] = eqx.field(static=True)
    readout_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        flux_ode: FluxOde,
        flux_nn: FluxNode,
        op: CombineOp | Callable[[jax.Array, jax.Array], jax.Array],
    ) -> None:
        for attr in ("variable_names", "flux_names"):
            if getattr(flux_nn, attr) != getattr(flux_ode, attr):
                msg = (
                    f"flux_nn.{attr} {getattr(flux_nn, attr)} differ from "
                    f"flux_ode.{attr} {getattr(flux_ode, attr)}"
                )
                raise ValueError(msg)
        self.flux_ode = flux_ode
        self.flux_nn = flux_nn
        self.op = _resolve_op(op)
        self.variable_names = flux_ode.variable_names
        self.flux_names = flux_ode.flux_names
        self.readout_names = flux_ode.readout_names

    def fluxes(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``op(flux_ode.fluxes(t, y, args), flux_nn.fluxes(t, y, args))``."""
        return self.op(
            self.flux_ode.fluxes(t, y, args), self.flux_nn.fluxes(t, y, args)
        )

    def _flux_parts(
        self, ts: jax.Array, zs: jax.Array, args: jax.Array
    ) -> dict[str, jax.Array]:
        """Report the mechanistic and neural fluxes that ``op`` combines."""
        return {
            "ode_fluxes": jax.vmap(self.flux_ode.fluxes)(ts, zs, args),
            "nn_fluxes": jax.vmap(self.flux_nn.fluxes)(ts, zs, args),
        }

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """``flux_ode.nv(fluxes(t, y, args))``."""
        return self.flux_ode.nv(self.fluxes(time, y, args))

    def readouts(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """The mechanistic component's readouts."""
        return self.flux_ode.readouts(t, y, args)
