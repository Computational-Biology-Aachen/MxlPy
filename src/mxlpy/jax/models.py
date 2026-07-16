"""JAX model classes for mechanistic and neural ODE components."""

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal, Protocol, Self, cast

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import PRNGKeyArray, PyTree

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy.jax.simulation import FluxOdeSimulation
from mxlpy.meta._via_sym_repr import generate_model_code_jax

__all__ = [
    "Anode",
    "Base",
    "Encoder",
    "FluxAnode",
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
    "Rhs",
    "SoftLatentMapper",
    "Ude",
]

###############################################################################
# Types
###############################################################################

type Method = Callable[[], diffrax.AbstractRungeKutta]
type Rhs = Callable[[PyTree, PyTree, PyTree], jax.Array]
type Nv = Callable[[PyTree], PyTree]
type Encoder = Callable[[jax.Array], jax.Array]


class DerivedFn(Protocol):
    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Calculate derived quantitites."""
        ...


class JaxModel(Protocol):
    """Protocol for JAX-compatible ODE models.

    Any object implementing ``__call__`` and ``integrate`` can be used
    with the training utilities in ``train.py``.
    """

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Evaluate the ODE right-hand side at time ``t`` and state ``y``."""
        ...

    def integrate(
        self,
        ts: jax.Array,
        y0: jax.Array,
        max_steps: int,
        args: jax.Array | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
    ) -> jax.Array:
        """Integrate the model from ``ts[0]`` to ``ts[-1]`` and return saved states."""
        ...

    def integrate_protocol(
        self,
        ts: list[jax.Array],
        y0: jax.Array,
        protocol: jax.Array,
        max_steps: int,
        args: jax.Array | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
    ) -> jax.Array:
        """Integrate through a sequence of protocol steps and return saved states."""
        ...


def _default_derived(
    time: PyTree,  # noqa: ARG001
    y: PyTree,  # noqa: ARG001
    args: PyTree,  # noqa: ARG001
) -> jax.Array:
    return jnp.array([])


###############################################################################
# ABC
###############################################################################


class Base(eqx.Module, ABC):
    """Abstract base class for equinox ODE models.

    Provides a default ``integrate`` implementation via diffrax.
    Failed solves return zeros rather than propagating NaN values.
    """

    @abstractmethod
    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Evaluate the ODE right-hand side at time ``t`` and state ``y``."""
        ...

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
    ) -> jax.Array:
        """Integrate from ``t0`` to ``t1``, saving at ``saveat_ts``.

        Shared by :meth:`integrate` and :meth:`integrate_protocol`.  Failed
        solves return zeros instead of propagating NaN values.

        Parameters
        ----------
        t0 : jax.Array
            Start time.
        t1 : jax.Array
            End time.
        y0 : jax.Array
            Initial condition at ``t0``.
        args : jax.Array or None
            Additional arguments forwarded to the RHS.
        saveat_ts : jax.Array
            Time points at which to save the solution; must lie within
            ``[t0, t1]``.
        max_steps : int
            Maximum number of adaptive solver steps before aborting.
        rtol : float
            Relative tolerance for the PID step-size controller.
        atol : float
            Absolute tolerance for the PID step-size controller.
        method : Method
            Diffrax Runge-Kutta solver class.

        Returns
        -------
        jax.Array
            Saved states, shape ``(len(saveat_ts), n_state)``.
        """
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self),
            method(),
            t0=t0,
            t1=t1,
            dt0=None,
            y0=y0,
            args=args,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            saveat=diffrax.SaveAt(ts=saveat_ts),
            max_steps=max_steps,
        )
        ys = cast(jax.Array, sol.ys)
        return lax.cond(
            sol.result == diffrax.RESULTS.successful,
            lambda: ys,
            lambda: jnp.nan_to_num(
                ys,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
        )

    def integrate(
        self,
        ts: jax.Array,
        y0: jax.Array,
        max_steps: int,
        args: jax.Array | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
    ) -> jax.Array:
        """Integrate from ``ts[0]`` to ``ts[-1]`` and return saved states.

        Parameters
        ----------
        ts : jax.Array
            Sorted time points at which to save the solution, shape ``(T,)``.
        y0 : jax.Array
            Initial condition.
        max_steps : int
            Maximum number of adaptive solver steps before aborting.
        rtol : float
            Relative tolerance for the PID step-size controller.
        atol : float
            Absolute tolerance for the PID step-size controller.
        method : Method
            Diffrax Runge-Kutta solver class.

        Returns
        -------
        jax.Array
            Saved states, shape ``(T, n_state)``.  Failed solves return zeros.
        """
        return self._solve(
            t0=ts[0],
            t1=ts[-1],
            y0=y0,
            args=args,
            saveat_ts=ts,
            max_steps=max_steps,
            rtol=rtol,
            atol=atol,
            method=method,
        )

    def integrate_protocol(
        self,
        ts: list[jax.Array],
        y0: jax.Array,
        protocol: jax.Array,
        max_steps: int,
        args: jax.Array | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
    ) -> jax.Array:
        """Integrate through a sequence of protocol steps and return saved states.

        Mirrors :meth:`mxlpy.simulator.Simulator.simulate_protocol_time_course`:
        each protocol step supplies a piecewise-constant external argument
        (e.g. a free parameter such as light intensity) that is active for
        the duration of that step, and the trajectory is continuous across
        step boundaries (each step's initial condition is the previous
        step's final saved state).

        Note that this default implementation integrates in state space
        directly; subclasses that override :meth:`integrate` to run in a
        transformed space (:class:`Anode`, :class:`FluxAnode`) would need a
        matching override here to encode/decode around the per-step loop.

        Parameters
        ----------
        ts : list[jax.Array]
            Per-step, ascending, absolute time points at which to save the
            solution -- one array per protocol step.  The **last** entry of
            each array must be that step's transition time (the point at
            which ``protocol`` switches to its next row), since that value
            both bounds the integration and seeds the next step's initial
            condition. The first step is integrated from ``t=0``.
        y0 : jax.Array
            Initial condition at ``t=0``.
        protocol : jax.Array
            Per-step external argument, shape ``(len(ts), n_args)``.
            ``protocol[i]`` is held constant while integrating the
            ``ts[i]`` window.
        args : jax.Array or None
            Additional arguments held constant across all steps and
            concatenated after ``protocol[i]`` (e.g. trainable parameters
            that are not part of the protocol).
        max_steps : int
            Maximum number of adaptive solver steps per step, before aborting.
        rtol : float
            Relative tolerance for the PID step-size controller.
        atol : float
            Absolute tolerance for the PID step-size controller.
        method : Method
            Diffrax Runge-Kutta solver class.

        Returns
        -------
        jax.Array
            Saved states concatenated across all steps, shape
            ``(sum(len(t) for t in ts), n_state)``.  Does **not** include
            ``y0`` itself; prepend it if the initial condition is also an
            observation.
        """
        y = y0
        t_start = jnp.zeros_like(ts[0][0])
        results = []
        for step_ts, step_args in zip(ts, protocol, strict=True):
            combined_args = step_args if args is None else jnp.concat((args, step_args))
            ys = self._solve(
                t0=t_start,
                t1=step_ts[-1],
                y0=y,
                args=combined_args,
                saveat_ts=step_ts,
                max_steps=max_steps,
                rtol=rtol,
                atol=atol,
                method=method,
            )
            results.append(ys)
            y = ys[-1]
            t_start = step_ts[-1]
        return jnp.concatenate(results, axis=0)

    def integrate_to_steady_state(
        self,
        y0: jax.Array,
        max_steps: int,
        args: jax.Array | None = None,
        t1: float = 1e6,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        steady_state_rtol: float | None = None,
        steady_state_atol: float | None = None,
        method: Method = diffrax.Tsit5,
    ) -> tuple[jax.Array, jax.Array]:
        """Integrate until a steady state is reached and return the final state.

        Analogous to
        :meth:`mxlpy.simulator.Simulator.simulate_to_steady_state`, but uses
        diffrax's event mechanism rather than mxlpy's polling-based
        integrator loop, since JAX requires a statically-shaped return value
        (the final state, not a variable-length trajectory).  Oscillation
        detection is not supported: unlike the mxlpy integrators, diffrax has
        no way to inspect trajectory history from within the event
        condition, so oscillating systems will simply run until ``t1`` or
        ``max_steps`` and return whatever state they end on -- callers should
        design their own oscillation check on the returned state if needed.

        Parameters
        ----------
        y0 : jax.Array
            Initial condition at ``t=0``.
        args : jax.Array or None
            Additional arguments forwarded to the RHS.
        t1 : float
            Upper bound on integration time; reaching it without triggering
            the steady-state event counts as a failed solve.
        rtol : float
            Relative tolerance for the PID step-size controller.
        atol : float
            Absolute tolerance for the PID step-size controller.
        steady_state_rtol : float or None
            Relative tolerance for the steady-state event; defaults to ``rtol``.
        steady_state_atol : float or None
            Absolute tolerance for the steady-state event; defaults to ``atol``.
        max_steps : int
            Maximum number of adaptive solver steps before aborting.
        method : Method
            Diffrax Runge-Kutta solver class.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            Convergence time (scalar) and final (steady-state) state, shape
            ``(n_state,)``.  Failed solves (steady state not reached by
            ``t1``, or a genuine solver failure) return the real time paired
            with a zeroed-out state.
        """
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self),
            method(),
            t0=0.0,
            t1=t1,
            dt0=None,
            y0=y0,
            args=args,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            saveat=diffrax.SaveAt(t1=True),
            event=diffrax.Event(
                diffrax.steady_state_event(
                    rtol=steady_state_rtol,
                    atol=steady_state_atol,
                )
            ),
            max_steps=max_steps,
        )
        t = cast(jax.Array, sol.ts)[-1]
        y = cast(jax.Array, sol.ys)[-1]
        return t, lax.cond(
            sol.result == diffrax.RESULTS.event_occurred,
            lambda: y,
            lambda: jnp.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0),
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

    def decode(self, zs: jax.Array) -> jax.Array:
        """Decode latent trajectories back to observation space."""
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

    def decode(self, zs: jax.Array) -> jax.Array:
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

    def decode(self, zs: jax.Array) -> jax.Array:
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

    def decode(self, zs: jax.Array) -> jax.Array:
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

    def decode(self, zs: jax.Array, alpha: float) -> jax.Array:
        """Decode latent trajectories using an annealed mixture of hard and learned maps.

        Parameters
        ----------
        zs : jax.Array
            Latent trajectories, shape ``(T, n_latent)``.
        alpha : float
            Mixing coefficient. ``0`` returns pure hard (slice) decoding;
            ``1`` returns pure learned decoding.

        Returns
        -------
        jax.Array
            Decoded observations, shape ``(T, n_obs)``.
        """
        hard = zs[:, : self.n_obs]
        learned = jax.vmap(self.decoder)(zs)
        return (1 - alpha) * hard + alpha * learned


###############################################################################
# Odes
###############################################################################


class Ode(Base):
    """ODE model with trainable parameter vector.

    Trainable parameters are appended to ``args`` at each evaluation so that
    ``rhs`` receives ``concat(args, pars)`` as its third argument. This matches
    the argument layout produced by :func:`generate_model_code_jax`, which emits
    ``args = free_parameters + parameters_to_fit`` (free/runtime parameters
    first, trainable parameters last).

    Parameters
    ----------
    rhs : Rhs
        Right-hand-side function ``(t, y, args) -> dy/dt``.
    pars : jax.Array
        Trainable parameter vector.
    """

    rhs: Rhs
    out_scale: jax.Array
    pars: jax.Array  # trainable parameters
    derived_fn: DerivedFn

    def __init__(
        self,
        rhs: Rhs,
        pars: jax.Array,
        derived_fn: DerivedFn | None = None,
        out_scale: jax.Array | None = None,
    ) -> None:
        self.rhs = rhs
        self.pars = pars
        self.derived_fn = _default_derived if derived_fn is None else derived_fn
        self.out_scale = jnp.array([1.0]) if out_scale is None else out_scale

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Evaluate the ODE right-hand side.

        Parameters
        ----------
        t : PyTree
            Current time.
        y : PyTree
            Current state.
        args : PyTree
            Free/runtime arguments; trainable ``pars`` are appended after these.

        Returns
        -------
        jax.Array
            State derivative ``dy/dt``.
        """
        return self.out_scale * self.rhs(time, y, jnp.concat((args, self.pars)))

    def derived(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Get derived quantitites."""
        return self.derived_fn(t, y, jnp.concat((self.pars, args)))

    @classmethod
    def from_mxlpy(
        cls,
        mxlpy_model: KineticModelBuilder,
        parameters_to_fit: list[str] | None = None,
        free_parameters: list[str] | None = None,
    ) -> Self:
        """Construct an :class:`Ode` from an mxlpy :class:`~mxlpy.Model`.

        Generates JAX-compatible RHS code from the model via symbolic code
        generation, executes it, and initialises the trainable parameter
        vector from the model's current parameter values.

        Parameters
        ----------
        mxlpy_model : Model
            The mechanistic model to convert.
        parameters_to_fit : list[str] or None
            Names of parameters that become trainable (stored in ``pars``).
            If ``None``, no parameters are made trainable.
        free_parameters : list[str] or None
            Names of parameters that are passed as external ``args`` at
            call time rather than baked into the RHS.  If ``None``, no
            parameters are treated as free.

        Returns
        -------
        Self
            A new :class:`Ode` instance whose ``rhs`` is the generated JAX
            function and whose ``pars`` are initialised from the model's
            current parameter values for the keys in ``parameters_to_fit``.
        """
        parameters_to_fit = [] if parameters_to_fit is None else parameters_to_fit
        free_parameters = [] if free_parameters is None else free_parameters
        codegen = generate_model_code_jax(
            mxlpy_model,
            parameters_to_fit=parameters_to_fit,
            free_parameters=free_parameters,
        )

        # A single shared namespace, so names bound by ``codegen.imports``
        # (e.g. ``jnp``, ``functools``) are visible to ``codegen.model``
        # regardless of what this module happens to import at the top level.
        generated: dict[str, object] = {}

        exec(codegen.imports, generated)  # noqa: S102
        exec(codegen.model, generated)  # noqa: S102

        model_pars = mxlpy_model.get_parameter_values()
        pars = jnp.array([model_pars[k] for k in parameters_to_fit])
        return cls(rhs=generated["model"], pars=pars)


class FluxOde(Base):
    """ODE model defined via reaction fluxes and a stoichiometric map.

    ``dy/dt = nv(fluxes(t, y, args))``

    Parameters
    ----------
    fluxes : Rhs
        Flux function ``(t, y, args) -> flux_vector``.
    nv : Nv
        Stoichiometric map ``flux_vector -> dy/dt``.
    variable_names : tuple[str, ...] or None
        Names of the state variables, in the order ``fluxes``/``nv`` expect
        and return them.  Populated automatically by :meth:`from_mxlpy`;
        ``None`` for hand-built instances.  Required by ``simulate_*``.
    flux_names : tuple[str, ...] or None
        Names of the fluxes, in the order returned by ``fluxes``.  Populated
        automatically by :meth:`from_mxlpy`; ``None`` for hand-built
        instances.  Required by ``simulate_*``.
    """

    fluxes: Rhs
    nv: Nv
    pars: jax.Array  # trainable parameters
    variable_names: tuple[str, ...] | None = eqx.field(static=True, default=None)
    flux_names: tuple[str, ...] | None = eqx.field(static=True, default=None)
    derived_fn: DerivedFn

    def __init__(
        self,
        fluxes: Rhs,
        nv: Nv,
        pars: jax.Array,
        variable_names: tuple[str, ...] | None = None,
        flux_names: tuple[str, ...] | None = None,
        derived_fn: DerivedFn | None = None,
    ) -> None:
        self.fluxes = fluxes
        self.nv = nv
        self.pars = pars
        self.variable_names = variable_names
        self.flux_names = flux_names
        self.derived_fn = _default_derived if derived_fn is None else derived_fn

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Evaluate the ODE right-hand side via fluxes.

        Parameters
        ----------
        t : PyTree
            Current time.
        y : PyTree
            Current state.
        args : PyTree
            Free/runtime arguments; trainable ``pars`` are appended after these
            before being forwarded to ``fluxes``.

        Returns
        -------
        jax.Array
            State derivative ``dy/dt``.
        """
        return self.nv(self.fluxes(time, y, jnp.concat((args, self.pars))))

    def derived(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Get derived quantitites."""
        return self.derived_fn(t, y, jnp.concat((self.pars, args)))

    @classmethod
    def from_mxlpy(
        cls,
        mxlpy_model: KineticModelBuilder,
        parameters_to_fit: list[str] | None = None,
        free_parameters: list[str] | None = None,
    ) -> Self:
        """Construct an :class:`Ode` from an mxlpy :class:`~mxlpy.Model`.

        Generates JAX-compatible RHS code from the model via symbolic code
        generation, executes it, and initialises the trainable parameter
        vector from the model's current parameter values.

        Parameters
        ----------
        mxlpy_model : Model
            The mechanistic model to convert.
        parameters_to_fit : list[str] or None
            Names of parameters that become trainable (stored in ``pars``).
            If ``None``, no parameters are made trainable.
        free_parameters : list[str] or None
            Names of parameters that are passed as external ``args`` at
            call time rather than baked into the RHS.  If ``None``, no
            parameters are treated as free.

        Returns
        -------
        Self
            A new :class:`Ode` instance whose ``rhs`` is the generated JAX
            function and whose ``pars`` are initialised from the model's
            current parameter values for the keys in ``parameters_to_fit``.
        """
        parameters_to_fit = [] if parameters_to_fit is None else parameters_to_fit
        free_parameters = [] if free_parameters is None else free_parameters
        codegen = generate_model_code_jax(
            mxlpy_model,
            parameters_to_fit=parameters_to_fit,
            free_parameters=free_parameters,
        )

        # A single shared namespace, so names bound by ``codegen.imports``
        # (e.g. ``jnp``, ``functools``) are visible to ``codegen.fluxes``/
        # ``codegen.nv`` regardless of what this module happens to import at
        # the top level.
        generated: dict[str, object] = {}

        exec(codegen.imports, generated)  # noqa: S102
        exec(codegen.fluxes, generated)  # noqa: S102
        exec(codegen.nv, generated)  # noqa: S102

        model_pars = mxlpy_model.get_parameter_values()
        pars = jnp.array([model_pars[k] for k in parameters_to_fit])
        variable_names = tuple(mxlpy_model.get_initial_conditions())
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
        return cls(
            fluxes=generated["fluxes"],
            nv=generated["nv"],
            pars=pars,
            variable_names=variable_names,
            flux_names=flux_names,
        )

    def _require_names(self) -> None:
        if self.variable_names is None or self.flux_names is None:
            msg = (
                "FluxOde has no variable_names/flux_names set; construct it "
                "via FluxOde.from_mxlpy(...), or set them explicitly, before "
                "calling simulate_*."
            )
            raise ValueError(msg)

    def simulate_time_course(
        self,
        ts: jax.Array,
        y0: jax.Array,
        max_steps: int,
        args: jax.Array | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
    ) -> FluxOdeSimulation:
        """Integrate and return a named, DataFrame-based simulation result.

        Parameters
        ----------
        ts : jax.Array
            Sorted time points at which to save the solution, shape ``(T,)``.
        y0 : jax.Array
            Initial condition.
        args : jax.Array or None
            External arguments, held constant across the trajectory.
        max_steps : int
            Maximum number of adaptive solver steps before aborting.
        rtol : float
            Relative tolerance for the PID step-size controller.
        atol : float
            Absolute tolerance for the PID step-size controller.
        method : Method
            Diffrax Runge-Kutta solver class.

        Returns
        -------
        FluxOdeSimulation
            Named simulation result.
        """
        self._require_names()
        ys = self.integrate(
            ts, y0, args, max_steps=max_steps, rtol=rtol, atol=atol, method=method
        )
        row_args = jnp.zeros((0,)) if args is None else args
        args_per_row = jnp.broadcast_to(row_args, (ts.shape[0], row_args.shape[0]))
        return FluxOdeSimulation(
            model=self,
            ts=ts,
            ys=ys,
            args=args_per_row,
            variable_names=cast(tuple, self.variable_names),
            flux_names=cast(tuple, self.flux_names),
        )

    def simulate_protocol_time_course(
        self,
        ts: list[jax.Array],
        y0: jax.Array,
        protocol: jax.Array,
        max_steps: int,
        args: jax.Array | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
    ) -> FluxOdeSimulation:
        """Integrate through a protocol and return a named simulation result.

        Prepends the ``y0``/``t=0`` row, since :meth:`integrate_protocol`
        deliberately never includes it (its per-step windows start strictly
        after the previous step's end) -- without this, the result would be
        inconsistent with :meth:`simulate_time_course`, which does include
        its starting point.

        Parameters
        ----------
        ts : list[jax.Array]
            Per-step, ascending, absolute time points; see
            :meth:`integrate_protocol`.
        y0 : jax.Array
            Initial condition at ``t=0``.
        protocol : jax.Array
            Per-step external argument, shape ``(len(ts), n_args)``.
        args : jax.Array or None
            Additional arguments held constant across all steps; see
            :meth:`integrate_protocol`.
        max_steps : int
            Maximum number of adaptive solver steps per step, before aborting.
        rtol : float
            Relative tolerance for the PID step-size controller.
        atol : float
            Absolute tolerance for the PID step-size controller.
        method : Method
            Diffrax Runge-Kutta solver class.

        Returns
        -------
        FluxOdeSimulation
            Named simulation result, including the initial ``t=0`` row.
        """
        self._require_names()
        ys = self.integrate_protocol(
            ts,
            y0,
            protocol,
            args,
            max_steps=max_steps,
            rtol=rtol,
            atol=atol,
            method=method,
        )
        full_ts = jnp.concatenate((jnp.zeros((1,)), jnp.concatenate(ts)))
        full_ys = jnp.concatenate((y0[None, :], ys), axis=0)

        # Same combination rule integrate_protocol uses internally, so the
        # args recovered here match what was actually active at each row.
        combined_args = [
            step_args if args is None else jnp.concat((args, step_args))
            for step_args in protocol
        ]
        args_per_row = jnp.concatenate(
            [jnp.broadcast_to(combined_args[0], (1, combined_args[0].shape[0]))]
            + [
                jnp.broadcast_to(a, (step_ts.shape[0], a.shape[0]))
                for a, step_ts in zip(combined_args, ts, strict=True)
            ],
            axis=0,
        )
        return FluxOdeSimulation(
            model=self,
            ts=full_ts,
            ys=full_ys,
            args=args_per_row,
            variable_names=cast(tuple, self.variable_names),
            flux_names=cast(tuple, self.flux_names),
        )

    def simulate_to_steady_state(
        self,
        y0: jax.Array,
        max_steps: int,
        args: jax.Array | None = None,
        t1: float = 1e6,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        steady_state_rtol: float | None = None,
        steady_state_atol: float | None = None,
        method: Method = diffrax.Tsit5,
    ) -> FluxOdeSimulation:
        """Integrate to steady state and return a named, 1-row simulation result.

        Parameters
        ----------
        y0 : jax.Array
            Initial condition at ``t=0``.
        args : jax.Array or None
            Additional arguments forwarded to the RHS.
        t1 : float
            Upper bound on integration time; see
            :meth:`integrate_to_steady_state`.
        rtol : float
            Relative tolerance for the PID step-size controller.
        atol : float
            Absolute tolerance for the PID step-size controller.
        steady_state_rtol : float or None
            Relative tolerance for the steady-state event; defaults to ``rtol``.
        steady_state_atol : float or None
            Absolute tolerance for the steady-state event; defaults to ``atol``.
        max_steps : int
            Maximum number of adaptive solver steps before aborting.
        method : Method
            Diffrax Runge-Kutta solver class.

        Returns
        -------
        FluxOdeSimulation
            Named simulation result with a single row, indexed by the actual
            convergence time.
        """
        self._require_names()
        t, y = self.integrate_to_steady_state(
            y0,
            args,
            t1=t1,
            rtol=rtol,
            atol=atol,
            steady_state_rtol=steady_state_rtol,
            steady_state_atol=steady_state_atol,
            max_steps=max_steps,
            method=method,
        )
        row_args = jnp.zeros((0,)) if args is None else args
        return FluxOdeSimulation(
            model=self,
            ts=t[None],
            ys=y[None, :],
            args=row_args[None, :],
            variable_names=cast(tuple, self.variable_names),
            flux_names=cast(tuple, self.flux_names),
        )


###############################################################################
# Nodes
###############################################################################


class Node(Base):
    """Neural ODE: an MLP maps state to its derivative.

    Output is scaled by ``out_scale`` (default 0.1) to keep initial
    dynamics small and aid early training stability.

    Parameters
    ----------
    n_obs : int
        State dimension (input and output size of the MLP).
    width : int
        Hidden layer width.
    depth : int
        Number of hidden layers.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    out_scale : jax.Array or None
        Scalar multiplier applied to the MLP output.  Defaults to 0.1.
    """

    out_scale: jax.Array
    nn: eqx.nn.MLP
    derived_fn: DerivedFn

    def __init__(
        self,
        n_obs: int,
        width: int,
        depth: int,
        key: PRNGKeyArray,
        n_args: int = 0,
        out_scale: jax.Array | None = None,
        derived_fn: DerivedFn | None = None,
    ) -> None:
        self.out_scale = jnp.array([0.1]) if out_scale is None else out_scale
        self.nn = eqx.nn.MLP(
            in_size=n_obs + n_args,
            out_size=n_obs,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )
        self.derived_fn = _default_derived if derived_fn is None else derived_fn

    def __call__(
        self,
        time: PyTree,  # noqa: ARG002 ; for API stability
        y: PyTree,
        args: PyTree,
    ) -> jax.Array:
        """Evaluate the neural ODE right-hand side.

        Parameters
        ----------
        t : PyTree
            Current time (unused; kept for API compatibility).
        y : PyTree
            Current state vector.
        args : PyTree
            Additional arguments (unused; kept for API compatibility).

        Returns
        -------
        jax.Array
            Scaled MLP output ``out_scale * nn(y)``.
        """
        return self.out_scale * self.nn(jnp.concat((y, args)))

    def derived(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Get derived quantitites."""
        return self.derived_fn(t, y, args)


class FluxNode(Base):
    """Neural ODE with flux-based formulation and stoichiometric map.

    An MLP predicts reaction fluxes; ``nv`` maps them to state derivatives.

    Parameters
    ----------
    n_obs : int
        State dimension (MLP input size).
    n_flux : int
        Number of reaction fluxes (MLP output size).
    width : int
        Hidden layer width.
    depth : int
        Number of hidden layers.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    nv : Nv
        Stoichiometric map ``flux_vector -> dy/dt``.
    out_scale : jax.Array or None
        Scalar multiplier applied to the flux MLP output.  Defaults to 0.1.
    """

    out_scale: jax.Array
    flux_nn: eqx.nn.MLP
    nv: Nv
    derived_fn: DerivedFn

    def __init__(
        self,
        n_obs: int,
        n_flux: int,
        width: int,
        depth: int,
        key: PRNGKeyArray,
        nv: Nv,
        out_scale: jax.Array | None = None,
        derived_fn: DerivedFn | None = None,
    ) -> None:
        self.out_scale = jnp.array([0.1]) if out_scale is None else out_scale
        self.flux_nn = eqx.nn.MLP(
            in_size=n_obs,
            out_size=n_flux,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )
        self.nv = nv
        self.derived_fn = _default_derived if derived_fn is None else derived_fn

    def fluxes(
        self,
        t: PyTree,  # noqa: ARG002 ; for API stability
        y: PyTree,
        args: PyTree,
    ) -> jax.Array:
        """Compute scaled neural flux predictions.

        Parameters
        ----------
        t : PyTree
            Current time (unused; kept for API compatibility).
        y : PyTree
            Current state vector.
        args : PyTree
            Additional arguments (unused; kept for API compatibility).

        Returns
        -------
        jax.Array
            Flux vector ``out_scale * flux_nn(y)``.
        """
        return self.out_scale * self.flux_nn(jnp.concat((y, args)))

    def derived(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Get derived quantitites."""
        return self.derived_fn(t, y, args)

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Evaluate the neural ODE right-hand side via fluxes.

        Parameters
        ----------
        t : PyTree
            Current time.
        y : PyTree
            Current state.
        args : PyTree
            Additional arguments forwarded to ``fluxes``.

        Returns
        -------
        jax.Array
            State derivative ``nv(fluxes(t, y, args))``.
        """
        return self.nv(self.fluxes(time, y, args))


###############################################################################
# UDEs
###############################################################################


class Ude(Base):
    """Universal Differential Equation: ODE combined with a neural network.

    The ODE and neural terms are combined element-wise via ``op``
    (addition or multiplication).

    Parameters
    ----------
    ode : Ode
        Mechanistic ODE component.
    nn : Node
        Neural network component.
    op : {"+" , "*"}
        Operator for combining ODE and neural outputs.
    """

    ode: Ode
    nn: Node
    op: Callable[[jax.Array, jax.Array], jax.Array]

    def __init__(
        self,
        ode: Ode,
        nn: Node,
        op: Literal["+", "-", "*", "/"] | Callable[[jax.Array, jax.Array], jax.Array],
    ) -> None:
        ops = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }

        self.ode = ode
        self.nn = nn
        self.op = ops[op] if isinstance(op, str) else op

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Evaluate the UDE right-hand side.

        Parameters
        ----------
        t : PyTree
            Current time.
        y : PyTree
            Current state.
        args : PyTree
            Additional arguments forwarded to both ODE and neural components.

        Returns
        -------
        jax.Array
            ``op(ode(t, y, args), nn(t, y, args))``.
        """
        return self.op(self.ode(time, y, args), self.nn(time, y, args))

    def derived(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Get derived quantitites."""
        return self.ode.derived(t, y, args)


class FluxUde(Base):
    """Flux-based Universal Differential Equation.

    Combines a mechanistic flux ODE with a neural flux network via
    element-wise multiplication of their respective dy/dt outputs.

    Parameters
    ----------
    flux_ode : FluxOde
        Mechanistic flux ODE component.
    flux_nn : FluxNode
        Neural flux network component.
    """

    flux_ode: FluxOde
    flux_node: FluxNode
    op: Callable[[jax.Array, jax.Array], jax.Array]
    derived_fn: DerivedFn

    def __init__(
        self,
        flux_ode: FluxOde,
        flux_nn: FluxNode,
        op: Literal["+", "-", "*", "/"] | Callable[[jax.Array, jax.Array], jax.Array],
        derived_fn: DerivedFn | None = None,
    ) -> None:
        ops = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }

        self.op = ops[op] if isinstance(op, str) else op
        self.flux_ode = flux_ode
        self.flux_node = flux_nn
        self.derived_fn = _default_derived if derived_fn is None else derived_fn

    def ode_fluxes(self, ts: jax.Array, ys: jax.Array, args: jax.Array) -> jax.Array:
        """Return the mechanistic flux vector.

        Parameters
        ----------
        ts : jax.Array
            Time points.
        ys : jax.Array
            State values.
        args : jax.Array
            Additional arguments.

        Returns
        -------
        jax.Array
            Flux vector from the ODE component.
        """
        return jax.vmap(self.flux_ode.fluxes, in_axes=(0, 0, None))(ts, ys, args)

    def node_fluxes(self, ts: jax.Array, ys: jax.Array, args: jax.Array) -> jax.Array:
        """Return the neural flux vector.

        Parameters
        ----------
        ts : jax.Array
            Time points.
        ys : jax.Array
            State values.
        args : jax.Array
            Additional arguments.

        Returns
        -------
        jax.Array
            Flux vector from the neural network component.
        """
        return jax.vmap(self.flux_node.fluxes, in_axes=(0, 0, None))(ts, ys, args)

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Evaluate the FluxUDE right-hand side.

        Parameters
        ----------
        t : PyTree
            Current time.
        y : PyTree
            Current state.
        args : PyTree
            Additional arguments.

        Returns
        -------
        jax.Array
            Element-wise product of ODE and neural dy/dt outputs.
        """
        return self.op(self.flux_ode(time, y, args), self.flux_nn(time, y, args))


###############################################################################
# Anodes
###############################################################################


class Anode(Base):
    """Augmented Neural ODE: integrates in latent space, decodes to observations.

    The initial condition is encoded into a higher-dimensional latent
    space via ``latent_mapper``; the ODE runs there; the trajectory is
    decoded back to observation space on return.

    Parameters
    ----------
    n_obs : int
        Number of observed state dimensions.
    n_hidden : int
        Number of additional latent dimensions.
    width : int
        MLP hidden layer width.
    depth : int
        Number of MLP hidden layers.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    out_scale : jax.Array or None
        Scalar multiplier applied to the MLP output.  Defaults to 0.1.
    latent_mapper : LatentMapperFn or None
        Factory that constructs the latent mapper.
        Defaults to :class:`SoftLatentMapper`.
    """

    n_obs: int
    n_args: int
    out_scale: jax.Array
    nn: eqx.nn.MLP
    latent_mapper: LatentMapper
    derived_fn: DerivedFn

    def __init__(
        self,
        n_obs: int,
        n_hidden: int,
        width: int,
        depth: int,
        key: PRNGKeyArray,
        n_args: int = 0,
        out_scale: jax.Array | None = None,
        latent_mapper: LatentMapperFn | None = None,
        derived_fn: DerivedFn | None = None,
    ) -> None:
        self.out_scale = jnp.array([0.1]) if out_scale is None else out_scale
        self.n_obs = n_obs
        self.n_args = n_obs + n_args

        self.nn = eqx.nn.MLP(
            in_size=n_obs + n_args + n_hidden,
            out_size=n_obs + n_hidden,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )
        self.latent_mapper = (
            SoftLatentMapper(n_obs=self.n_obs, n_hidden=n_hidden, key=key)
            if latent_mapper is None
            else latent_mapper(n_obs=self.n_obs, n_hidden=n_hidden, key=key)
        )
        self.derived_fn = _default_derived if derived_fn is None else derived_fn

    def __call__(
        self,
        time: PyTree,  # noqa: ARG002 ; for API stability
        y: PyTree,
        args: PyTree,
    ) -> jax.Array:
        """Evaluate the latent-space ODE right-hand side.

        Parameters
        ----------
        t : PyTree
            Current time (unused; kept for API compatibility).
        y : PyTree
            Current latent state vector.
        args : PyTree
            Additional arguments (unused; kept for API compatibility).

        Returns
        -------
        jax.Array
            Latent derivative ``out_scale * nn(y)``.
        """
        return self.out_scale * self.nn(jnp.concat((y, args)))

    def derived(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Get derived quantitites."""
        return self.derived_fn(t, y, args)

    def integrate(
        self,
        ts: jax.Array,
        y0: jax.Array,
        max_steps: int,
        args: jax.Array | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
    ) -> jax.Array:
        """Integrate in latent space and decode to observation space.

        Parameters
        ----------
        ts : jax.Array
            Time points, shape ``(T,)``.
        y0 : jax.Array
            Initial observation, shape ``(n_obs,)``.
        max_steps : int
            Maximum solver steps.
        rtol : float
            Relative tolerance.
        atol : float
            Absolute tolerance.
        method : Method
            Diffrax solver class.

        Returns
        -------
        jax.Array
            Decoded trajectories, shape ``(T, n_obs)``.
        """
        z0 = self.latent_mapper.encode(y0)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self),
            method(),
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            args=args,
            y0=z0,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=max_steps,
        )
        zs = cast(jax.Array, sol.ys)
        return lax.cond(
            sol.result == diffrax.RESULTS.successful,
            lambda: self.latent_mapper.decode(zs),
            lambda: jnp.nan_to_num(
                zs[:, : self.n_obs],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
        )


class FluxAnode(Base):
    """Augmented flux Neural ODE with a Markov correction term.

    Integrates in latent space using two MLPs: one predicts reaction
    fluxes (``flux_nn``), the other predicts dynamics for the hidden
    latent dimensions (``markov_nn``).

    Parameters
    ----------
    n_obs : int
        Number of observed state dimensions.
    n_hidden : int
        Number of additional latent dimensions.
    n_flux : int
        Number of reaction fluxes.
    flux_width : int
        Hidden layer width for the flux MLP.
    flux_depth : int
        Number of hidden layers for the flux MLP.
    markov_width : int
        Hidden layer width for the Markov MLP.
    markov_depth : int
        Number of hidden layers for the Markov MLP.
    key : PRNGKeyArray
        JAX random key for weight initialisation.
    nv : Nv
        Stoichiometric map ``flux_vector -> dy/dt``.
    flux_scale : jax.Array or None
        Scalar multiplier for flux MLP output.  Defaults to 0.1.
    markov_scale : jax.Array or None
        Scalar multiplier for Markov MLP output.  Defaults to 0.1.
    latent_mapper : LatentMapperFn or None
        Factory that constructs the latent mapper.
        Defaults to :class:`HardLatentMapper`.
    """

    n_obs: int
    n_args: int
    nv: Nv
    flux_nn: eqx.nn.MLP
    markov_nn: eqx.nn.MLP
    flux_scale: jax.Array
    markov_scale: jax.Array
    latent_mapper: LatentMapper
    derived_fn: DerivedFn

    def __init__(
        self,
        n_obs: int,
        n_hidden: int,
        n_args: int,
        n_flux: int,
        flux_width: int,
        flux_depth: int,
        markov_width: int,
        markov_depth: int,
        key: PRNGKeyArray,
        nv: Nv,
        flux_scale: jax.Array | None = None,
        markov_scale: jax.Array | None = None,
        latent_mapper: LatentMapperFn | None = None,
        derived_fn: DerivedFn | None = None,
    ) -> None:
        self.n_obs = n_obs
        self.n_args = n_args
        self.nv = nv
        self.flux_scale = jnp.array([0.1]) if flux_scale is None else flux_scale
        self.markov_scale = jnp.array([0.1]) if markov_scale is None else markov_scale
        n_latent = n_obs + n_hidden

        self.flux_nn = eqx.nn.MLP(
            in_size=n_latent + n_args,
            out_size=n_flux,
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
        self.latent_mapper = (
            HardLatentMapper(n_obs, n_hidden, key)
            if latent_mapper is None
            else latent_mapper(n_obs, n_hidden, key)
        )
        self.derived_fn = _default_derived if derived_fn is None else derived_fn

    def fluxes(
        self,
        t: PyTree,  # noqa: ARG002 ; for API stability
        y: PyTree,
        args: PyTree,
    ) -> jax.Array:
        """Compute scaled neural flux predictions.

        Parameters
        ----------
        t : PyTree
            Current time (unused; kept for API compatibility).
        y : PyTree
            Current latent state vector.
        args : PyTree
            Additional arguments (unused; kept for API compatibility).

        Returns
        -------
        jax.Array
            Flux vector ``flux_scale * flux_nn(y)``.
        """
        return self.flux_scale * self.flux_nn(jnp.concat((y, args)))

    def derived(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Get derived quantitites."""
        return self.derived_fn(t, y, args)

    def __call__(self, time: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Evaluate the latent-space ODE right-hand side.

        Concatenates the stoichiometric flux term with the Markov
        correction term for the hidden latent dimensions.

        Parameters
        ----------
        t : PyTree
            Current time.
        y : PyTree
            Current latent state vector.
        args : PyTree
            Additional arguments forwarded to ``fluxes``.

        Returns
        -------
        jax.Array
            Latent derivative ``concat(nv(fluxes), markov_scale * markov_nn(y))``.
        """
        return jnp.concat(
            (
                self.nv(self.fluxes(time, y, args)),
                self.markov_scale * self.markov_nn(jnp.concat((y, args))),
            )
        )

    def integrate(
        self,
        ts: jax.Array,
        y0: jax.Array,
        max_steps: int,
        args: jax.Array | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
    ) -> jax.Array:
        """Integrate in latent space and decode to observation space.

        Parameters
        ----------
        ts : jax.Array
            Time points, shape ``(T,)``.
        y0 : jax.Array
            Initial observation, shape ``(n_obs,)``.
        max_steps : int
            Maximum solver steps.
        rtol : float
            Relative tolerance.
        atol : float
            Absolute tolerance.
        method : Method
            Diffrax solver class.

        Returns
        -------
        jax.Array
            Decoded trajectories, shape ``(T, n_obs)``.
        """
        z0 = self.latent_mapper.encode(y0)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self),
            method(),
            t0=ts[0],
            t1=ts[-1],
            args=args,
            dt0=None,
            y0=z0,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=max_steps,
        )
        zs = cast(jax.Array, sol.ys)
        return lax.cond(
            sol.result == diffrax.RESULTS.successful,
            lambda: self.latent_mapper.decode(zs),
            lambda: jnp.nan_to_num(
                zs[:, : self.n_obs],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
        )
