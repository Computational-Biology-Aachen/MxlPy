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

from mxlpy.meta._via_sym_repr import generate_model_code_jax
from mxlpy.model import Model

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


class JaxModel(Protocol):
    """Protocol for JAX-compatible ODE models.

    Any object implementing ``__call__`` and ``integrate`` can be used
    with the training utilities in ``train.py``.
    """

    def __call__(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Evaluate the ODE right-hand side at time ``t`` and state ``y``."""
        ...

    def integrate(
        self,
        ts: jax.Array,
        y0: jax.Array,
        max_steps: int = 4096,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: Method = diffrax.Tsit5,
    ) -> jax.Array:
        """Integrate the model from ``ts[0]`` to ``ts[-1]`` and return saved states."""
        ...


###############################################################################
# ABC
###############################################################################


class Base(eqx.Module, ABC):
    """Abstract base class for equinox ODE models.

    Provides a default ``integrate`` implementation via diffrax.
    Failed solves return zeros rather than propagating NaN values.
    """

    @abstractmethod
    def __call__(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Evaluate the ODE right-hand side at time ``t`` and state ``y``."""
        ...

    def integrate(
        self,
        ts: jax.Array,
        y0: jax.Array,
        max_steps: int = 4096,
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
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self),
            method(),
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=y0,
            args=jnp.array([]),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            saveat=diffrax.SaveAt(ts=ts),
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


class SoftLatentMapper:
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
            in_features=self.n_hidden,
            out_features=self.n_latent,
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
        return jax.vmap(self.decoder, in_axes=(0, None))(zs)


class FullLatentMapper:
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

    def __init__(self, n_obs: int, n_hidden: int, key: PRNGKeyArray) -> None:
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
        return jax.vmap(self.decoder, in_axes=(0, None))(zs)


class HardLatentMapper:
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


class MixedLatentMapper:
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
        learned = jax.vmap(self.decoder, in_axes=(0, None))(zs)
        return (1 - alpha) * hard + alpha * learned


###############################################################################
# Odes
###############################################################################


class Ode(Base):
    """ODE model with trainable parameter vector.

    Parameters are prepended to ``args`` at each evaluation so that
    ``rhs`` receives ``concat(pars, args)`` as its third argument.

    Parameters
    ----------
    rhs : Rhs
        Right-hand-side function ``(t, y, args) -> dy/dt``.
    pars : jax.Array
        Trainable parameter vector.
    """

    rhs: Rhs
    pars: jax.Array  # trainable parameters

    def __init__(self, rhs: Rhs, pars: jax.Array) -> None:
        self.rhs = rhs
        self.pars = pars

    def __call__(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Evaluate the ODE right-hand side.

        Parameters
        ----------
        t : PyTree
            Current time.
        y : PyTree
            Current state.
        args : PyTree
            Additional arguments appended after ``pars``.

        Returns
        -------
        jax.Array
            State derivative ``dy/dt``.
        """
        return self.rhs(t, y, jnp.concat((self.pars, args)))

    @classmethod
    def from_mxlpy(
        cls,
        mxlpy_model: Model,
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

        generated = {}

        exec(codegen.imports, globals(), generated)  # noqa: S102
        exec(codegen.model, globals(), generated)  # noqa: S102

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
    """

    fluxes: Rhs
    nv: Nv
    pars: jax.Array  # trainable parameters

    def __init__(self, fluxes: Rhs, nv: Nv, pars: jax.Array) -> None:
        self.fluxes = fluxes
        self.nv = nv
        self.pars = pars

    def __call__(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
        """Evaluate the ODE right-hand side via fluxes.

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
            State derivative ``dy/dt``.
        """
        return self.nv(self.fluxes(t, y, args))

    @classmethod
    def from_mxlpy(
        cls,
        mxlpy_model: Model,
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

        generated = {}

        exec(codegen.imports, globals(), generated)  # noqa: S102
        exec(codegen.fluxes, globals(), generated)  # noqa: S102
        exec(codegen.nv, globals(), generated)  # noqa: S102

        model_pars = mxlpy_model.get_parameter_values()
        pars = jnp.array([model_pars[k] for k in parameters_to_fit])
        return cls(
            fluxes=generated["fluxes"],
            nv=generated["nv"],
            pars=pars,
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

    def __init__(
        self,
        n_obs: int,
        width: int,
        depth: int,
        key: PRNGKeyArray,
        out_scale: jax.Array | None = None,
    ) -> None:
        self.out_scale = jnp.array([0.1]) if out_scale is None else out_scale
        self.nn = eqx.nn.MLP(
            in_size=n_obs,
            out_size=n_obs,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )

    def __call__(
        self,
        t: PyTree,  # noqa: ARG002 ; for API stability
        y: PyTree,
        args: PyTree,  # noqa: ARG002 ; for API stability
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
        return self.out_scale * self.nn(y)


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

    def __init__(
        self,
        n_obs: int,
        n_flux: int,
        width: int,
        depth: int,
        key: PRNGKeyArray,
        nv: Nv,
        out_scale: jax.Array | None = None,
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

    def fluxes(
        self,
        t: PyTree,  # noqa: ARG002 ; for API stability
        y: PyTree,
        args: PyTree,  # noqa: ARG002 ; for API stability
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
        return self.out_scale * self.flux_nn(y)

    def __call__(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
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
        return self.nv(self.fluxes(t, y, args))


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

    def __call__(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
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
        return self.op(self.ode(t, y, args), self.nn(t, y, args))


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

    def __init__(
        self,
        flux_ode: FluxOde,
        flux_nn: FluxNode,
        op: Literal["+", "-", "*", "/"] | Callable[[jax.Array, jax.Array], jax.Array],
    ) -> None:
        ops = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }

        self.op = ops[op] if isinstance(op, str) else op
        self.flux_ode = flux_ode
        self.flux_nn = flux_nn

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

    def __call__(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
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
        return self.op(self.flux_ode(t, y, args), self.flux_nn(t, y, args))


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
    out_scale: jax.Array
    nn: eqx.nn.MLP
    latent_mapper: LatentMapper

    def __init__(
        self,
        n_obs: int,
        n_hidden: int,
        width: int,
        depth: int,
        key: PRNGKeyArray,
        out_scale: jax.Array | None = None,
        latent_mapper: LatentMapperFn | None = None,
    ) -> None:
        self.out_scale = jnp.array([0.1]) if out_scale is None else out_scale
        self.n_obs = n_obs
        self.nn = eqx.nn.MLP(
            in_size=n_obs,
            out_size=n_obs,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )
        self.latent_mapper = (
            SoftLatentMapper(n_obs, n_hidden, key)
            if latent_mapper is None
            else latent_mapper(n_obs, n_hidden, key)
        )

    def __call__(
        self,
        t: PyTree,  # noqa: ARG002 ; for API stability
        y: PyTree,
        args: PyTree,  # noqa: ARG002 ; for API stability
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
        return self.out_scale * self.nn(y)

    def integrate(
        self,
        ts: jax.Array,
        y0: jax.Array,
        max_steps: int = 4096,
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
            y0=z0,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=max_steps,
        )
        zs = cast(jax.Array, sol.ys)
        return lax.cond(
            sol.result == diffrax.RESULTS.successful,
            lambda: jax.vmap(self.latent_mapper.decode, in_axes=(0, None))(zs),
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
    flux_scale : jax.Array or None
        Scalar multiplier for flux MLP output.  Defaults to 0.1.
    markov_scale : jax.Array or None
        Scalar multiplier for Markov MLP output.  Defaults to 0.1.
    latent_mapper : LatentMapperFn or None
        Factory that constructs the latent mapper.
        Defaults to :class:`HardLatentMapper`.
    """

    n_obs: int
    nv: Nv
    encoder: eqx.nn.Linear
    flux_nn: eqx.nn.MLP
    markov_nn: eqx.nn.MLP
    decoder: eqx.nn.Linear
    flux_scale: jax.Array
    markov_scale: jax.Array
    latent_mapper: LatentMapper

    def __init__(
        self,
        n_obs: int,
        n_hidden: int,
        n_flux: int,
        flux_width: int,
        flux_depth: int,
        markov_width: int,
        markov_depth: int,
        key: PRNGKeyArray,
        flux_scale: jax.Array | None = None,
        markov_scale: jax.Array | None = None,
        latent_mapper: LatentMapperFn | None = None,
    ) -> None:
        self.flux_scale = jnp.array([0.1]) if flux_scale is None else flux_scale
        self.markov_scale = jnp.array([0.1]) if markov_scale is None else markov_scale
        n_latent = n_obs + n_hidden
        self.flux_nn = eqx.nn.MLP(
            in_size=n_latent,
            out_size=n_flux,
            width_size=flux_width,
            depth=flux_depth,
            activation=jax.nn.softplus,
            key=key,
        )
        self.markov_nn = eqx.nn.MLP(
            in_size=n_latent,
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

    def fluxes(
        self,
        t: PyTree,  # noqa: ARG002 ; for API stability
        y: PyTree,
        args: PyTree,  # noqa: ARG002 ; for API stability
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
        return self.flux_scale * self.flux_nn(y)

    def __call__(self, t: PyTree, y: PyTree, args: PyTree) -> jax.Array:
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
                self.nv(self.fluxes(t, y, args)),
                self.flux_scale * self.markov_nn(y),
            )
        )

    def integrate(
        self,
        ts: jax.Array,
        y0: jax.Array,
        max_steps: int = 4096,
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
