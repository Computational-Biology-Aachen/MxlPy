"""Universal Differential Equation base class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import equinox as eqx

__all__ = ["JaxFit", "UDE"]

if TYPE_CHECKING:
    from mxlpy.jax._codegen import JaxExport


class UDE(eqx.Module):
    """Base class for Universal Differential Equations.

    Combines a mechanistic ODE exported from an mxlpy :class:`~mxlpy.Model`
    with an Equinox neural network module that provides an additive correction.

    Subclass and override :meth:`__call__` to define how the mechanistic and
    neural contributions are combined.  The default implementation returns
    the pure mechanistic derivative (no neural correction).

    The ``ode`` field is declared **static** so that JAX never traces through
    the mechanistic model itself — only the neural network weights receive
    gradients during training.

    Parameters
    ----------
    ode
        Exported mechanistic model from :meth:`~mxlpy.Model.to_jax`.
    mlp
        Equinox neural network module providing the additive correction.

    Examples
    --------
    >>> class MyUDE(UDE):
    ...     q_max: float
    ...     i_max: float
    ...
    ...     def __call__(self, t, y, args):
    ...         mech = self.ode.rhs(t, y, args)
    ...         nn_in = jnp.array([y[0] / self.q_max, args[0] / self.i_max])
    ...         return mech + self.mlp(nn_in)

    """

    ode: JaxExport = eqx.field(static=True)
    mlp: eqx.nn.MLP

    def __call__(self, t: Any, y: Any, args: Any) -> Any:
        """Evaluate the UDE right-hand side.

        Override this in subclasses to add the neural correction.

        Parameters
        ----------
        t
            Current time.
        y
            State vector (JAX array of shape ``(n_vars,)``).
        args
            Parameter array produced by the protocol for this segment.

        Returns
        -------
        jnp.ndarray
            Derivatives ``dy/dt`` of shape ``(n_vars,)``.

        """
        return self.ode.rhs(t, y, args)


@dataclass
class JaxFit:
    """Result of a JAX-based fitting run.

    Attributes
    ----------
    ude
        Trained UDE module (Equinox pytree with updated neural weights).
    losses
        MSE loss value recorded after each epoch.

    Examples
    --------
    >>> result = fit.protocol_time_course(ude, protocol, data, minimizer=...)
    >>> trained_ude = result.ude
    >>> plt.semilogy(result.losses)

    """

    ude: eqx.Module
    losses: list[float]
