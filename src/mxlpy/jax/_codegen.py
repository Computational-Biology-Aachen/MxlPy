"""JAX export for mxlpy models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import sympy
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve

from mxlpy.symbolic import to_symbolic_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from diffrax import AbstractSolver, AbstractStepSizeController

    from mxlpy.model import Model

__all__ = [
    "JaxExport",
    "to_jax_export",
]


@dataclass
class JaxExport:
    """JAX-compatible export of an mxlpy Model.

    Contains a Diffrax-style RHS function ``f(t, y, args)`` derived from
    the model's symbolic equations, together with helpers for building
    initial conditions and parameter arrays.

    Attributes
    ----------
    rhs
        ODE right-hand side with signature ``(t, y, args) -> jnp.ndarray``.
        Compatible with ``diffrax.ODETerm``.
    variable_names
        Ordered names of state variables.  Index ``i`` in the ``y`` vector
        corresponds to ``variable_names[i]``.
    parameter_names
        Ordered names of kinetic parameters.  For models without surrogates
        ``args`` is a 1-D JAX array whose index ``i`` corresponds to
        ``parameter_names[i]``.  For models with surrogates ``args`` is a
        ``(params_array, surrogates_dict)`` tuple.
    initial_conditions
        Default initial values keyed by variable name.
    parameter_values
        Default parameter values keyed by parameter name.
    has_surrogates
        ``True`` when the model contained surrogate (e.g. neural network)
        components.

    Examples
    --------
    >>> export = to_jax_export(model)
    >>> y0 = export.get_y0()
    >>> args = export.get_args()
    >>> dydt = export.rhs(0.0, y0, args)

    """

    rhs: Callable[[Any, Any, Any], Any]
    variable_names: list[str]
    parameter_names: list[str]
    initial_conditions: dict[str, float]
    parameter_values: dict[str, float]
    has_surrogates: bool

    def get_y0(self) -> Any:
        """Return initial conditions as a JAX array.

        Returns
        -------
        jnp.ndarray
            1-D array of initial values in ``variable_names`` order.

        Examples
        --------
        >>> y0 = export.get_y0()

        """
        return jnp.array([self.initial_conditions[n] for n in self.variable_names])

    def get_args(self, **overrides: float) -> Any:
        """Return kinetic parameters as a JAX array.

        Parameters
        ----------
        **overrides
            Optional per-parameter overrides.  Only the kinetic parameters
            are affected; surrogate modules must be supplied separately when
            ``has_surrogates`` is ``True``.

        Returns
        -------
        jnp.ndarray
            1-D array of parameter values in ``parameter_names`` order.

        Examples
        --------
        >>> args = export.get_args(k1=2.0)

        """
        vals = self.parameter_values.copy()
        vals.update(overrides)
        return jnp.array([vals[n] for n in self.parameter_names])

    def simulate(
        self,
        y0: Any,
        t0: float,
        t1: float,
        args: Any,
        *,
        time_points: Any | None = None,
        steps: int = 100,
        dt0: float | None = None,
        solver: AbstractSolver | None = None,
        stepsize_controller: AbstractStepSizeController | None = None,
        max_steps: int | None = None,
    ) -> tuple[Any, Any]:
        """Integrate the model using Diffrax.

        Parameters
        ----------
        y0
            Initial state vector (JAX array of shape ``(n_vars,)``).
        t0
            Start time.
        t1
            End time.
        args
            Parameter array (or ``(params, surrogates)`` tuple for UDE models).
        time_points
            Explicit time points at which to save the solution.  When given,
            overrides ``steps``.
        steps
            Number of evenly-spaced save points between ``t0`` and ``t1``.
            Used only when ``time_points`` is ``None``.  Default: 100.
        dt0
            Initial step size.  Defaults to ``(t1 - t0) / steps``.
        solver
            Diffrax solver.  Defaults to ``Dopri5()``.
        stepsize_controller
            Step-size controller.  Defaults to
            ``PIDController(rtol=1e-6, atol=1e-6)``.
        max_steps
            Maximum number of internal solver steps (``None`` = unlimited).

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            ``(times, values)`` where ``times`` has shape ``(n_steps,)`` and
            ``values`` has shape ``(n_steps, n_vars)``.

        Examples
        --------
        >>> times, values = export.simulate(export.get_y0(), 0.0, 10.0, export.get_args())

        """
        if solver is None:
            solver = Dopri5()
        if stepsize_controller is None:
            stepsize_controller = PIDController(rtol=1e-6, atol=1e-6)
        if time_points is None:
            time_points = jnp.linspace(t0, t1, steps)
        if dt0 is None:
            dt0 = float(t1 - t0) / steps

        sol = diffeqsolve(
            ODETerm(self.rhs),
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            saveat=SaveAt(ts=time_points),  # type: ignore[arg-type]
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
            args=args,
        )
        return sol.ts, sol.ys


def to_jax_export(model: Model) -> JaxExport:
    """Convert an mxlpy Model to a JAX-compatible export.

    Uses the model's symbolic representation as an intermediate step:
    each rate function is parsed to SymPy via AST analysis, the full ODE
    system is assembled symbolically, and each equation is then compiled
    to a JAX function via ``sympy.lambdify``.

    Parameters
    ----------
    model
        Model to export.

    Returns
    -------
    JaxExport
        JAX export containing the RHS function and metadata.

    Raises
    ------
    ValueError
        If any rate function cannot be converted to a SymPy expression.

    Examples
    --------
    >>> export = to_jax_export(model)
    >>> y0 = export.get_y0()
    >>> args = export.get_args()
    >>> # Use with Diffrax
    >>> times, values = export.simulate(y0, 0.0, 10.0, args)
    >>> # Differentiate through parameters
    >>> import jax
    >>> grad_fn = jax.grad(lambda a: export.rhs(0.0, y0, a).sum(), argnums=0)
    >>> grad = grad_fn(args)

    """
    # Build symbolic model — raises ValueError for unparseable functions
    sym = to_symbolic_model(model)

    variable_names = list(sym.variables.keys())
    parameter_names = list(sym.parameters.keys())

    var_syms = list(sym.variables.values())
    param_syms = list(sym.parameters.values())
    all_syms = var_syms + param_syms

    # Lambdify each ODE equation with jax.numpy as the math backend
    lambdified: list[Callable[..., Any]] = [
        sympy.lambdify(all_syms, eq, modules=[jnp, "numpy"]) for eq in sym.eqs
    ]

    n_vars = len(variable_names)
    n_params = len(parameter_names)

    has_surrogates = bool(model._surrogates)  # noqa: SLF001

    if not has_surrogates:
        # Pure mechanistic model — args is a flat parameter array
        def rhs(_t: Any, y: Any, args: Any) -> Any:
            var_vals = [y[i] for i in range(n_vars)]
            par_vals = [args[i] for i in range(n_params)]
            all_vals = var_vals + par_vals
            return jnp.stack([fn(*all_vals) for fn in lambdified])

    else:
        # UDE model — args is (params_array, surrogates_dict)
        # Pre-compute static index maps for surrogate inputs
        _var_idx = {name: i for i, name in enumerate(variable_names)}
        _par_idx = {name: i for i, name in enumerate(parameter_names)}

        surrogate_configs: list[dict[str, Any]] = []
        for surr_name, surr in model._surrogates.items():  # noqa: SLF001
            arg_sources: list[tuple[int, bool]] = []
            for a in surr.args:
                if a in _var_idx:
                    arg_sources.append((_var_idx[a], True))
                elif a in _par_idx:
                    arg_sources.append((_par_idx[a], False))
                else:
                    msg = (
                        f"Surrogate '{surr_name}' argument '{a}' is not a variable "
                        "or parameter. Derived quantities as surrogate inputs are "
                        "not yet supported."
                    )
                    raise ValueError(msg)

            # (var_idx, coefficient) pairs for each surrogate output
            stoich_by_output: list[list[tuple[int, float]]] = []
            for stoich_dict in surr.stoichiometries.values():
                contributions: list[tuple[int, float]] = []
                for var_name, coeff in stoich_dict.items():
                    if isinstance(coeff, float | int):
                        contributions.append((_var_idx[var_name], float(coeff)))
                stoich_by_output.append(contributions)

            surrogate_configs.append(
                {
                    "name": surr_name,
                    "arg_sources": arg_sources,
                    "stoich_by_output": stoich_by_output,
                }
            )

        def rhs(_t: Any, y: Any, args: Any) -> Any:
            params, surrogates = args
            var_vals = [y[i] for i in range(n_vars)]
            par_vals = [params[i] for i in range(n_params)]
            all_vals = var_vals + par_vals

            dxdt = jnp.stack([fn(*all_vals) for fn in lambdified])

            for config in surrogate_configs:
                surr_module = surrogates[config["name"]]
                surr_input = jnp.stack(
                    [
                        y[idx] if is_var else params[idx]
                        for idx, is_var in config["arg_sources"]
                    ]
                )
                surr_fluxes = surr_module(surr_input)
                for out_idx, contributions in enumerate(config["stoich_by_output"]):
                    flux = surr_fluxes[out_idx]
                    for var_idx, coeff in contributions:
                        dxdt = dxdt.at[var_idx].add(coeff * flux)

            return dxdt

    return JaxExport(
        rhs=rhs,
        variable_names=variable_names,
        parameter_names=parameter_names,
        initial_conditions=model.get_initial_conditions(),
        parameter_values=model.get_parameter_values(),
        has_surrogates=has_surrogates,
    )
