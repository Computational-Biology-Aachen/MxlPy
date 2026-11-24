# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Forward and finite-difference sensitivities, see the package docstring."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy.integrators import DefaultIntegrator
from mxlpy.simulator import Simulator
from mxlpy.symbolic.jacobian import (
    CompiledDerivatives,
    all_parameter_values,
    compile_derivatives,
    constant_values,
)
from mxlpy.types import InitialAssignment, Result

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mxlpy._ode_builder import OdeModelBuilder
    from mxlpy._steady_state_builder import SteadyStateModelBuilder
    from mxlpy.integrators import IntegratorType
    from mxlpy.types import Array, ArrayLike, Rhs

__all__ = ["ForwardMethod", "ForwardSensitivities", "steady_state", "time_course"]

_LOGGER = logging.getLogger(__name__)
_WARNED: set[str] = set()

type ForwardMethod = Literal["auto", "forward", "finite_difference"]
type _Model = KineticModelBuilder | OdeModelBuilder
type _AnyModel = KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder


@dataclass(slots=True)
class ForwardSensitivities:
    """Sensitivities of a simulation.

    Attributes
    ----------
    simulated
        Simulated values (time x variables, derived quantities and reactions)
    derivatives
        For every parameter / variable initial value, the derivative of
        `simulated` w.r.t. it, with the same shape as `simulated`
    method
        How the derivatives were computed

    """

    simulated: pd.DataFrame
    derivatives: dict[str, pd.DataFrame]
    method: Literal["forward", "finite_difference"]

    @property
    def wrt(self) -> list[str]:
        """Names of the parameters / variables differentiated w.r.t."""
        return list(self.derivatives)

    def to_array(self) -> Array:
        """Derivatives as array of shape (time, column, parameter)."""
        return np.stack([s.to_numpy() for s in self.derivatives.values()], axis=-1)


###############################################################################
# Helpers
###############################################################################


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        _LOGGER.warning(msg)


def _step(value: float, rel: float) -> float:
    return rel * abs(value) if value != 0 else rel


def _values(model: _Model, variables: pd.DataFrame) -> pd.DataFrame:
    """Variables, derived quantities and reactions, excluding parameters."""
    if isinstance(model, KineticModelBuilder):
        return model.get_args_time_course(
            variables=variables,
            include_variables=True,
            include_parameters=False,
            include_derived_parameters=False,
            include_derived_variables=True,
            include_reactions=True,
            include_surrogate_variables=True,
            include_surrogate_fluxes=True,
            include_readouts=True,
        )
    args = model.get_args_time_course(
        variables=variables,
        include_parameters=False,
        include_derived_parameters=False,
        include_derived_variables=True,
        include_diff_eqs=False,
        include_surrogate_variables=True,
        include_surrogate_targets=False,
        include_readouts=True,
    )
    extra = [c for c in args.columns if c not in variables.columns]
    return pd.concat([variables, args.loc[:, extra]], axis=1)


def _select_times(time: Array, time_points: Array) -> NDArray[np.bool_]:
    """Mask of the rows of `time` that are one of the requested time points."""
    return np.isclose(time[:, None], time_points[None, :], rtol=1e-12, atol=1e-12).any(
        axis=1
    )


def _assigned_parameters(model: _AnyModel) -> list[str]:
    """Parameters computed from other parameters by an initial assignment."""
    return [
        k
        for k, v in model.get_raw_parameters(as_copy=False).items()
        if isinstance(v.value, InitialAssignment)
    ]


def _free_parameters(model: _AnyModel) -> list[str]:
    """Parameters that can be perturbed: not computed from other parameters."""
    assigned = set(_assigned_parameters(model))
    return [k for k in model.get_addressable_names() if k not in assigned]


def _computed_from(model: _AnyModel) -> dict[str, list[str]]:
    """Derived and assigned parameters with the free parameters they depend on.

    Resolved transitively through other derived and assigned parameters.
    """
    args = {
        k: v.value.args
        for k, v in model.get_raw_parameters(as_copy=False).items()
        if isinstance(v.value, InitialAssignment)
    } | {k: v.args for k, v in model.get_derived_parameters().items()}
    free = set(_free_parameters(model))

    def resolve(name: str, seen: frozenset[str]) -> list[str]:
        out: list[str] = []
        for a in args[name]:
            if a in free:
                out.append(a)
            elif a in args and a not in seen:
                out.extend(resolve(a, seen | {a}))
        return list(dict.fromkeys(out))

    return {k: resolve(k, frozenset({k})) for k in args}


def _check_not_computed(model: _AnyModel, names: list[str]) -> None:
    """Raise for derived and assigned parameters, pointing to their sources.

    Perturbing them is ill-defined: setting a value replaces the assignment
    with a constant instead of perturbing it.
    """
    sources = _computed_from(model)
    if bad := {k: sources[k] for k in names if k in sources}:
        computed = ", ".join(f"{k!r} (from {v})" for k, v in bad.items())
        msg = f"Can't perturb derived parameters or those set by initial assignments: {computed}. Use the parameters they are computed from instead"
        raise ValueError(msg)


def _check_parameters(model: _AnyModel, names: list[str]) -> None:
    """Raise unless `names` are distinct parameters that can be perturbed."""
    _check_not_computed(model, names)
    free = set(_free_parameters(model))
    if unknown := [k for k in names if k not in free]:
        msg = f"Unknown parameters {unknown}"
        raise ValueError(msg)
    if len(set(names)) != len(names):
        msg = "Duplicate names in parameters"
        raise ValueError(msg)


def _check_names(model: _Model, wrt: list[str]) -> tuple[list[str], list[str]]:
    """Split `wrt` into parameters and variables (initial values)."""
    _check_not_computed(model, wrt)
    variables = set(model.get_variable_names())
    parameters = set(_free_parameters(model))
    if unknown := [k for k in wrt if k not in variables | parameters]:
        msg = f"Unknown parameters / variables {unknown}"
        raise ValueError(msg)
    if len(set(wrt)) != len(wrt):
        msg = "Duplicate names in parameters"
        raise ValueError(msg)
    return [k for k in wrt if k in parameters], [k for k in wrt if k in variables]


def _parameter_fd(
    model: _Model, parameter: str, displacement: float, read: Callable[[], Array]
) -> Array:
    """Central difference of `read()` w.r.t. a parameter, restoring it after."""
    value = all_parameter_values(model)[parameter]
    h = _step(value, displacement)
    try:
        model.update_addressable({parameter: value + h})
        up = read()
        model.update_addressable({parameter: value - h})
        down = read()
    finally:
        model.update_addressable({parameter: value})
    return (up - down) / (2 * h)


###############################################################################
# Forward sensitivities
###############################################################################


@dataclass
class _Derivatives:
    """Evaluates the compiled derivatives for the current model parameters."""

    model: _Model
    compiled: CompiledDerivatives
    _cache: object = None
    _params: Array = field(default_factory=lambda: np.empty(0))
    _consts: Array = field(default_factory=lambda: np.empty(0))

    def __post_init__(self) -> None:
        c = self.compiled
        self.n = len(self.model.get_variable_names())
        self.m = len(c.wrt)
        self.n_rhs = len(c.rhs_nonzero)
        self.rhs_idx = (
            tuple(np.array(i, dtype=np.intp) for i in zip(*c.rhs_nonzero, strict=True))
            if c.rhs_nonzero
            else None
        )
        self.out_idx = (
            tuple(np.array(i, dtype=np.intp) for i in zip(*c.out_nonzero, strict=True))
            if c.out_nonzero
            else None
        )

    def _eval(self, t: float, x: Array) -> Array:
        # Every model change replaces its cache, see `jacobian_fn`
        if (current := self.model._cache) is None or current is not self._cache:  # noqa: SLF001
            values = all_parameter_values(self.model)
            self._cache = self.model._cache  # noqa: SLF001
            self._params = np.array(
                [values[k] for k in self.model.get_addressable_names()], dtype=float
            )
            self._consts = constant_values(self.model, self.compiled.constant_names)
        with np.errstate(all="ignore"):
            out = self.compiled.fn(t, x, self._params, self._consts)
        return np.asarray(out, dtype=float)

    def rhs(self, t: float, x: Array) -> tuple[Array, Array]:
        """D rhs / d x and d rhs / d wrt."""
        full = np.zeros((self.n, self.n + self.m))
        if self.rhs_idx is not None:
            full[self.rhs_idx] = self._eval(t, x)[: self.n_rhs]
        return full[:, : self.n], full[:, self.n :]

    def outputs(self, t: float, x: Array) -> tuple[Array, Array]:
        """D outputs / d x and d outputs / d wrt."""
        full = np.zeros((len(self.compiled.outputs), self.n + self.m))
        if self.out_idx is not None:
            full[self.out_idx] = self._eval(t, x)[self.n_rhs :]
        return full[:, : self.n], full[:, self.n :]


@dataclass
class _Setup:
    """Everything needed to integrate the forward sensitivities."""

    derivs: _Derivatives
    # d (compiled wrt) / d (requested), constant in time
    chain: Array
    # d x0 / d (requested)
    s0: Array
    var_names: list[str]
    outputs: list[str]


def _raw_initial_values(model: _Model) -> dict[str, float | InitialAssignment]:
    """Initial values of all states, including those of hidden-state blocks."""
    hidden: dict[str, float | InitialAssignment] = {
        **model._hidden_state_initial_values(),  # noqa: SLF001
        **model._hidden_state_initial_assignments(),  # noqa: SLF001
    }
    if isinstance(model, KineticModelBuilder):
        return {
            k: v.initial_value
            for k, v in model.get_raw_variables(as_copy=False).items()
        } | hidden
    return {
        k: v.initial_value for k, v in model.get_raw_diff_eqs(as_copy=False).items()
    } | hidden


def _initial_sources(model: _Model, *, y0_from_model: bool) -> set[str] | None:
    """Free parameters initial assignments depend on, None if unknown.

    Of parameters and, if the initial values come from the model, of
    variables. Only these need the (numerical) chain-rule factors.
    """
    sources = _computed_from(model)
    out: set[str] = set()
    for k in _assigned_parameters(model):
        out |= set(sources[k])
    if not y0_from_model:
        return out
    free = set(_free_parameters(model))
    for v in _raw_initial_values(model).values():
        if not isinstance(v, InitialAssignment):
            continue
        for a in v.args:
            if a in free:
                out.add(a)
            elif a in sources:
                out |= set(sources[a])
            else:
                return None
    return out


def _chain_factors(
    model: _Model,
    wrt: list[str],
    *,
    y0_from_model: bool,
    displacement: float,
) -> tuple[list[str], Array, Array]:
    """Constant chain-rule factors of the requested parameters / initial values.

    Initial assignments of parameters and variables are computed once from
    the parameters, so their derivatives are constant and taken numerically.
    Returns the parameters to compile derivatives for (the requested ones,
    then those set by initial assignments), d (compiled) / d (requested) and
    d x0 / d (requested).
    """
    pars, vars_ = _check_names(model, wrt)
    var_names = model.get_variable_names()
    derived = _assigned_parameters(model)
    compiled_wrt = pars + derived

    m = len(wrt)
    chain = np.zeros((len(compiled_wrt), m))
    s0 = np.zeros((len(var_names), m))
    sources = _initial_sources(model, y0_from_model=y0_from_model)

    def read() -> Array:
        values = all_parameter_values(model)
        x0 = model.get_initial_conditions()
        return np.array(
            [values[k] for k in derived]
            + [x0[k] for k in var_names] * int(y0_from_model),
            dtype=float,
        )

    for j, name in enumerate(wrt):
        if name in vars_:
            s0[var_names.index(name), j] = 1.0
            continue
        chain[compiled_wrt.index(name), j] = 1.0
        if sources is None or name in sources:
            d = _parameter_fd(model, name, displacement, read)
            chain[len(pars) :, j] = d[: len(derived)]
            if y0_from_model:
                s0[:, j] = d[len(derived) :]
    return compiled_wrt, chain, s0


def _setup(
    model: _Model,
    wrt: list[str],
    outputs: list[str],
    *,
    y0_from_model: bool,
    displacement: float,
) -> _Setup:
    """Compile the derivatives and the constant chain-rule factors."""
    compiled_wrt, chain, s0 = _chain_factors(
        model, wrt, y0_from_model=y0_from_model, displacement=displacement
    )
    compiled = compile_derivatives(model, wrt=compiled_wrt, outputs=outputs)
    return _Setup(
        derivs=_Derivatives(model, compiled),
        chain=chain,
        s0=s0,
        var_names=model.get_variable_names(),
        outputs=outputs,
    )


@dataclass
class _AugmentedRhs:
    """Model rhs plus `dS/dt = J S + df/dtheta` for the flattened `S`."""

    model: _Model
    setup: _Setup

    def __post_init__(self) -> None:
        self.n = len(self.setup.var_names)
        self.m = self.setup.chain.shape[1]

    def __call__(self, t: float, z: Array) -> Array:
        x = np.asarray(z[: self.n], dtype=float)
        s = np.asarray(z[self.n :], dtype=float).reshape(self.n, self.m)
        jac, dfdp = self.setup.derivs.rhs(t, x)
        f = np.asarray(self.model(t, x), dtype=float)
        return np.concatenate([f, (jac @ s + dfdp @ self.setup.chain).ravel()])

    def jacobian(self, t: float, z: Array) -> Array:
        """Block-diagonal approximation, ignoring the second derivatives.

        Only used for the Newton iterations of implicit solvers, so this
        affects their convergence, not the accuracy of the solution.
        """
        jac, _ = self.setup.derivs.rhs(t, np.asarray(z[: self.n], dtype=float))
        size = self.n * (1 + self.m)
        out = np.zeros((size, size))
        out[: self.n, : self.n] = jac
        out[self.n :, self.n :] = np.kron(jac, np.eye(self.m))
        return out


def _assemble(
    setup: _Setup,
    wrt: list[str],
    values: pd.DataFrame,
    x: Array,
    s: Array,
) -> dict[str, pd.DataFrame]:
    """Combine variable sensitivities `s` (time, n, m) with output ones."""
    n_out = len(setup.outputs)
    s_out = np.zeros((len(values), n_out, len(wrt)))
    if n_out > 0:
        for i, t in enumerate(values.index):
            gx, gp = setup.derivs.outputs(float(t), x[i])
            s_out[i] = gx @ s[i] + gp @ setup.chain

    columns = setup.var_names + setup.outputs
    position = {k: i for i, k in enumerate(setup.var_names)}
    out_position = {k: i for i, k in enumerate(setup.outputs)}
    result = {}
    for j, name in enumerate(wrt):
        data = np.column_stack(
            [
                s[:, position[c], j] if c in position else s_out[:, out_position[c], j]
                for c in columns
            ]
        )
        result[name] = pd.DataFrame(data, index=values.index, columns=columns)
    return result


def _forward_time_course(
    model: _Model,
    time_points: Array,
    *,
    wrt: list[str],
    y0: dict[str, float],
    integrator: IntegratorType,
    setup: _Setup,
) -> Result[ForwardSensitivities]:
    n, m = len(setup.var_names), len(wrt)
    rhs = _AugmentedRhs(model, setup)
    z0 = np.concatenate(
        [np.array([y0[k] for k in setup.var_names], dtype=float), setup.s0.ravel()]
    )
    # Integrators accept arrays as rhs return values, not only tuples
    res = integrator(cast("Rhs", rhs), tuple(z0), rhs.jacobian).integrate_time_course(
        time_points=time_points
    )
    if isinstance(tc := res.value, Exception):
        return Result(tc)

    time = np.asarray(tc.time, dtype=float)
    z = np.asarray(tc.values, dtype=float)
    mask = _select_times(time, time_points)
    time, z = time[mask], z[mask]
    x = z[:, :n]
    s = z[:, n:].reshape(len(time), n, m)
    values = _values(model, pd.DataFrame(x, index=time, columns=setup.var_names))
    return Result(
        ForwardSensitivities(
            simulated=values,
            derivatives=_assemble(setup, wrt, values, x, s),
            method="forward",
        )
    )


def _forward_steady_state(
    model: _Model,
    wrt: list[str],
    y0: dict[str, float],
    integrator: IntegratorType,
    setup: _Setup,
    *,
    tolerance: float,
    rel_norm: bool,
) -> Result[ForwardSensitivities]:
    """Implicit function theorem: J dx*/dtheta = -df/dtheta.

    J is singular for models with conserved quantities, whose totals are
    fixed by the initial values instead: L dx*/dtheta = L dx0/dtheta for the
    left null space L of J.
    """
    res = (
        Simulator(model, y0=y0, integrator=integrator)
        .simulate_to_steady_state(tolerance=tolerance, rel_norm=rel_norm)
        .get_result()
    )
    if isinstance(sim := res.value, Exception):
        return Result(sim)
    variables = sim.get_variables(
        include_derived_variables=False,
        include_surrogate_variables=False,
        include_readouts=False,
    )
    variables = variables.loc[:, setup.var_names].iloc[[-1]]
    t = float(variables.index[0])
    x = variables.to_numpy(dtype=float)[0]

    jac, dfdp = setup.derivs.rhs(t, x)
    left_null = _left_null_space(jac)
    a = np.vstack([jac, left_null])
    b = np.vstack([-dfdp @ setup.chain, left_null @ setup.s0])
    s = np.linalg.lstsq(a, b, rcond=None)[0]

    values = _values(model, variables)
    return Result(
        ForwardSensitivities(
            simulated=values,
            derivatives=_assemble(setup, wrt, values, x[None, :], s[None, :, :]),
            method="forward",
        )
    )


###############################################################################
# Finite differences
###############################################################################


def _fd(
    model: _Model,
    wrt: list[str],
    y0: dict[str, float] | None,
    simulate: Callable[[dict[str, float]], Result[pd.DataFrame]],
    displacement: float,
) -> Result[ForwardSensitivities]:
    """Central differences of whole simulations.

    `simulate(y0)` returns the values for the current model parameters.
    """
    pars, _ = _check_names(model, wrt)

    def initial() -> dict[str, float]:
        return model.get_initial_conditions() if y0 is None else dict(y0)

    base = simulate(initial())
    if isinstance(values := base.value, Exception):
        return Result(values)

    result: dict[str, pd.DataFrame] = {}
    for name in wrt:
        if name in pars:

            def read() -> Array:
                res = simulate(initial())
                if isinstance(v := res.value, Exception):
                    raise v
                return v.to_numpy(dtype=float)

            try:
                diff = _parameter_fd(model, name, displacement, read)
            except Exception as e:  # noqa: BLE001
                return Result(e)
        else:
            x0 = initial()
            h = _step(x0[name], displacement)
            up = simulate(x0 | {name: x0[name] + h})
            down = simulate(x0 | {name: x0[name] - h})
            if isinstance(u := up.value, Exception):
                return Result(u)
            if isinstance(d := down.value, Exception):
                return Result(d)
            diff = (u.to_numpy(dtype=float) - d.to_numpy(dtype=float)) / (2 * h)
        result[name] = pd.DataFrame(diff, index=values.index, columns=values.columns)

    return Result(
        ForwardSensitivities(
            simulated=values, derivatives=result, method="finite_difference"
        )
    )


def _raw_variables(model: _Model, sim: object) -> pd.DataFrame:
    return sim.get_variables(  # type: ignore[attr-defined]
        include_derived_variables=False,
        include_surrogate_variables=False,
        include_readouts=False,
    ).loc[:, model.get_variable_names()]


###############################################################################
# Public API
###############################################################################


def _output_names(model: _Model) -> list[str]:
    """Derived quantities and reactions sensitivities can be computed for."""
    names = model.get_variable_names()
    y0 = model.get_initial_conditions()
    frame = pd.DataFrame([[y0[k] for k in names]], index=[0.0], columns=names)
    return [c for c in _values(model, frame).columns if c not in names]


def _unwrap(integrator: IntegratorType) -> tuple[object, dict[str, object]]:
    """The integrator class and the keywords of e.g. functools.partial(Native, rtol=...)."""
    base: object = integrator
    kwargs: dict[str, object] = {}
    while isinstance(base, functools.partial):
        kwargs = base.keywords | kwargs
        base = base.func
    return base, kwargs


def _base_integrator(integrator: IntegratorType) -> object:
    """The integrator class, unwrapping e.g. functools.partial(Native, rtol=...)."""
    return _unwrap(integrator)[0]


def _left_null_space(jac: Array) -> Array:
    """Rows spanning the left null space of the Jacobian (the conserved quantities)."""
    u, sv, _ = np.linalg.svd(jac)
    tol = max(jac.shape) * np.finfo(float).eps * (sv.max() if sv.size else 0.0)
    return u[:, sv <= max(tol, 1e-12)].T


def _unsupported(model: _Model, integrator: IntegratorType) -> str | None:
    """Why the integrator can't solve the augmented system, if it can't."""
    base = _base_integrator(integrator)
    name = getattr(base, "__name__", type(base).__name__)
    if getattr(base, "compiles_model", False):
        return f"the {name} integrator compiles the model itself"
    if getattr(model, "_events", None):
        return "the model has events"
    return None


def _resolve(
    model: _Model,
    wrt: list[str],
    outputs: list[str] | None,
    integrator: IntegratorType,
    method: ForwardMethod,
    *,
    y0_from_model: bool,
    displacement: float,
) -> tuple[_Setup | None, list[str]]:
    """Validate the inputs, then compile the forward sensitivities if possible.

    Returns no setup if finite differences are needed instead.
    """
    _check_names(model, wrt)
    available = _output_names(model)
    if outputs is None:
        outputs = available
    elif unknown := [k for k in outputs if k not in available]:
        msg = f"Unknown outputs {unknown}, choose from {available}"
        raise ValueError(msg)

    if method == "finite_difference":
        return None, outputs

    if (reason := _unsupported(model, integrator)) is not None:
        if method == "forward":
            msg = f"Forward sensitivities aren't supported, as {reason}"
            raise ValueError(msg)
        return None, outputs

    try:
        return _setup(
            model, wrt, outputs, y0_from_model=y0_from_model, displacement=displacement
        ), outputs
    except Exception as e:
        if method == "forward":
            raise
        _warn_once(
            f"Using finite-difference sensitivities, as the model can't be differentiated symbolically: {e}"
        )
        return None, outputs


def time_course(
    model: _Model,
    time_points: ArrayLike,
    wrt: list[str],
    *,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    outputs: list[str] | None = None,
    method: ForwardMethod = "auto",
    displacement: float = 1e-3,
) -> Result[ForwardSensitivities]:
    """Sensitivities of a time course.

    Examples
    --------
        >>> sens = forward_sensitivities.time_course(model, [1, 2, 5], ["k1", "x"])
        >>> sens.unwrap_or_err().derivatives["k1"]

    Parameters
    ----------
    model
        Model to simulate
    time_points
        Time points at which the sensitivities are returned
    wrt
        Parameters and variables (for their initial values) to differentiate w.r.t.
    y0
        Initial values. If None, they are taken from the model, so initial
        assignments depending on the parameters are differentiated, too
    integrator
        Integrator, defaults to the default integrator
    outputs
        Derived quantities and reactions to include. Defaults to all
    method
        `forward` integrates the sensitivity equations, `finite_difference`
        perturbs every parameter and simulates again, `auto` uses forward
        sensitivities unless the model can't be differentiated symbolically
    displacement
        Relative step for the finite differences. Too small steps amplify the
        integration error, ~ (integrator tolerance) ** (1/3) works well

    Returns
    -------
    Result[ForwardSensitivities]
        Values and their sensitivities at `time_points`

    """
    time_points = np.asarray(time_points, dtype=float)
    integrator = DefaultIntegrator if integrator is None else integrator
    setup, outputs = _resolve(
        model,
        wrt,
        outputs,
        integrator,
        method,
        y0_from_model=y0 is None,
        displacement=displacement,
    )
    columns = model.get_variable_names() + outputs

    if setup is not None:
        start = model.get_initial_conditions() if y0 is None else y0
        res = _forward_time_course(
            model,
            time_points,
            wrt=wrt,
            y0=start,
            integrator=integrator,
            setup=setup,
        )
        if isinstance(sens := res.value, Exception):
            return res
        sens.simulated = sens.simulated.loc[:, columns]
        sens.derivatives = {k: v.loc[:, columns] for k, v in sens.derivatives.items()}
        return res

    def simulate(start: dict[str, float]) -> Result[pd.DataFrame]:
        res = (
            Simulator(model, y0=start, integrator=integrator)
            .simulate_time_course(time_points)
            .get_result()
        )
        if isinstance(sim := res.value, Exception):
            return Result(sim)
        variables = _raw_variables(model, sim)
        mask = _select_times(variables.index.to_numpy(dtype=float), time_points)
        return Result(_values(model, variables.loc[mask]).loc[:, columns])

    return _fd(model, wrt, y0, simulate, displacement)


def steady_state(
    model: _Model,
    wrt: list[str],
    *,
    y0: dict[str, float] | None = None,
    integrator: IntegratorType | None = None,
    outputs: list[str] | None = None,
    method: ForwardMethod = "auto",
    displacement: float = 1e-3,
    tolerance: float = 1e-6,
    rel_norm: bool = False,
) -> Result[ForwardSensitivities]:
    """Sensitivities of the steady state.

    Examples
    --------
        >>> sens = forward_sensitivities.steady_state(model, ["k1", "k2"])
        >>> sens.unwrap_or_err().derivatives["k1"]

    Parameters
    ----------
    model
        Model to simulate
    wrt
        Parameters and variables (for their initial values) to differentiate
        w.r.t. Initial values only matter for conserved quantities
    y0
        Initial values. If None, they are taken from the model, so initial
        assignments depending on the parameters are differentiated, too
    integrator
        Integrator, defaults to the default integrator
    outputs
        Derived quantities and reactions to include. Defaults to all
    method
        `forward` uses the implicit function theorem, `finite_difference`
        perturbs every parameter and simulates again, `auto` uses forward
        sensitivities unless the model can't be differentiated symbolically
    displacement
        Relative step for the finite differences. Too small steps amplify the
        integration error, ~ (integrator tolerance) ** (1/3) works well
    tolerance
        Steady-state tolerance
    rel_norm
        Whether to use the relative norm for the steady-state check

    Returns
    -------
    Result[ForwardSensitivities]
        Steady-state values and their sensitivities (a single row)

    """
    integrator = DefaultIntegrator if integrator is None else integrator
    setup, outputs = _resolve(
        model,
        wrt,
        outputs,
        integrator,
        method,
        y0_from_model=y0 is None,
        displacement=displacement,
    )
    columns = model.get_variable_names() + outputs

    if setup is not None:
        start = model.get_initial_conditions() if y0 is None else y0
        res = _forward_steady_state(
            model,
            wrt,
            start,
            integrator,
            setup,
            tolerance=tolerance,
            rel_norm=rel_norm,
        )
        if isinstance(sens := res.value, Exception):
            return res
        sens.simulated = sens.simulated.loc[:, columns]
        sens.derivatives = {k: v.loc[:, columns] for k, v in sens.derivatives.items()}
        return res

    def simulate(start: dict[str, float]) -> Result[pd.DataFrame]:
        res = (
            Simulator(model, y0=start, integrator=integrator)
            .simulate_to_steady_state(tolerance=tolerance, rel_norm=rel_norm)
            .get_result()
        )
        if isinstance(sim := res.value, Exception):
            return Result(sim)
        variables = _raw_variables(model, sim).iloc[[-1]]
        return Result(_values(model, variables).loc[:, columns])

    return _fd(model, wrt, y0, simulate, displacement)
