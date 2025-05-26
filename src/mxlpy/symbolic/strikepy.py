# ruff: noqa: D100, D101, D102, D103, D104, D105, D106, D107, D200, D203, D400, D401

"""Reimplementation of strikepy from.

StrikePy: https://github.com/afvillaverde/StrikePy
STRIKE-GOLDD: https://github.com/afvillaverde/strike-goldd

FIXME:
- no handling of derived variables
- performance issues of generic_rank
"""

from __future__ import annotations

import textwrap
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from math import ceil, inf
from time import time
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import symbtools as st
import sympy
import sympy as sym
import tqdm
from sympy import Matrix
from sympy.matrices import zeros

if TYPE_CHECKING:
    from .symbolic_model import SymbolicModel

__all__ = [
    "Options",
    "Result",
    "StrikepyModel",
    "check_identifiability",
    "strike_goldd",
]


@dataclass
class Options:
    check_observability: bool = True
    max_lie_time: float = inf
    non_zero_known_input_derivatives: list[int] = field(default_factory=lambda: [100])
    non_zero_unknown_input_derivatives: list[int] = field(default_factory=lambda: [100])
    prev_ident_pars: set[sympy.Symbol] = field(default_factory=set)


@dataclass
class StrikepyModel:
    states: list[sym.Symbol]
    pars: list[sym.Symbol]
    eqs: list[sym.Expr]
    outputs: list[sym.Symbol]
    known_inputs: list[sym.Symbol] = field(default_factory=list)
    unknown_inputs: list[sym.Symbol] = field(default_factory=list)


@dataclass
class Result:
    rank: int
    model: StrikepyModel
    is_fispo: bool
    par_ident: list
    par_unident: list
    state_obs: list
    state_unobs: list
    input_obs: list
    input_unobs: list

    def all_inputs_observable(self) -> bool:
        return bool(
            len(self.par_ident) == len(self.model.pars)
            and len(self.model.unknown_inputs) > 0
        )

    def summary(self) -> str:
        return textwrap.dedent(f"""\
        Summary
        =======
        The model {"is" if self.is_fispo else "is not"} FISPO.
        Identifiable parameters: {self.par_ident}
        Unidentifiable parameters: {self.par_unident}
        Identifiable variables: {self.state_obs}
        Unidentifiable variables: {self.state_unobs}
        Identifiable inputs: {self.input_obs}
        Unidentifiable inputs: {self.input_unobs}
        """)


def _rationalize_all_numbers(expr: sym.Matrix) -> sym.Matrix:
    """Convert all numbers in expr to sympy.Rational-objects."""
    numbers_atoms = list(expr.atoms(sym.Number))
    rationalized_number_tpls = [(n, sym.Rational(n)) for n in numbers_atoms]
    return expr.subs(rationalized_number_tpls)


def _calculate_num_rank(inp: tuple[int, list[int]], onx: Matrix) -> tuple[int, int]:
    idx, indices = inp
    return idx, st.generic_rank(onx.col(indices))


def _elim_and_recalc(
    *,
    model: StrikepyModel,
    res: Result,
    options: Options,
    unmeas_xred_indices: list[int],
    onx: sym.Matrix,
    unidflag: bool,
    w1: list[sym.Symbol],
) -> None:
    onx = _rationalize_all_numbers(onx)
    par_ident = res.par_ident
    state_obs = res.state_obs
    input_obs = res.input_obs

    r = sym.shape(onx)[1]
    new_ident_pars = par_ident
    new_nonid_pars = []
    new_obs_states = state_obs
    new_unobs_states = []
    new_obs_in = input_obs
    new_unobs_in = []

    all_indices: list[tuple[int, list[int]]] = []
    for idx in range(len(model.pars)):
        if model.pars[idx] not in par_ident:
            indices = list(range(r))
            indices.pop(len(model.states) + idx)
            all_indices.append((idx, indices))
    with ProcessPoolExecutor() as ppe:
        num_ranks = list(ppe.map(partial(_calculate_num_rank, onx=onx), all_indices))
    for idx, num_rank in num_ranks:
        if num_rank == res.rank:
            if unidflag:
                new_nonid_pars.append(model.pars[idx])
        else:
            new_ident_pars.append(model.pars[idx])

    # At each iteration we try removing a different state from 'xred':
    if options.check_observability:
        all_indices = []
        for idx in range(len(unmeas_xred_indices)):
            orig_idx = unmeas_xred_indices[idx]
            if model.states[orig_idx] not in state_obs:
                indices = list(range(r))
                indices.pop(orig_idx)
                all_indices.append((orig_idx, indices))
        with ProcessPoolExecutor() as ppe:
            num_ranks = list(
                ppe.map(partial(_calculate_num_rank, onx=onx), all_indices)
            )
        for orig_idx, num_rank in num_ranks:
            if num_rank == res.rank:
                if unidflag:
                    new_unobs_states.append(model.states[orig_idx])
            else:
                new_obs_states.append(model.states[orig_idx])

    # At each iteration we try removing a different column from onx:
    all_indices = []
    for idx in range(len(w1)):
        if w1[idx] not in input_obs:
            indices = list(range(r))
            indices.pop(len(model.states) + len(model.pars) + idx)
            all_indices.append((idx, indices))
    with ProcessPoolExecutor() as ppe:
        num_ranks = list(ppe.map(partial(_calculate_num_rank, onx=onx), all_indices))
    for idx, num_rank in num_ranks:
        if num_rank == res.rank:
            if unidflag:
                new_unobs_in.append(w1[idx])
        else:
            new_obs_in.append(w1[idx])

    res.par_ident = new_ident_pars
    res.par_unident = new_nonid_pars
    res.state_obs = new_obs_states
    res.state_unobs = new_unobs_states
    res.input_obs = new_obs_in
    res.input_unobs = new_unobs_in


def _remove_identified_parameters(model: StrikepyModel, options: Options) -> None:
    if len(options.prev_ident_pars) != 0:
        model.pars = [i for i in model.pars if i not in options.prev_ident_pars]


def _get_measured_states(model: StrikepyModel) -> tuple[list[sym.Symbol], list[int]]:
    # Check which states are directly measured, if any.
    # Basically it is checked if any state is directly on the output,
    # then that state is directly measurable.
    is_measured: list[bool] = [False for i in range(len(model.states))]
    for i, state in enumerate(model.states):
        if state in model.outputs:
            is_measured[i] = True

    measured_state_idxs: list[int] = [i for i, j in enumerate(is_measured) if j]
    unmeasured_state_idxs = [i for i, j in enumerate(is_measured) if not j]
    measured_state_names = [model.states[i] for i in measured_state_idxs]
    return measured_state_names, unmeasured_state_idxs


def _create_derivatives(
    elements: list[sym.Symbol], n_min_lie_derivatives: int, n_derivatives: list[int]
) -> list[list[sym.Symbol]]:
    derivatives: list[list[float | sym.Symbol]] = []
    for ind_u, element in enumerate(elements):
        auxiliar: list[float | sym.Symbol] = [sym.Symbol(f"{element}")]
        for k in range(n_min_lie_derivatives):
            auxiliar.append(sym.Symbol(f"{element}_d{k + 1}"))  # noqa: PERF401
        derivatives.append(auxiliar)

        if len(derivatives[0]) >= n_derivatives[ind_u] + 1:
            for i in range(len(derivatives[0][(n_derivatives[ind_u] + 1) :])):
                derivatives[ind_u][(n_derivatives[ind_u] + 1) + i] = 0
    return derivatives  # type: ignore


def _create_w1_vector(
    model: StrikepyModel, w_der: list[list[sym.Symbol]]
) -> tuple[list[sym.Symbol], list[sym.Symbol]]:
    w1vector = []
    w1vector_dot = []

    if len(model.unknown_inputs) == 0:
        return w1vector, w1vector_dot

    w1vector.extend(w_der[:-1])
    w1vector_dot.extend(w_der[1:])

    # -- Include as states only nonzero inputs / derivatives:
    nzi = []
    nzj = []
    nz_w1vec = []
    for fila in range(len(w1vector)):
        if w1vector[fila][0] != 0:
            nzi.append([fila])
            nzj.append([1])
            nz_w1vec.append(w1vector[fila])

    w1vector = nz_w1vec
    w1vector_dot = w1vector_dot[0 : len(nzi)]
    return w1vector, w1vector_dot


def _create_xaug_faug(
    model: StrikepyModel,
    w1vector: list[sym.Symbol],
    w1vector_dot: list[sym.Symbol],
) -> tuple[npt.NDArray, npt.NDArray]:
    xaug = np.array(model.states)
    xaug = np.append(xaug, model.pars, axis=0)  # type: ignore
    if len(w1vector) != 0:
        xaug = np.append(xaug, w1vector, axis=0)  # type: ignore

    faug = np.atleast_2d(np.array(model.eqs, dtype=object)).T
    faug = np.append(faug, zeros(len(model.pars), 1), axis=0)
    if len(w1vector) != 0:
        faug = np.append(faug, w1vector_dot, axis=0)  # type: ignore
    return xaug, faug


def _compute_extra_term(
    extra_term: npt.NDArray,
    ind: int,
    past_lie: sym.Matrix,
    input_der: list[list[sym.Symbol]],
    zero_input_der_dummy_name: sym.Symbol,
) -> npt.NDArray:
    for i in range(ind):
        column = len(input_der) - 1
        if i < column:
            lo_u_der = input_der[i]
            if lo_u_der == 0:
                lo_u_der = zero_input_der_dummy_name
            lo_u_der = np.array([lo_u_der])
            hi_u_der = input_der[i + 1]
            hi_u_der = Matrix([hi_u_der])
            intermedio = past_lie.jacobian(lo_u_der) * hi_u_der
            extra_term = extra_term + intermedio if extra_term else intermedio
    return extra_term


def _compute_n_min_lie_derivatives(model: StrikepyModel) -> int:
    n_outputs = len(model.outputs)
    n_states = len(model.states)
    n_unknown_pars = len(model.pars)
    n_unknown_inp = len(model.unknown_inputs)
    n_vars_to_observe = n_states + n_unknown_pars + n_unknown_inp
    return ceil((n_vars_to_observe - n_outputs) / n_outputs)


def _test_fispo(
    model: StrikepyModel,
    res: Result,
    measured_state_names: list[sym.Symbol],
) -> bool:
    if len(res.par_ident) == len(model.pars) and (
        len(res.state_obs) + len(measured_state_names)
    ) == len(model.states):
        res.is_fispo = True
        res.state_obs = model.states
        res.input_obs = model.unknown_inputs
        res.par_ident = model.pars

    return res.is_fispo


def _create_onx(
    model: StrikepyModel,
    n_min_lie_derivatives: int,
    options: Options,
    w1vector: list[sym.Symbol],
    xaug: npt.ArrayLike,
    faug: npt.ArrayLike,
    input_der: list[list[sym.Symbol]],
    zero_input_der_dummy_name: sym.Symbol,
) -> tuple[npt.NDArray, sym.Matrix]:
    onx = np.array(
        zeros(
            len(model.outputs) * (1 + n_min_lie_derivatives),
            len(model.states) + len(model.pars) + len(w1vector),
        )
    )
    jacobian = sym.Matrix(model.outputs).jacobian(xaug)

    # first row(s) of onx (derivative of the output with respect to the vector states+unknown parameters).
    onx[0 : len(model.outputs)] = np.array(jacobian)

    past_Lie = sym.Matrix(model.outputs)
    extra_term = np.array(0)

    # loop as long as I don't complete the preset Lie derivatives or go over the maximum time
    t_start = time()

    onx[(len(model.outputs)) : 2 * len(model.outputs)] = past_Lie.jacobian(xaug)
    for ind in range(1, n_min_lie_derivatives):
        if (time() - t_start) > options.max_lie_time:
            msg = "More Lie derivatives would be needed to analyse the model."
            raise TimeoutError(msg)

        lie_derivatives = Matrix(
            (onx[(ind * len(model.outputs)) : (ind + 1) * len(model.outputs)][:]).dot(
                faug
            )
        )
        extra_term = _compute_extra_term(
            extra_term,
            ind=ind,
            past_lie=past_Lie,
            input_der=input_der,
            zero_input_der_dummy_name=zero_input_der_dummy_name,
        )

        ext_Lie = lie_derivatives + extra_term if extra_term else lie_derivatives
        past_Lie = ext_Lie
        onx[((ind + 1) * len(model.outputs)) : (ind + 2) * len(model.outputs)] = (
            sym.Matrix(ext_Lie).jacobian(xaug)
        )
    return onx, cast(sym.Matrix, past_Lie)


def strike_goldd(model: StrikepyModel, options: Options | None = None) -> Result:
    options = Options() if options is None else options

    # Check if the size of nnzDerU and nnzDerW are appropriate
    if len(model.known_inputs) > len(options.non_zero_known_input_derivatives):
        msg = (
            "The number of known inputs is higher than the size of nnzDerU "
            "and must have the same size."
        )
        raise ValueError(msg)
    if len(model.unknown_inputs) > len(options.non_zero_unknown_input_derivatives):
        msg = (
            "The number of unknown inputs is higher than the size of nnzDerW "
            "and must have the same size."
        )
        raise ValueError(msg)

    _remove_identified_parameters(model, options)

    res = Result(
        rank=0,
        is_fispo=False,
        model=model,
        par_ident=[],
        par_unident=[],
        state_obs=[],
        state_unobs=[],
        input_obs=[],
        input_unobs=[],
    )

    lastrank = None
    unidflag = False
    skip_elim: bool = False

    n_min_lie_derivatives = _compute_n_min_lie_derivatives(model)
    measured_state_names, unmeasured_state_idxs = _get_measured_states(model)

    input_der = _create_derivatives(
        model.known_inputs,
        n_min_lie_derivatives=n_min_lie_derivatives,
        n_derivatives=options.non_zero_known_input_derivatives,
    )
    zero_input_der_dummy_name = sym.Symbol("zero_input_der_dummy_name")

    w_der: list[list[sym.Symbol]] = _create_derivatives(
        model.unknown_inputs,
        n_min_lie_derivatives=n_min_lie_derivatives,
        n_derivatives=options.non_zero_unknown_input_derivatives,
    )

    w1vector, w1vector_dot = _create_w1_vector(
        model,
        w_der=w_der,
    )

    xaug, faug = _create_xaug_faug(
        model,
        w1vector=w1vector,
        w1vector_dot=w1vector_dot,
    )

    onx, past_Lie = _create_onx(
        model,
        n_min_lie_derivatives=n_min_lie_derivatives,
        options=options,
        w1vector=w1vector,
        xaug=xaug,
        faug=faug,
        input_der=input_der,
        zero_input_der_dummy_name=zero_input_der_dummy_name,
    )

    t_start = time()

    pbar = tqdm.tqdm(desc="Main loop")
    while True:
        pbar.update(1)
        if time() - t_start > options.max_lie_time:
            msg = "More Lie derivatives would be needed to see if the model is structurally unidentifiable as a whole."
            raise TimeoutError(msg)

        # FIXME: For some problems this starts to be really slow
        # can't directly be fixed by using numpy.linalg.matrix_rank because
        # that can't handle the symbolic stuff
        res.rank = st.generic_rank(_rationalize_all_numbers(Matrix(onx)))

        # If the onx matrix already has full rank... all is observable and identifiable
        if res.rank == len(xaug):
            res.state_obs = model.states
            res.input_obs = model.unknown_inputs
            res.par_ident = model.pars
            break

        # If there are unknown inputs, we may want to check id/obs of (x,p,w) and not of dw/dt:
        if len(model.unknown_inputs) > 0:
            _elim_and_recalc(
                model=model,
                res=res,
                options=options,
                unmeas_xred_indices=unmeasured_state_idxs,
                onx=Matrix(onx),
                unidflag=unidflag,
                w1=w1vector,
            )

            if _test_fispo(
                model=model, res=res, measured_state_names=measured_state_names
            ):
                break

        # If possible (& necessary), calculate one more Lie derivative and retry:
        if n_min_lie_derivatives < len(xaug) and res.rank != lastrank:
            ind = n_min_lie_derivatives
            n_min_lie_derivatives = (
                n_min_lie_derivatives + 1
            )  # One is added to the number of derivatives already made
            extra_term = np.array(0)  # reset for each new Lie derivative
            # - Known input derivatives: ----------------------------------
            # Extra terms of extended Lie derivatives
            # may have to add extra input derivatives (note that 'nd' has grown):
            if len(model.known_inputs) > 0:
                input_der = _create_derivatives(
                    model.known_inputs,
                    n_min_lie_derivatives=n_min_lie_derivatives,
                    n_derivatives=options.non_zero_known_input_derivatives,
                )
                extra_term = _compute_extra_term(
                    extra_term,
                    ind=ind,
                    past_lie=past_Lie,
                    input_der=input_der,
                    zero_input_der_dummy_name=zero_input_der_dummy_name,
                )

            # add new derivatives, if they are not zero
            if len(model.unknown_inputs) > 0:
                prev_size = len(w1vector)
                w_der = _create_derivatives(
                    model.unknown_inputs,
                    n_min_lie_derivatives=n_min_lie_derivatives + 1,
                    n_derivatives=options.non_zero_unknown_input_derivatives,
                )
                w1vector, w1vector_dot = _create_w1_vector(
                    model,
                    w_der=w_der,
                )
                xaug, faug = _create_xaug_faug(
                    model, w1vector=w1vector, w1vector_dot=w1vector_dot
                )

                # Augment size of the Obs-Id matrix if needed
                new_size = len(w1vector)
                onx = np.append(
                    onx,
                    zeros((ind + 1) * len(model.outputs), new_size - prev_size),
                    axis=1,
                )

            newLie = Matrix(
                (
                    onx[(ind * len(model.outputs)) : (ind + 1) * len(model.outputs)][:]  # type: ignore
                ).dot(faug)  # type: ignore
            )
            past_Lie = newLie + extra_term if extra_term else newLie
            newOnx = sym.Matrix(past_Lie).jacobian(xaug)
            onx = np.append(onx, newOnx, axis=0)
            lastrank = res.rank

        # If that is not possible, there are several possible causes:
        # This is the case when you have onx with all possible derivatives done
        # and it is not full rank, the maximum time for the next derivative has passed
        # or the matrix no longer increases in rank as derivatives are increased.
        else:
            # The maximum number of Lie derivatives has been reached
            if n_min_lie_derivatives >= len(xaug):
                unidflag = True
            elif res.rank == lastrank:
                onx = onx[0 : (-1 - (len(model.outputs) - 1))]  # type: ignore
                # It is indicated that the number of derivatives needed was
                # one less than the number of derivatives made
                n_min_lie_derivatives = n_min_lie_derivatives - 1
                unidflag = True

            if not skip_elim and not res.is_fispo:
                # Eliminate columns one by one to check identifiability
                # of the associated parameters
                _elim_and_recalc(
                    model=model,
                    res=res,
                    options=options,
                    unmeas_xred_indices=unmeasured_state_idxs,
                    onx=Matrix(onx),
                    unidflag=unidflag,
                    w1=w1vector,
                )

                if _test_fispo(
                    model=model, res=res, measured_state_names=measured_state_names
                ):
                    break

                break
    pbar.close()
    return res


def check_identifiability(model: SymbolicModel, outputs: list[sympy.Symbol]) -> Result:
    strike_model = StrikepyModel(
        states=list(model.variables.values()),
        pars=list(model.parameters.values()),
        eqs=model.eqs,
        outputs=outputs,
    )
    return strike_goldd(strike_model)
