# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Hidden-state blocks: latent states driven by a small neural network.

A block owns `n_hidden` states `h` with

    dh/dt = nn_h(x / s, h)              # dense layers, linear output layer
    target_k  <-  mechanism(base, (W h)_k)

A block without hidden states (`n_hidden = 0`, mxlweb's `nn_blocks`) is a
Markovian correction: the network's outputs act on the targets directly,
`target_k <- mechanism(base, scale * nn(x / s)_k)`.

`nn_h` is lowered into named equations: one internal node per hidden unit,
which only refers to the nodes of the previous layer by name. A unit is
thus never expanded into each of its dependents, so the size of the
equations grows with the number of weights, not exponentially with depth.
`dh_i/dt` is an additive term on `h_i`, and each target gets one term with the
block's mechanism, so blocks reuse all of the term machinery (`mxlpy._terms`).

The weights are not parameters. They are stored as arrays, `(out, in)`, and
addressed by name (mxlweb-core's convention, identifier-safe):
`{block}_w{l}_{i}_{j}` / `{block}_b{l}_{i}` for layer `l` of `nn_h`,
`{block}_W_{k}_{i}` for the readout of target `k`. The states are
`{block}_h{i}`, their initial values `h(0)` the block's `h0`.
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np
import sympy

from mxlpy.meta.sympy_tools import expr_to_fn
from mxlpy.types import Derived, InitialAssignment, Mechanism, StructureToken, Term

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from mxlpy.types import Array, MechanismLike

__all__ = [
    "ACTIVATIONS",
    "HiddenStates",
    "HiddenStatesMixin",
    "LoweredHiddenStates",
    "NNLayer",
    "WEIGHT_WARNING_THRESHOLD",
]

_LOGGER = logging.getLogger(__name__)

WEIGHT_WARNING_THRESHOLD = 300

type Activation = Literal["tanh", "softplus", "sigmoid", "relu"]


def _softplus(x: sympy.Expr) -> sympy.Expr:
    """`log(1 + exp(x))` in its overflow-free form (as mxlweb-core)."""
    return cast(
        sympy.Expr,
        sympy.Max(x, 0) + sympy.log(1 + sympy.exp(-sympy.Abs(x))),
    )


ACTIVATIONS: dict[str, Callable[[sympy.Expr], sympy.Expr]] = {
    "tanh": lambda x: cast(sympy.Expr, sympy.tanh(x)),
    "softplus": _softplus,
    "sigmoid": lambda x: cast(sympy.Expr, 1 / (1 + sympy.exp(-x))),
    "relu": lambda x: cast(sympy.Expr, sympy.Max(x, 0)),
}


@dataclass(frozen=True, slots=True)
class NNLayer:
    """A dense layer of a block's network.

    Attributes
    ----------
    width
        Number of units
    activation
        Name of the elementwise activation (see `ACTIVATIONS`), None for linear
    expression
        The activation over the symbol `x`, for activations not in
        `ACTIVATIONS` (e.g. read from a file)

    """

    width: int
    activation: str | None = None
    expression: sympy.Expr | None = None

    def apply(self, x: sympy.Expr) -> sympy.Expr:
        """The activation of `x`."""
        if self.activation is None:
            return x
        if self.expression is not None:
            return cast(sympy.Expr, self.expression.xreplace({sympy.Symbol("x"): x}))
        return ACTIVATIONS[self.activation](x)


@dataclass(slots=True)
class LoweredHiddenStates:
    """A block as states, internal nodes, terms and weights.

    Weights are referenced by name, so the lowered form only depends on the
    block's architecture, not on its weight values.
    """

    states: list[str]
    nodes: dict[str, Derived]
    terms: dict[str, Term]


@dataclass(kw_only=True, slots=True)
class HiddenStates:
    """A hidden-state block, see the module docstring.

    Attributes
    ----------
    n_hidden
        Number of hidden states, 0 for a Markovian correction
    layers
        Layers of the network, the last one is its output layer (`n_hidden`
        units, or one per target without hidden states)
    inputs
        Model names `nn_h` reads besides the hidden states
    input_scales
        Constants the inputs are divided by
    output_scales
        Constants each target's term is multiplied by, e.g. a characteristic
        rate, so the weights can stay dimensionless
    targets
        Variables or reactions corrected by the block, one readout row each
    mechanism
        How each readout combines with its target
    weights
        Weight matrices of `nn_h`'s layers, `(out, in)`
    biases
        Biases of `nn_h`'s layers
    readout
        `W`, `(n_targets, n_hidden)`
    h0
        Initial values of the hidden states
    h0_assignments
        Initial assignments replacing `h0` by state index, e.g. to fit one
        `h(0)` per data set through a parameter each
    scale
        Parameter multiplying every target's term, None for none
    seed
        Seed of the initial weights

    """

    n_hidden: int
    layers: list[NNLayer]
    inputs: list[str]
    input_scales: Array
    targets: list[str]
    output_scales: Array
    mechanism: Mechanism
    weights: list[Array]
    biases: list[Array]
    readout: Array
    h0: Array
    h0_assignments: dict[int, InitialAssignment] = field(default_factory=dict)
    scale: str | None = None
    seed: int = 0
    _lowered: tuple[str, LoweredHiddenStates] | None = field(
        default=None, repr=False, compare=False
    )

    def __getstate__(self) -> dict[str, Any]:
        """State without the lowered equations, whose functions are generated."""
        return {
            k: getattr(self, k) for k in self.__dataclass_fields__ if k != "_lowered"
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the state, the equations are lowered again on use."""
        for k, v in state.items():
            object.__setattr__(self, k, v)
        self._lowered = None

    def state_names(self, name: str) -> list[str]:
        """Names of the hidden states."""
        return [f"{name}_h{i}" for i in range(self.n_hidden)]

    def term_names(self, name: str) -> list[str]:
        """Names of the terms: `dh/dt` of every state, then every target."""
        return [f"{name}_dh{i}" for i in range(self.n_hidden)] + [
            f"{name}_{t}" for t in self.targets
        ]

    def node_names(self, name: str) -> list[str]:
        """Names of the units of all layers but the output layer."""
        return [
            f"{name}_a{i}_{j}"
            for i, layer in enumerate(self.layers[:-1])
            for j in range(layer.width)
        ]

    def _weight_slots(self, name: str) -> dict[str, tuple[Array, tuple[int, ...]]]:
        slots: dict[str, tuple[Array, tuple[int, ...]]] = {}
        for layer, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            for i in range(w.shape[0]):
                slots[f"{name}_b{layer}_{i}"] = (b, (i,))
                for j in range(w.shape[1]):
                    slots[f"{name}_w{layer}_{i}_{j}"] = (w, (i, j))
        for k in range(self.readout.shape[0]):
            for i in range(self.readout.shape[1]):
                slots[f"{name}_W_{k}_{i}"] = (self.readout, (k, i))
        return slots

    def weight_names(self, name: str) -> list[str]:
        """Names of all weights, `nn_h` first, then the readout."""
        return list(self._weight_slots(name))

    def nn_weight_names(self, name: str) -> list[str]:
        """Names of the weights and biases of `nn_h`."""
        n = sum(w.size + b.size for w, b in zip(self.weights, self.biases, strict=True))
        return self.weight_names(name)[:n]

    def readout_names(self, name: str) -> list[str]:
        """Names of the readout weights `W`, row by row."""
        return [
            f"{name}_W_{k}_{i}"
            for k in range(self.readout.shape[0])
            for i in range(self.readout.shape[1])
        ]

    def weight_values(self, name: str) -> dict[str, float]:
        """Values of all weights by name."""
        return {k: float(a[idx]) for k, (a, idx) in self._weight_slots(name).items()}

    def set_weights(self, name: str, values: Mapping[str, float]) -> None:
        """Set weights by name."""
        slots = self._weight_slots(name)
        for k, v in values.items():
            array, idx = slots[k]
            array[idx] = v

    @property
    def n_weights(self) -> int:
        """Number of weights, including biases and the readout."""
        return self.readout.size + sum(
            w.size + b.size for w, b in zip(self.weights, self.biases, strict=True)
        )

    def lower(self, name: str) -> LoweredHiddenStates:
        """States, internal nodes and terms of the block (cached)."""
        if self._lowered is not None and self._lowered[0] == name:
            return self._lowered[1]
        states = self.state_names(name)
        prev: list[sympy.Expr] = [
            sympy.Symbol(x) / sympy.Float(s)
            for x, s in zip(self.inputs, self.input_scales, strict=True)
        ]
        prev.extend(sympy.Symbol(h) for h in states)

        def unit(i: int, j: int) -> sympy.Expr:
            n_in = self.weights[i].shape[1]
            pre = sympy.Add(
                sympy.Symbol(f"{name}_b{i}_{j}"),
                *(sympy.Symbol(f"{name}_w{i}_{j}_{k}") * prev[k] for k in range(n_in)),
            )
            return self.layers[i].apply(cast(sympy.Expr, pre))

        nodes: dict[str, Derived] = {}
        for i, layer in enumerate(self.layers[:-1]):
            names = [f"{name}_a{i}_{j}" for j in range(layer.width)]
            for j, node in enumerate(names):
                fn, args = expr_to_fn(unit(i, j), origin=node)
                nodes[node] = Derived(fn=fn, args=args)
            prev = [sympy.Symbol(n) for n in names]

        out = len(self.layers) - 1
        outputs = [unit(out, j) for j in range(self.layers[-1].width)]
        additive = Mechanism.of("additive")
        terms: dict[str, Term] = {}
        for j, h in enumerate(states):
            fn, args = expr_to_fn(outputs[j], origin=f"{name}_dh{j}")
            terms[f"{name}_dh{j}"] = Term(
                fn=fn, args=args, target=h, mechanism=additive
            )
        scale = sympy.Integer(1) if self.scale is None else sympy.Symbol(self.scale)
        for k, target in enumerate(self.targets):
            out_scale = float(self.output_scales[k])
            if states:
                readout = sympy.Add(
                    *(
                        sympy.Symbol(f"{name}_W_{k}_{j}") * sympy.Symbol(h)
                        for j, h in enumerate(states)
                    )
                )
            else:
                readout = outputs[k]
            if out_scale != 1:
                readout = sympy.Float(out_scale) * readout
            fn, args = expr_to_fn(
                cast(sympy.Expr, scale * readout), origin=f"{name}_{target}"
            )
            terms[f"{name}_{target}"] = Term(
                fn=fn, args=args, target=target, mechanism=self.mechanism
            )
        lowered = LoweredHiddenStates(states=states, nodes=nodes, terms=terms)
        self._lowered = (name, lowered)
        return lowered


def _glorot(
    rng: np.random.Generator, fan_out: int, fan_in: int, scale: float = 1.0
) -> Array:
    limit = math.sqrt(6 / (fan_in + fan_out))
    return scale * rng.uniform(-limit, limit, size=(fan_out, fan_in))


def init_hidden_states(
    *,
    n_hidden: int,
    layers: list[NNLayer],
    inputs: list[str],
    input_scales: Array,
    targets: list[str],
    mechanism: Mechanism,
    init_scale: float,
    seed: int,
) -> HiddenStates:
    """Glorot-initialised block.

    The readout is `init_scale` times Glorot: small, so the block starts as a
    small correction, but not exactly zero, which would give the weights of
    `nn_h` no gradient. The biases of `nn_h` are random as well, so `h` is not
    identically zero even where the inputs are.
    """
    rng = np.random.default_rng(seed)
    sizes = [len(inputs) + n_hidden, *(layer.width for layer in layers)]
    weights = [_glorot(rng, o, i) for i, o in itertools.pairwise(sizes)]
    biases: list[Array] = [rng.uniform(-0.1, 0.1, size=o) for o in sizes[1:]]
    return HiddenStates(
        n_hidden=n_hidden,
        layers=layers,
        inputs=inputs,
        input_scales=input_scales,
        targets=targets,
        output_scales=np.ones(len(targets)),
        mechanism=mechanism,
        weights=weights,
        biases=biases,
        readout=_glorot(rng, len(targets), n_hidden, init_scale),
        h0=np.zeros(n_hidden),
        seed=seed,
    )


def _check_scales(name: str, scales: Mapping[str, float]) -> None:
    if bad := sorted(k for k, v in scales.items() if not (float(v) > 0)):
        msg = (
            f"Hidden states {name!r}: input scales must be positive, not those of {bad}"
        )
        raise ValueError(msg)


class HiddenStatesMixin:
    """Hidden-state blocks and addressable names, shared by the builders."""

    __slots__ = ()

    if TYPE_CHECKING:
        _hidden_states: dict[str, HiddenStates]
        _terms: dict[str, Term]
        _cache: Any
        _structure: StructureToken
        _ids: dict[str, str]

        def _insert_id(self, *, name: str, ctx: str) -> None: ...
        def _remove_id(self, *, name: str) -> None: ...
        def _check_term_target(self, name: str, target: str) -> None: ...
        def _user_variable_names(self) -> list[str]: ...
        def get_parameter_names(self) -> list[str]: ...
        def get_parameter_values(self) -> dict[str, float]: ...
        def _invalidate(self) -> None: ...
        def update_parameters(self, parameters: Mapping[str, Any]) -> Self: ...

    ###########################################################################
    # Blocks
    ###########################################################################

    def add_hidden_states(
        self,
        name: str,
        *,
        n_hidden: int,
        targets: list[str],
        mechanism: MechanismLike | Mechanism,
        layers: Iterable[int] = (8,),
        activation: Activation = "tanh",
        inputs: list[str] | None = None,
        input_scales: Mapping[str, float] | None = None,
        init_scale: float = 1e-2,
        seed: int = 0,
    ) -> Self:
        """Add latent states driven by a small neural network.

        `dh/dt = nn_h(x / s, h)` and each target is corrected by the term
        `(W h)_k` with `mechanism`. The hidden states give the model memory:
        delays, unmodelled slow processes and history dependence, which a
        correction depending only on the current state can't provide.

        The weights are not parameters: `get_weight_names()` lists them,
        `get_addressable_names()` lists parameters and weights for fits,
        sensitivities and the simulator. `remove_hidden_states` restores the
        mechanistic model exactly.

        Examples
        --------
            >>> model.add_hidden_states(
            ...     "aug", n_hidden=2, targets=["S", "P"], mechanism="relative"
            ... )

        Parameters
        ----------
        name
            Name of the block, prefix of all its names
        n_hidden
            Number of hidden states
        targets
            Variables or (kinetic models) reactions the block corrects
        mechanism
            How each readout combines with its target, required:
            `"additive"`, `"relative"` or a callable `(base, term) -> value`
        layers
            Widths of the hidden layers of `nn_h`
        activation
            Activation of the hidden layers, `"tanh"`, `"softplus"`,
            `"sigmoid"` or `"relu"`. The output layer is linear.
        inputs
            Names `nn_h` reads besides the hidden states, all variables by
            default
        input_scales
            Constants to divide inputs by, 1 by default. Inputs of very
            different magnitudes make the network hard to train.
        init_scale
            Scale of the initial readout `W` (relative to Glorot). Small,
            but not zero, see `mxlpy._hidden_states.init_hidden_states`.
        seed
            Seed of the initial weights

        Returns
        -------
        Self
            The instance of the model with the added block.

        """
        if n_hidden < 1:
            msg = f"Hidden states {name!r}: n_hidden must be at least 1"
            raise ValueError(msg)
        widths = [int(i) for i in layers]
        if any(i < 1 for i in widths):
            msg = f"Hidden states {name!r}: layer widths must be at least 1"
            raise ValueError(msg)
        if activation not in ACTIVATIONS:
            msg = f"Hidden states {name!r}: unknown activation {activation!r}, use one of {sorted(ACTIVATIONS)}"
            raise ValueError(msg)
        if not targets:
            msg = f"Hidden states {name!r}: at least one target is required"
            raise ValueError(msg)
        if len(set(targets)) != len(targets):
            msg = f"Hidden states {name!r}: duplicate targets {targets}"
            raise ValueError(msg)
        for target in targets:
            self._check_term_target(name, target)
        inputs = self._user_variable_names() if inputs is None else list(inputs)
        if unknown := [i for i in inputs if i not in self._ids and i != "time"]:
            msg = f"Hidden states {name!r}: unknown inputs {unknown}"
            raise KeyError(msg)
        scales = {} if input_scales is None else dict(input_scales)
        if unknown := sorted(set(scales) - set(inputs)):
            msg = f"Hidden states {name!r}: scales for unknown inputs {unknown}"
            raise KeyError(msg)
        _check_scales(name, scales)
        block = init_hidden_states(
            n_hidden=n_hidden,
            layers=[NNLayer(w, activation) for w in widths] + [NNLayer(n_hidden)],
            inputs=inputs,
            input_scales=np.array([float(scales.get(i, 1.0)) for i in inputs]),
            targets=list(targets),
            mechanism=Mechanism.of(mechanism),
            init_scale=init_scale,
            seed=seed,
        )
        if block.n_weights > WEIGHT_WARNING_THRESHOLD:
            _LOGGER.warning(
                "Hidden states %r have %d weights. Their equations are compiled "
                "symbolically, consider a smaller network.",
                name,
                block.n_weights,
            )
        return self._add_block(name, block)

    def _add_block(self, name: str, block: HiddenStates) -> Self:
        """Add a block as is, e.g. a copy or one read from a file."""
        if len(set(block.targets)) != len(block.targets):
            msg = f"Hidden states {name!r}: duplicate targets {block.targets}"
            raise ValueError(msg)
        for target in block.targets:
            self._check_term_target(name, target)
        if unknown := [i for i in block.inputs if i not in self._ids and i != "time"]:
            msg = f"Hidden states {name!r}: unknown inputs {unknown}"
            raise KeyError(msg)
        _check_scales(name, dict(zip(block.inputs, block.input_scales, strict=True)))
        _check_scales(name, dict(zip(block.targets, block.output_scales, strict=True)))
        self._insert_block_ids(name, block)
        self._invalidate()
        self._hidden_states[name] = block
        return self

    def _block_owned_names(self, name: str, block: HiddenStates) -> list[str]:
        return [
            name,
            *block.state_names(name),
            *block.node_names(name),
            *block.term_names(name),
            *block.weight_names(name),
        ]

    def _insert_block_ids(self, name: str, block: HiddenStates) -> None:
        inserted: list[str] = []
        try:
            for owned in self._block_owned_names(name, block):
                self._insert_id(name=owned, ctx="hidden_states")
                inserted.append(owned)
        except (KeyError, NameError):
            for owned in inserted:
                self._remove_id(name=owned)
            raise

    def remove_hidden_states(self, name: str) -> Self:
        """Remove a hidden-state block, restoring the model without it exactly.

        Parameters
        ----------
        name
            Name of the block

        Returns
        -------
        Self
            The instance of the model with the block removed.

        """
        block = self._hidden_states.pop(name)
        for owned in self._block_owned_names(name, block):
            self._remove_id(name=owned)
        self._invalidate()
        return self

    def get_raw_hidden_states(self, *, as_copy: bool = True) -> dict[str, HiddenStates]:
        """Get the hidden-state blocks.

        Parameters
        ----------
        as_copy
            If True, return a deep copy of the blocks.
            If False, return the internal dictionary directly.

        """
        if as_copy:
            return copy.deepcopy(self._hidden_states)
        return self._hidden_states

    def get_hidden_state_names(self) -> list[str]:
        """Names of the states owned by hidden-state blocks."""
        return [h for k, v in self._hidden_states.items() for h in v.state_names(k)]

    def update_hidden_states(
        self,
        name: str,
        *,
        input_scales: Mapping[str, float] | None = None,
        output_scales: Mapping[str, float] | None = None,
        h0: Iterable[float] | None = None,
    ) -> Self:
        """Update the input or output scales or the initial values of a block.

        Weights are updated by name through `update_addressable`.

        Parameters
        ----------
        name
            Name of the block
        input_scales
            New scales by input name, the others are kept
        output_scales
            New scales by target name, the others are kept
        h0
            New initial values of the hidden states

        Returns
        -------
        Self
            The instance of the model with the updated block.

        """
        block = self._hidden_states[name]
        if input_scales is not None:
            if unknown := sorted(set(input_scales) - set(block.inputs)):
                msg = f"Hidden states {name!r}: scales for unknown inputs {unknown}"
                raise KeyError(msg)
            _check_scales(name, input_scales)
            for k, v in input_scales.items():
                block.input_scales[block.inputs.index(k)] = float(v)
            block._lowered = None  # noqa: SLF001, scales are constants of the equations
            self._invalidate()
        if output_scales is not None:
            if unknown := sorted(set(output_scales) - set(block.targets)):
                msg = f"Hidden states {name!r}: scales for unknown targets {unknown}"
                raise KeyError(msg)
            _check_scales(name, output_scales)
            for k, v in output_scales.items():
                block.output_scales[block.targets.index(k)] = float(v)
            block._lowered = None  # noqa: SLF001
            self._invalidate()
        if h0 is not None:
            values = np.asarray(list(h0), dtype=float)
            if values.shape != block.h0.shape:
                msg = (
                    f"Hidden states {name!r}: expected {block.n_hidden} initial values"
                )
                raise ValueError(msg)
            block.h0[:] = values
            if block.h0_assignments:
                block.h0_assignments.clear()
                self._invalidate()
            self._cache = None
        return self

    def _update_hidden_state(self, name: str, initial_value: Any) -> bool:
        """Set `h(0)` of a hidden state, False if `name` isn't one."""
        for k, block in self._hidden_states.items():
            if name in (states := block.state_names(k)):
                i = states.index(name)
                if isinstance(initial_value, InitialAssignment):
                    block.h0_assignments[i] = initial_value
                    self._invalidate()
                    return True
                if block.h0_assignments.pop(i, None) is not None:
                    self._invalidate()
                block.h0[i] = float(initial_value)
                self._cache = None
                return True
        return False

    def _check_not_hidden_state(self, name: str) -> None:
        if name in self.get_hidden_state_names():
            msg = f"{name!r} belongs to a hidden-state block, remove the block instead"
            raise ValueError(msg)

    def _check_not_block_input(self, name: str) -> None:
        reading = [
            k
            for k, v in self._hidden_states.items()
            if name in v.inputs
            or name == v.scale
            or any(name in ia.args for ia in v.h0_assignments.values())
        ]
        if reading:
            msg = f"{name!r} is read by the hidden-state blocks {reading}, remove them first"
            raise ValueError(msg)

    def _lowered_hidden_states(self) -> list[LoweredHiddenStates]:
        return [v.lower(k) for k, v in self._hidden_states.items()]

    def _hidden_state_initial_values(self) -> dict[str, float]:
        """Numeric `h(0)`, without the states set by initial assignments."""
        return {
            h: float(v)
            for k, block in self._hidden_states.items()
            for i, (h, v) in enumerate(zip(block.state_names(k), block.h0, strict=True))
            if i not in block.h0_assignments
        }

    def _hidden_state_initial_assignments(self) -> dict[str, InitialAssignment]:
        return {
            block.state_names(k)[i]: ia
            for k, block in self._hidden_states.items()
            for i, ia in block.h0_assignments.items()
        }

    def _hidden_state_nodes(self) -> dict[str, Derived]:
        return {k: v for b in self._lowered_hidden_states() for k, v in b.nodes.items()}

    def _all_terms(self) -> dict[str, Term]:
        """User terms, then the terms of every block."""
        terms = dict(self._terms)
        for block in self._lowered_hidden_states():
            terms |= block.terms
        return terms

    def _hidden_state_targets(self) -> dict[str, list[str]]:
        return {k: list(v.targets) for k, v in self._hidden_states.items()}

    ###########################################################################
    # Addressable names
    ###########################################################################

    def get_weight_names(self) -> list[str]:
        """Names of the weights of all hidden-state blocks."""
        return [w for k, v in self._hidden_states.items() for w in v.weight_names(k)]

    def get_weight_values(self) -> dict[str, float]:
        """Values of the weights of all hidden-state blocks by name."""
        out: dict[str, float] = {}
        for k, v in self._hidden_states.items():
            out |= v.weight_values(k)
        return out

    def get_addressable_names(self) -> list[str]:
        """Parameters and weights: every name a value can be set for.

        The layout of the parameter vector of compiled code, sensitivities and
        fits. Mechanistic analyses use `get_parameter_names` instead.
        """
        return self.get_parameter_names() + self.get_weight_names()

    def get_addressable_values(self) -> dict[str, float]:
        """Values of `get_addressable_names`."""
        return self.get_parameter_values() | self.get_weight_values()

    def update_addressable(self, values: Mapping[str, float]) -> Self:
        """Update parameters and weights by name.

        Parameters
        ----------
        values
            New values by parameter or weight name

        Returns
        -------
        Self
            The instance of the model with the updated values.

        """
        weights: dict[str, dict[str, float]] = {}
        parameters: dict[str, float] = {}
        owner = {
            w: k for k, v in self._hidden_states.items() for w in v.weight_names(k)
        }
        known = set(self.get_parameter_names())
        if unknown := sorted(k for k in values if k not in owner and k not in known):
            msg = f"Unknown parameters or weights {unknown}"
            raise KeyError(msg)
        for k, v in values.items():
            if (block := owner.get(k)) is not None:
                weights.setdefault(block, {})[k] = v
            else:
                parameters[k] = v
        if parameters:
            self.update_parameters(parameters)
        if weights:
            for block, w in weights.items():
                self._hidden_states[block].set_weights(block, w)
            self._cache = None
        return self
