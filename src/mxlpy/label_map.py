"""Label Mapping Module for Metabolic Models.

This module provides functionality for mapping between labeled metabolites and their
isotopomers in metabolic models. It handles:

- Mapping between labeled and unlabeled species
- Generation of isotopomer combinations
- Calculation of total concentrations across isotopomers

Classes:
    LabelMapper: Maps between labeled and unlabeled metabolites to their isotopomers

"""

from __future__ import annotations

import itertools as it
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np

from mxlpy.model import Model

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


__all__ = [
    "LabelMapper",
]


def _total_concentration(*args: float) -> float:
    """Calculate sum of isotopomer concentrations.

    Args:
        *args: Individual isotopomer concentrations to sum

    Returns:
        float: Total concentration across all isotopomers

    Examples:
        >>> total_concentration(0.1, 0.2, 0.3)
        0.6

    """
    return cast(float, np.sum(args, axis=0))


def _generate_binary_labels(
    base_name: str,
    num_labels: int,
) -> list[str]:
    """Create binary label string.

    Examples:
        >>> _generate_binary_labels(base_name='cpd', num_labels=0)
        ['cpd']

        >>> _generate_binary_labels(base_name='cpd', num_labels=1)
        ['cpd__0', 'cpd__1']

        >>> _generate_binary_labels(base_name='cpd', num_labels=2)
        ['cpd__00', 'cpd__01', 'cpd__10', 'cpd__11']

    Args:
        base_name : str
            Name of the compound
        num_labels : int
            Number of label positions in the compound

    Returns:
        isotopomers : list(str)
            Returns a list of all label isotopomers of the compound

    """
    if num_labels > 0:
        return [
            base_name + "__" + "".join(i)
            for i in it.product(("0", "1"), repeat=num_labels)
        ]
    return [base_name]


def _split_label_string(
    label: str,
    labels_per_compound: list[int],
) -> list[str]:
    """Split label string according to labels given in label list.

    The labels in the label list correspond to the number of
    label positions in the compound.

    Examples:
        >>> _split_label_string(label="01", labels_per_compound=[2])
        ["01"]

        >>> _split_label_string(label="01", labels_per_compound=[1, 1])
        ["0", "1"]

        >>> _split_label_string(label="0011", labels_per_compound=[4])
        ["0011"]

        >>> _split_label_string(label="0011", labels_per_compound=[3, 1])
        ["001", "1"]

        >>> _split_label_string(label="0011", labels_per_compound=[2, 2])
        ["00", "11"]

        >>> _split_label_string(label="0011", labels_per_compound=[1, 3])
        ["0", "011"]

    Args:
        label : str
            Label string to split
        labels_per_compound : list(int)
            List of label positions per compound

    Returns:
        split_labels: List of split labels

    """
    split_labels = []
    cnt = 0
    for i in range(len(labels_per_compound)):
        split_labels.append(label[cnt : cnt + labels_per_compound[i]])
        cnt += labels_per_compound[i]
    return split_labels


def _map_substrates_to_products(
    rate_suffix: str,
    labelmap: list[int],
) -> str:
    """Map the rate_suffix to products using the labelmap.

    Examples:
        >>> _map_substrates_to_products(rate_suffix="01", labelmap=[1, 0])
        "10"

        >>> _map_substrates_to_products(rate_suffix="01", labelmap=[0, 1])
        "01"

        >>> _map_substrates_to_products(rate_suffix="01", labelmap=[1, 1])
        "11"

    Args:
        rate_suffix : str
            Label string of the substrate
        labelmap : list(int)
            List of label positions per compound

    Returns:
        str: Label string of the product

    """
    return "".join([rate_suffix[i] for i in labelmap])


def _unpack_stoichiometries(
    stoichiometries: Mapping[str, int],
) -> tuple[list[str], list[str]]:
    """Split stoichiometries into substrates and products.

    Examples:
        >>> _unpack_stoichiometries({"A": -1, "B": -2, "C": 1})
            (["A", "B", "B"], ["C"])

    Args:
        stoichiometries : dict(str: int)

    Returns:
        substrates : list(str)
        products : list(str)

    """
    substrates = []
    products = []
    for k, v in stoichiometries.items():
        if v < 0:
            substrates.extend([k] * -v)
        else:
            products.extend([k] * v)
    return substrates, products


def _get_labels_per_variable(
    label_variables: dict[str, int],
    compounds: list[str],
) -> list[int]:
    """Get labels per compound.

    This is used for _split_label string.
    Adds 0 for non-label compounds, to show that they get no label.

    Examples:
        >>> _get_labels_per_variable({"A": 1, "B": 2}, ["A", "B", "C"])
            [1, 2, 0]

    Args:
        label_variables : dict(str: int)
        compounds : list(str)

    Returns:
        list(int)

    """
    return [label_variables.get(compound, 0) for compound in compounds]


def _repack_stoichiometries(
    new_substrates: list[str],
    new_products: list[str],
) -> dict[str, float]:
    """Pack substrates and products into stoichiometric dict.

    Examples:
        >>> _repack_stoichiometries(["A", "B"], ["C"])
            {"A": -1, "B": -1, "C": 1}

    Args:
        new_substrates : list(str)
        new_products : list(str)

    Returns:
        dict(str: int)

    """
    new_stoichiometries: defaultdict[str, int] = defaultdict(int)
    for arg in new_substrates:
        new_stoichiometries[arg] -= 1
    for arg in new_products:
        new_stoichiometries[arg] += 1
    return dict(new_stoichiometries)


def _assign_compound_labels(
    base_compounds: list[str],
    label_suffixes: list[str],
) -> list[str]:
    """Assign the correct suffixes.

    Examples:
        >>> _assign_compound_labels(["A", "B"], ["", "01"])
            ["A", "B__01"]

    Args:
        base_compounds: the names of the compounds without labels
        label_suffixes: the labels to add to the compounds

    Returns:
        new compounds labels

    """
    new_compounds = []
    for i, compound in enumerate(base_compounds):
        if label_suffixes[i] != "":
            new_compounds.append(compound + "__" + label_suffixes[i])
        else:
            new_compounds.append(compound)
    return new_compounds


def _get_external_labels(
    *,
    total_product_labels: int,
    total_substrate_labels: int,
) -> str:
    """Get external labels.

    Examples:
        >>> _get_external_labels(total_product_labels=2, total_substrate_labels=1)
            "1"

        >>> _get_external_labels(total_product_labels=1, total_substrate_labels=1)
            ""

    Args:
        total_product_labels: total number of labels in the product
        total_substrate_labels: total number of labels in the substrate

    """
    n_external_labels = total_product_labels - total_substrate_labels
    if n_external_labels > 0:
        external_label_string = ["1"] * n_external_labels
        return "".join(external_label_string)
    return ""


def _create_isotopomer_reactions(
    model: Model,
    label_variables: dict[str, int],
    rate_name: str,
    function: Callable,
    stoichiometry: Mapping[str, int],
    labelmap: list[int],
    args: list[str],
) -> None:
    """Create isotopomer reactions.

    Examples:
        >>> _create_isotopomer_reactions(
        ...     model,
        ...     label_variables={"A": 1, "B": 2},
        ...     rate_name="rxn",
        ...     function=lambda x: x,
        ...     stoichiometry={"A": -1, "B": -2, "C": 1},
        ...     labelmap=[0, 1],
        ...     args=["A", "B", "C"]
        ... )

    Args:
        model: Model instance
        label_variables: dict(str: int)
        rate_name: str
        function: Callable
        stoichiometry: dict(str: int)
        labelmap: list(int)
        args: list(str)

    """
    base_substrates, base_products = _unpack_stoichiometries(
        stoichiometries=stoichiometry
    )
    labels_per_substrate = _get_labels_per_variable(
        label_variables=label_variables,
        compounds=base_substrates,
    )
    labels_per_product = _get_labels_per_variable(
        label_variables=label_variables,
        compounds=base_products,
    )
    total_substrate_labels = sum(labels_per_substrate)
    total_product_labels = sum(labels_per_product)

    if len(labelmap) - total_substrate_labels < 0:
        msg = (
            f"Labelmap 'missing' {abs(len(labelmap) - total_substrate_labels)} label(s)"
        )
        raise ValueError(msg)

    external_labels = _get_external_labels(
        total_product_labels=total_product_labels,
        total_substrate_labels=total_substrate_labels,
    )

    for rate_suffix in (
        "".join(i) for i in it.product(("0", "1"), repeat=total_substrate_labels)
    ):
        rate_suffix += external_labels  # noqa: PLW2901
        # This is the magic
        product_suffix = _map_substrates_to_products(
            rate_suffix=rate_suffix, labelmap=labelmap
        )
        product_labels = _split_label_string(
            label=product_suffix, labels_per_compound=labels_per_product
        )
        substrate_labels = _split_label_string(
            label=rate_suffix, labels_per_compound=labels_per_substrate
        )

        new_substrates = _assign_compound_labels(
            base_compounds=base_substrates, label_suffixes=substrate_labels
        )
        new_products = _assign_compound_labels(
            base_compounds=base_products, label_suffixes=product_labels
        )
        new_stoichiometry = _repack_stoichiometries(
            new_substrates=new_substrates, new_products=new_products
        )
        new_rate_name = rate_name + "__" + rate_suffix

        replacements = dict(zip(base_substrates, new_substrates, strict=True)) | dict(
            zip(base_products, new_products, strict=True)
        )

        model.add_reaction(
            name=new_rate_name,
            fn=function,
            stoichiometry=new_stoichiometry,
            args=[replacements.get(k, k) for k in args],
        )


@dataclass(slots=True)
class LabelMapper:
    """Maps between labeled and unlabeled species in metabolic models.

    Handles generation and mapping of isotopomers, including:
    - Creating all possible isotopomer combinations
    - Building labeled reaction networks
    - Calculating total concentrations

    Args:
        model: Model instance to map labels for
        label_variables: Dict mapping species to number of labels
        label_maps: Dict mapping reactions to label transfer patterns

    Examples:
        >>> mapper = LabelMapper(model)
        >>> isotopomers = mapper.get_isotopomers()

    """

    model: Model
    label_variables: dict[str, int] = field(default_factory=dict)
    label_maps: dict[str, list[int]] = field(default_factory=dict)

    def get_isotopomers(self) -> dict[str, list[str]]:
        """Get all possible isotopomers for each labeled species.

        Examples:
            >>> mapper.get_isotopomers()
                {cpd: [cpd__0, cpd__1], ...}

        Returns:
            Dict mapping species names to lists of isotopomer names

        """
        return {
            name: _generate_binary_labels(base_name=name, num_labels=num)
            for name, num in self.label_variables.items()
        }

    def get_isotopomer_of(self, name: str) -> list[str]:
        """Get all possible isotopomers for a specific species.

        Examples:
            >>> mapper.get_isotopomer_of("GAP")
                ['GAP__0', 'GAP__1']

        Args:
            name: Name of the labeled species

        Returns:
            List of isotopomer names

        """
        return _generate_binary_labels(
            base_name=name,
            num_labels=self.label_variables[name],
        )

    def get_isotopomers_by_regex(self, name: str, regex: str) -> list[str]:
        """Get isotopomers matching a regex pattern.

        Examples:
            >>> mapper.get_isotopomers_by_regex("GAP", "GAP__1[01]")
                ['GAP__10', 'GAP__11]

        Args:
            name: Name of the labeled species
            regex: Regular expression pattern to match

        Returns:
            List of matching isotopomer names

        """
        pattern = re.compile(regex)
        isotopomers = self.get_isotopomer_of(name=name)
        return [i for i in isotopomers if pattern.match(i)]

    def get_isotopomers_of_at_position(
        self, name: str, positions: int | list[int]
    ) -> list[str]:
        """Get isotopomers with specific label positions.

        Examples:
            >>> mapper.get_isotopomers_of_at_position("cpd", 0)
                ['cpd__10', 'cpd__00']
            >>> mapper.get_isotopomers_of_at_position("cpd", 1)
                ['cpd__01', 'cpd__00']

        Args:
            name: Name of the labeled species
            positions: Single position or list of positions to match

        Returns:
            List of matching isotopomer names

        """
        if isinstance(positions, int):
            positions = [positions]

        # Example for a variable with 3 labels
        # position 0 => GAP__1[01][01]
        # position 1 => GAP__[01]1[01]
        # position 2 => GAP__[01][01]1

        num_labels = self.label_variables[name]
        label_positions = ["[01]"] * num_labels
        for position in positions:
            label_positions[position] = "1"

        return self.get_isotopomers_by_regex(
            name, f"{name}__{''.join(label_positions)}"
        )

    def get_isotopomers_of_with_n_labels(self, name: str, n_labels: int) -> list[str]:
        """Get all isotopomers of a compound that have exactly n labels.

        Examples:
            >>> mapper.get_isotopomers_of_with_n_labels("GAP", 2)
                ['GAP__110', 'GAP__101', 'GAP__011']

        Args:
            name: Name of the labeled species
            n_labels: Number of labels to match

        Returns:
            List of isotopomer names with exactly n labels


        """
        label_positions = self.label_variables[name]
        label_patterns = [
            ["1" if i in positions else "0" for i in range(label_positions)]
            for positions in it.combinations(range(label_positions), n_labels)
        ]
        return [f"{name}__{''.join(i)}" for i in label_patterns]

    def build_model(
        self, initial_labels: dict[str, int | list[int]] | None = None
    ) -> Model:
        """Build new model with labeled species and reactions.

        Examples:
            >>> mapper = LabelMapper(
            ...     model,
            ...     label_variables={"A": 2, "B": 2},
            ...     label_maps={"v1": [0, 1], "v2": [1, 2]},
            ... )
            >>> mapper.build_model()
            >>> mapper.build_model(initial_labels={"A": 1})
            >>> mapper.build_model(initial_labels={"A": 1, "B": [0, 1]})

        Args:
            initial_labels: Dict mapping species to initial label positions.
                          Can be single position (int) or multiple (list).

        Returns:
            New Model instance with labeled components

        """
        isotopomers = self.get_isotopomers()
        initial_labels = {} if initial_labels is None else initial_labels

        m = Model()

        m.add_parameters(self.model.get_parameter_values())

        for name, dp in self.model.get_derived_parameters().items():
            m.add_derived(name, fn=dp.fn, args=dp.args)

        variables: dict[str, float] = {}
        for k, v in self.model.get_initial_conditions().items():
            if (isos := isotopomers.get(k)) is None:
                variables[k] = v
            else:
                label_pos = initial_labels.get(k)
                d = zip(isos, it.repeat(0), strict=False)
                variables.update(d)
                if label_pos is None:
                    variables[isos[0]] = v
                else:
                    if isinstance(label_pos, int):
                        label_pos = [label_pos]

                    suffix = "__" + "".join(
                        "1" if idx in label_pos else "0"
                        for idx in range(self.label_variables[k])
                    )
                    variables[f"{k}{suffix}"] = v

        m.add_variables(variables)

        for base_name, label_names in isotopomers.items():
            m.add_derived(
                name=f"{base_name}__total",
                fn=_total_concentration,
                args=label_names,
            )

        for name, dv in self.model.get_derived_variables().items():
            m.add_derived(
                name,
                fn=dv.fn,
                args=[f"{i}__total" if i in isotopomers else i for i in dv.args],
            )

        for rxn_name, rxn in self.model.get_raw_reactions().items():
            if (label_map := self.label_maps.get(rxn_name)) is None:
                m.add_reaction(
                    rxn_name,
                    rxn.fn,
                    args=[f"{i}__total" if i in isotopomers else i for i in rxn.args],
                    stoichiometry=rxn.stoichiometry,
                )
            else:
                _create_isotopomer_reactions(
                    model=m,
                    label_variables=self.label_variables,
                    rate_name=rxn_name,
                    stoichiometry=rxn.stoichiometry,  # type: ignore
                    function=rxn.fn,
                    labelmap=label_map,
                    args=rxn.args,
                )

        return m
