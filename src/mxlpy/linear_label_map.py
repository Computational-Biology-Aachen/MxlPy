"""Linear Label Mapping Module for Metabolic Models.

This module implements linear label mapping functionality for tracking isotope labels
through metabolic networks. It provides utilities for:

- Generating linear isotope label combinations
- Mapping labels between substrates and products
- Processing stoichiometric coefficients for label transfer

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mxlpy.model import Derived, Model

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

__all__ = [
    "LinearLabelMapper",
]


def _generate_isotope_labels(base_name: str, num_labels: int) -> list[str]:
    """Generate list of isotopomer names for a compound.

    Args:
        base_name: Base name of the compound
        num_labels: Number of label positions

    Returns:
        List of isotopomer names in format base_name__position

    Raises:
        ValueError: If num_labels <= 0

    Examples:
        >>> _generate_isotope_labels("x", 2)

        ['x__0', 'x__1']

    """
    if num_labels > 0:
        return [f"{base_name}__{i}" for i in range(num_labels)]
    msg = f"Compound {base_name} must have labels"
    raise ValueError(msg)


def _unpack_stoichiometries(
    stoichiometries: Mapping[str, float | Derived],
) -> tuple[dict[str, int], dict[str, int]]:
    """Split reaction stoichiometry into substrates and products.

    Args:
        stoichiometries: Dictionary of {species: coefficient} pairs

    Returns:
        Tuple of (substrates, products) dictionaries with integer coefficients

    Raises:
        NotImplementedError: If derived quantities are used in stoichiometry

    Examples:
        >>> _unpack_stoichiometries({"A": -1, "B": 2})
        ({"A": 1}, {"B": 2})

    """
    substrates = {}
    products = {}
    for k, v in stoichiometries.items():
        if isinstance(v, Derived):
            raise NotImplementedError

        if v < 0:
            substrates[k] = int(-v)
        else:
            products[k] = int(v)
    return substrates, products


def _stoichiometry_to_duplicate_list(stoichiometry: dict[str, int]) -> list[str]:
    """Convert stoichiometry dictionary to expanded list of species.

    Args:
        stoichiometry: Dictionary of {species: coefficient} pairs

    Returns:
        List with species repeated according to coefficients

    Examples:
        >>> _stoichiometry_to_duplicate_list({"A": 2, "B": 1})
        ['A', 'A', 'B']

    """
    long_form: list[str] = []
    for k, v in stoichiometry.items():
        long_form.extend([k] * v)
    return long_form


def _map_substrates_to_labelmap(
    substrates: list[str], labelmap: list[int]
) -> list[str]:
    """Map substrate labels to product label positions.

    Args:
        substrates: List of substrate names
        labelmap: List of integers mapping substrate positions to product positions

    Returns:
        Dictionary mapping substrate names to product label positions

    Examples:
        >>> _map_substrates_to_labelmap(['A', 'B'], [1, 0])
        {'A': 1, 'B': 0}

    """
    res = ["EXT"] * len(substrates)
    for substrate, pos in zip(substrates, labelmap, strict=True):
        res[pos] = substrate
    return res


def _add_label_influx_or_efflux(
    substrates: list[str],
    products: list[str],
    labelmap: list[int],
) -> tuple[list[str], list[str]]:
    """Add label influx or efflux to balance substrate and product lists.

    This function ensures that the substrate and product lists have equal length by adding
    external ("EXT") placeholders where needed. It also validates that the labelmap contains
    enough labels for all substrates.

    Args:
        substrates: List of substrate identifiers
        products: List of product identifiers
        labelmap: List of integer labels corresponding to substrate positions

    Returns:
        tuple: A tuple containing:
            - List of substrates (possibly extended with "EXT")
            - List of products (possibly extended with "EXT")

    Raises:
        ValueError: If the labelmap length is less than the number of substrates

    """
    # Add label outfluxes
    if (diff := len(substrates) - len(products)) > 0:
        products.extend(["EXT"] * diff)

    # Label influxes
    if (diff := len(products) - len(substrates)) > 0:
        substrates.extend(["EXT"] * diff)

    # Broken labelmap
    if (diff := len(labelmap) - len(substrates)) < 0:
        msg = f"Labelmap 'missing' {abs(diff)} label(s)"
        raise ValueError(msg)
    return substrates, products


def _relative_label_flux(label_percentage: float, v_ss: float) -> float:
    """Calculate relative label flux based on label percentage and steady-state flux."""
    return label_percentage * v_ss


def _one_div(y: float) -> float:
    """Calculate 1/y."""
    return 1 / y


def _neg_one_div(y: float) -> float:
    """Calculate -1/y."""
    return -1 / y


@dataclass(slots=True)
class LinearLabelMapper:
    """A class to map linear labels for a given model and build a model with isotopomers.

    Attributes:
        model (Model): The model to which the labels are mapped.
        label_variables (dict[str, int]): A dictionary mapping label names to their respective counts.
        label_maps (dict[str, list[int]]): A dictionary mapping reaction names to their respective label maps.

    Methods:
        get_isotopomers(variables: list[str]) -> dict[str, list[str]]:
            Generates isotopomers for the given variables based on label variables.

        build_model(
            initial_labels: dict[str, int | list[int]] | None = None
            Builds and returns a model with the given concentrations, fluxes, external label, and initial labels.

    """

    model: Model
    label_variables: dict[str, int] = field(default_factory=dict)
    label_maps: dict[str, list[int]] = field(default_factory=dict)

    def get_isotopomers(self, variables: list[str]) -> dict[str, list[str]]:
        """Generate a dictionary of isotopomers for the given variables.

        This method creates a dictionary where the keys are the variable names
        provided in the `variables` list, and the values are lists of isotopomer
        labels generated for each variable.

        Args:
            variables (list[str]): A list of variable names for which to generate
                                   isotopomer labels.

        Returns:
            dict[str, list[str]]: A dictionary where the keys are the variable names
                                  from the `variables` list, and the values are lists
                                  of isotopomer labels for each variable.

        """
        isotopomers = {
            name: _generate_isotope_labels(name, num)
            for name, num in self.label_variables.items()
        }
        return {k: isotopomers[k] for k in variables}

    def build_model(
        self,
        concs: pd.Series,
        fluxes: pd.Series,
        external_label: float = 1.0,
        initial_labels: dict[str, int | list[int]] | None = None,
    ) -> Model:
        """Build a metabolic model with labeled isotopomers and reactions.

        Examples:
            >>> mapper = LinearLabelMapper(
            ...     model,
            ...     label_variables={"A": 2, "B": 2},
            ...     label_maps={"v1": [0, 1], "v2": [1, 2]},
            ... )
            >>> mapper.build_model(concs, fluxes)

        Args:
            concs : pd.Series
                A pandas Series containing concentration values for metabolites.
            fluxes : pd.Series
                A pandas Series containing flux values for reactions.
            external_label : float, optional
                The label value for external metabolites, by default 1.0.
            initial_labels : dict[str, int | list[int]] | None, optional
                A dictionary specifying initial labeling positions for base compounds.
                Keys are compound names, and values are either a single integer or a list of integers
                indicating the positions to be labeled. Default is None.

        """
        isotopomers = {
            name: _generate_isotope_labels(name, num)
            for name, num in self.label_variables.items()
        }
        variables = {k: 0.0 for iso in isotopomers.values() for k in iso}
        if initial_labels is not None:
            for base_compound, label_positions in initial_labels.items():
                if isinstance(label_positions, int):
                    label_positions = [label_positions]  # noqa: PLW2901
                for pos in label_positions:
                    variables[f"{base_compound}__{pos}"] = 1 / len(label_positions)

        m = Model()
        m.add_variables(variables)
        m.add_parameters(concs.to_dict() | fluxes.to_dict() | {"EXT": external_label})

        rxns = self.model.get_raw_reactions()
        for rxn_name, label_map in self.label_maps.items():
            rxn = rxns[rxn_name]
            subs, prods = _unpack_stoichiometries(rxn.stoichiometry)

            subs = _stoichiometry_to_duplicate_list(subs)
            prods = _stoichiometry_to_duplicate_list(prods)
            subs = [j for i in subs for j in isotopomers[i]]
            prods = [j for i in prods for j in isotopomers[i]]
            subs, prods = _add_label_influx_or_efflux(subs, prods, label_map)
            subs = _map_substrates_to_labelmap(subs, label_map)
            for i, (substrate, product) in enumerate(zip(subs, prods, strict=True)):
                if substrate == product:
                    continue

                stoichiometry = {}
                if substrate != "EXT":
                    stoichiometry[substrate] = Derived(
                        fn=_neg_one_div, args=[substrate.split("__")[0]]
                    )
                if product != "EXT":
                    stoichiometry[product] = Derived(
                        fn=_one_div, args=[product.split("__")[0]]
                    )

                m.add_reaction(
                    name=f"{rxn_name}__{i}",
                    fn=_relative_label_flux,
                    stoichiometry=stoichiometry,
                    args=[substrate, rxn_name],
                )
        return m
