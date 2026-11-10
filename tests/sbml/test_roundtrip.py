# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

import pandas as pd

from mxlpy import KineticModelBuilder, fns
from mxlpy.paths import default_tmp_dir
from mxlpy.sbml import read, write
from mxlpy.types import Derived

TMP_DIR = default_tmp_dir(None, remove_old_cache=False)


def _canonicalise(self: KineticModelBuilder) -> KineticModelBuilder:
    """Canonicalise the model for comparison."""

    # Remove default compartment and any derived that reference it (e.g. x_amount)
    if "compartment" in self.get_raw_parameters(as_copy=False):
        for name, dp in list(self.get_raw_derived().items()):
            if "compartment" in dp.args:
                self.remove_derived(name)
        self.remove_parameter("compartment")

    def _handle_derived(dp: Derived) -> Derived:
        if dp.fn == fns.constant and (target := dp.args[0]) in self._derived:
            new = self._derived[target]
            self.remove_derived(target)
            self._cache = None
            return new

        return dp

    for name, rxn in self.get_raw_reactions().items():
        self.update_reaction(
            name,
            stoichiometry={
                k: _handle_derived(v) if isinstance(v, Derived) else v
                for k, v in rxn.stoichiometry.items()
            },
        )

    return self


def test_roundtrip_empty() -> None:
    m1 = KineticModelBuilder()
    m2 = _canonicalise(read(write(m1, TMP_DIR / "roundtrip_empty.xml")))
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args())


def test_roundtrip_parameter() -> None:
    m1 = KineticModelBuilder().add_parameter("k1", 2.0)
    m2 = _canonicalise(read(write(m1, TMP_DIR / "roundtrip_parameter.xml")))
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args())


def test_roundtrip_derived_parameter() -> None:
    m1 = (
        KineticModelBuilder()
        .add_parameter("k1", 2.0)
        .add_derived("d1", fns.constant, args=["k1"])
    )
    m2 = _canonicalise(read(write(m1, TMP_DIR / "roundtrip_derived_parameter.xml")))
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args())


def test_roundtrip_variable() -> None:
    m1 = KineticModelBuilder().add_variable("x", 2.0)
    m2 = _canonicalise(read(write(m1, TMP_DIR / "roundtrip_variable.xml")))
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args())


def test_roundtrip_derived_variable() -> None:
    m1 = (
        KineticModelBuilder()
        .add_variable("x", 2.0)
        .add_derived("d1", fns.constant, args=["x"])
    )
    m2 = _canonicalise(read(write(m1, TMP_DIR / "roundtrip_derived_variable.xml")))
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args())


def test_roundtrip_reaction() -> None:
    m1 = (
        KineticModelBuilder()
        .add_parameter("k1", 2.0)
        .add_variable("x", 2.0)
        .add_reaction("v1", fns.constant, args=["k1"], stoichiometry={"x": -1})
    )
    m2 = _canonicalise(read(write(m1, TMP_DIR / "roundtrip_reaction.xml")))
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args())


def test_roundtrip_derived_stoichiometry_explicit() -> None:
    m1 = (
        KineticModelBuilder()
        .add_parameter("k1", 2.0)
        .add_variable("x", 2.0)
        .add_reaction(
            "v1",
            fns.constant,
            args=["k1"],
            stoichiometry={"x": Derived(fn=fns.constant, args=["k1"])},
        )
    )
    m2 = _canonicalise(
        read(write(m1, TMP_DIR / "roundtrip_rxn_derived_stoich_explicit.xml"))
    )
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args().drop("v1_x_stoich"))


def test_roundtrip_derived_stoichiometry_implicit() -> None:
    m1 = (
        KineticModelBuilder()
        .add_parameter("k1", 2.0)
        .add_variable("x", 2.0)
        .add_reaction(
            "v1",
            fns.constant,
            args=["k1"],
            stoichiometry={"x": "k1"},
        )
    )

    m2 = _canonicalise(
        read(write(m1, TMP_DIR / "roundtrip_rxn_derived_stoich_implicit.xml"))
    )
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args().drop("v1_x_stoich"))


def _roundtrip_rhs(m1: KineticModelBuilder, name: str) -> None:
    m2 = read(write(m1, TMP_DIR / f"{name}.xml"))
    pd.testing.assert_series_equal(
        m1.get_right_hand_side(), m2.get_right_hand_side(), check_names=False
    )


def test_roundtrip_derived_stoichiometry_keeps_sign() -> None:
    # Derived stoichiometries used to always be exported as reactants, so a
    # produced species came back as consumed (and vice versa)
    m1 = (
        KineticModelBuilder()
        .add_parameters({"k1": 2.0, "n": 3.0})
        .add_variables({"x": 2.0, "y": 1.0})
        .add_reaction(
            "v1",
            fns.constant,
            args=["k1"],
            stoichiometry={
                "x": Derived(fn=fns.neg, args=["n"]),
                "y": Derived(fn=fns.constant, args=["n"]),
            },
        )
    )
    _roundtrip_rhs(m1, "roundtrip_rxn_derived_stoich_sign")


def test_roundtrip_derived_stoichiometry_same_species_in_two_reactions() -> None:
    # The species reference ids used to be `{species}ref`, so two reactions
    # with a derived stoichiometry for the same species clashed
    m1 = (
        KineticModelBuilder()
        .add_parameters({"k1": 2.0, "k2": 0.5, "n": 3.0})
        .add_variable("x", 2.0)
        .add_reaction(
            "v1",
            fns.constant,
            args=["k1"],
            stoichiometry={"x": Derived(fn=fns.constant, args=["n"])},
        )
        .add_reaction(
            "v2",
            fns.mass_action_1s,
            args=["x", "k2"],
            stoichiometry={"x": Derived(fn=fns.neg, args=["n"])},
        )
    )
    _roundtrip_rhs(m1, "roundtrip_rxn_derived_stoich_shared_species")
