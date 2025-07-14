import pandas as pd

from mxlpy import Model, fns
from mxlpy.paths import default_tmp_dir
from mxlpy.sbml import read, write
from mxlpy.types import Derived

TMP_DIR = default_tmp_dir(None, remove_old_cache=False)


def _canonicalise(self: Model) -> Model:
    """Canonicalise the model for comparison."""

    # Remove default compartment
    if "compartment" in self.get_raw_parameters(as_copy=False):
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
    m1 = Model()
    m2 = _canonicalise(read(write(m1, TMP_DIR / "roundtrip_empty.xml")))
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args())


def test_roundtrip_parameter() -> None:
    m1 = Model().add_parameter("k1", 2.0)
    m2 = _canonicalise(read(write(m1, TMP_DIR / "roundtrip_parameter.xml")))
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args())


def test_roundtrip_derived_parameter() -> None:
    m1 = Model().add_parameter("k1", 2.0).add_derived("d1", fns.constant, args=["k1"])
    m2 = _canonicalise(read(write(m1, TMP_DIR / "roundtrip_derived_parameter.xml")))
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args())


def test_roundtrip_variable() -> None:
    m1 = Model().add_variable("x", 2.0)
    m2 = _canonicalise(read(write(m1, TMP_DIR / "roundtrip_variable.xml")))
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args())


def test_roundtrip_derived_variable() -> None:
    m1 = Model().add_variable("x", 2.0).add_derived("d1", fns.constant, args=["x"])
    m2 = _canonicalise(read(write(m1, TMP_DIR / "roundtrip_derived_variable.xml")))
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args())


def test_roundtrip_reaction() -> None:
    m1 = (
        Model()
        .add_parameter("k1", 2.0)
        .add_variable("x", 2.0)
        .add_reaction("v1", fns.constant, args=["k1"], stoichiometry={"x": -1})
    )
    m2 = _canonicalise(read(write(m1, TMP_DIR / "roundtrip_reaction.xml")))
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args())


def test_roundtrip_derived_stoichiometry_explicit() -> None:
    m1 = (
        Model()
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
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args().drop("xref"))


def test_roundtrip_derived_stoichiometry_implicit() -> None:
    m1 = (
        Model()
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
    pd.testing.assert_series_equal(m1.get_args(), m2.get_args().drop("xref"))
