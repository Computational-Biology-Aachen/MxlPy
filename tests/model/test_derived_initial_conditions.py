import pandas as pd

from mxlpy import Derived, Model, fns


def test_derived_initial_from_variable() -> None:
    m = Model()
    m.add_variables(
        {
            "x": 1.0,
            "y": Derived(fn=fns.twice, args=["x"]),
        }
    )

    assert m.get_initial_conditions() == {"x": 1.0, "y": 2.0}

    pd.testing.assert_series_equal(
        m.get_args(),
        pd.Series(
            {
                "x": 1.0,
                "y": 2.0,
                "time": 0.0,
            }
        ),
    )


def test_derived_initial_from_derived() -> None:
    m = Model()
    m.add_variables(
        {
            "x": 1.0,
            "y": Derived(fn=fns.twice, args=["d1"]),
        }
    )
    m.add_derived(
        "d1",
        fn=fns.twice,
        args=["x"],
    )

    assert m.get_initial_conditions() == {"x": 1.0, "y": 4.0}

    pd.testing.assert_series_equal(
        m.get_args(),
        pd.Series(
            {
                "x": 1.0,
                "y": 4.0,
                "time": 0.0,
                "d1": 2.0,
            }
        ),
    )


def test_derived_initial_from_rate() -> None:
    m = Model()
    m.add_variables(
        {
            "x": 1.0,
            "y": Derived(fn=fns.twice, args=["v1"]),
        }
    )
    m.add_reaction(
        "v1",
        fn=fns.twice,
        args=["x"],
        stoichiometry={"x": -1, "y": 1},
    )

    assert m.get_initial_conditions() == {"x": 1.0, "y": 4.0}

    pd.testing.assert_series_equal(
        m.get_args(),
        pd.Series(
            {
                "x": 1.0,
                "y": 4.0,
                "time": 0.0,
                "v1": 2.0,
            }
        ),
    )
