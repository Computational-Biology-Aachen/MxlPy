import pandas as pd

from mxlpy import Model, fns
from mxlpy.types import Result


def test_get_producers_and_consumers() -> None:
    m = (
        Model()
        .add_variable("x", 1.0)
        .add_parameter("k", 2.0)
        .add_reaction("v_in", fns.constant, args=["k"], stoichiometry={"x": 2})
        .add_reaction("v_out", fns.constant, args=["k"], stoichiometry={"x": -2})
    )

    res = Result(
        model=m,
        raw_variables=[
            pd.DataFrame({"x": [1.0]}, index=[0.0]),
            pd.DataFrame({"x": [1.0]}, index=[1.0]),
        ],
        raw_parameters=[{"k": 1.0}, {"k": 2.0}],
    )

    pd.testing.assert_frame_equal(
        res.get_producers("x"),
        pd.DataFrame({"v_in": {0.0: 1.0, 1.0: 2.0}}),
    )
    pd.testing.assert_frame_equal(
        res.get_producers("x", scaled=True),
        pd.DataFrame({"v_in": {0.0: 2.0, 1.0: 4.0}}),
    )
    pd.testing.assert_frame_equal(
        res.get_consumers("x"),
        pd.DataFrame({"v_out": {0.0: 1.0, 1.0: 2.0}}),
    )
    pd.testing.assert_frame_equal(
        res.get_consumers("x", scaled=True),
        pd.DataFrame({"v_out": {0.0: 2.0, 1.0: 4.0}}),
    )
