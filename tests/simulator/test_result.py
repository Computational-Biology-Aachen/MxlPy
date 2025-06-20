import pandas as pd

from mxlpy.types import Result
from tests import models


def create_result() -> Result:
    return Result(
        model=models.m_1v_1p_1d_1r(),
        raw_variables=[
            pd.DataFrame(
                {"v1": [1, 2, 3]},
                index=[0, 1, 2],
                dtype=float,
            ),
        ],
        raw_parameters=[
            {"p1": 1.0},
        ],
    )


def test_normalise_split_results() -> None:
    """Test _normalise_split_results function."""
    # Create test data
    df1 = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    df2 = pd.DataFrame({"A": [5.0, 6.0], "B": [7.0, 8.0]})
    results = [df1, df2]

    # Test scalar normalization
    from mxlpy.types import _normalise_split_results

    normalized = _normalise_split_results(results, normalise=2.0)
    assert len(normalized) == 2
    assert normalized[0].iloc[0, 0] == 0.5  # 1.0 / 2.0
    assert normalized[1].iloc[0, 0] == 2.5  # 5.0 / 2.0

    # Test array normalization with matching length
    norm_array = [10.0, 20.0]
    normalized = _normalise_split_results(results, normalise=norm_array)
    assert len(normalized) == 2
    assert normalized[0].iloc[0, 0] == 0.1  # 1.0 / 10.0
    assert normalized[1].iloc[0, 0] == 0.25  # 5.0 / 20.0


def test_variables() -> None:
    res = create_result()
    pd.testing.assert_frame_equal(
        res.variables,
        pd.DataFrame(
            {"v1": {0: 1, 1: 2, 2: 3}, "d1": {0: 2.0, 1: 3.0, 2: 4.0}}, dtype=float
        ),
    )


def test_get_variables() -> None:
    res = create_result()
    pd.testing.assert_frame_equal(
        res.get_variables(include_derived_variables=True, include_readouts=True),
        pd.DataFrame(
            {"v1": {0: 1, 1: 2, 2: 3}, "d1": {0: 2.0, 1: 3.0, 2: 4.0}}, dtype=float
        ),
    )
    pd.testing.assert_frame_equal(
        res.get_variables(include_derived_variables=False, include_readouts=False),
        pd.DataFrame({"v1": {0: 1, 1: 2, 2: 3}}, dtype=float),
    )


def test_get_variables_not_concatenated() -> None:
    res = create_result()
    pd.testing.assert_frame_equal(
        res.get_variables(concatenated=False)[0],
        pd.DataFrame(
            {"v1": {0: 1, 1: 2, 2: 3}, "d1": {0: 2.0, 1: 3.0, 2: 4.0}}, dtype=float
        ),
    )


def test_fluxes() -> None:
    res = create_result()
    pd.testing.assert_frame_equal(
        res.fluxes,
        pd.DataFrame(
            {"r1": {0: 2.0, 1: 6.0, 2: 12.0}},
            dtype=float,
        ),
    )


def test_get_fluxes() -> None:
    res = create_result()
    pd.testing.assert_frame_equal(
        res.get_fluxes(include_surrogates=True),
        pd.DataFrame(
            {"r1": {0: 2.0, 1: 6.0, 2: 12.0}},
            dtype=float,
        ),
    )
    pd.testing.assert_frame_equal(
        res.get_fluxes(include_surrogates=False),
        pd.DataFrame(
            {"r1": {0: 2.0, 1: 6.0, 2: 12.0}},
            dtype=float,
        ),
    )


def test_get_fluxes_not_concatenated() -> None:
    res = create_result()
    pd.testing.assert_frame_equal(
        res.get_fluxes(concatenated=False)[0],
        pd.DataFrame({"r1": {0: 2.0, 1: 6.0, 2: 12.0}}, dtype=float),
    )


def test_get_combined() -> None:
    res = create_result()
    pd.testing.assert_frame_equal(
        res.get_combined(),
        pd.DataFrame(
            {
                "v1": {0: 1, 1: 2, 2: 3},
                "d1": {0: 2.0, 1: 3.0, 2: 4.0},
                "r1": {0: 2.0, 1: 6.0, 2: 12.0},
            },
            dtype=float,
        ),
    )


def test_get_new_y0() -> None:
    res = create_result()
    pd.testing.assert_series_equal(
        pd.Series(res.get_new_y0(), dtype=float),
        pd.Series({"v1": 3.0}, dtype=float),
    )
