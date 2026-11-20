# Copyright 2025 Marvin van Aalst, Anna Matuszyńska


def test_cosine_similarity() -> None:
    # FIXME: implement this
    assert True


def test_mae() -> None:
    # FIXME: implement this
    assert True


def test_mean() -> None:
    # FIXME: implement this
    assert True


def test_mean_absolute_percentage() -> None:
    # FIXME: implement this
    assert True


def test_mean_squared() -> None:
    # FIXME: implement this
    assert True


def test_mean_squared_logarithmic() -> None:
    # FIXME: implement this
    assert True


def test_rmse() -> None:
    # FIXME: implement this
    assert True


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from mxlpy.fit import GaussianNll  # noqa: E402


def test_gaussian_nll() -> None:
    data = pd.DataFrame({"x": [1.0, 2.0, np.nan], "y": [0.0, 1.0, 2.0]})
    pred = data + 0.5
    assert GaussianNll(0.5)(pred, data) == pytest.approx(0.5 * 5)
    sigma = pd.Series({"x": 0.5, "y": 1.0})
    assert GaussianNll(sigma)(pred, data) == pytest.approx(0.5 * (2 + 3 * 0.25))
    per_point = pd.DataFrame(0.25, index=data.index, columns=data.columns)
    assert GaussianNll(per_point)(pred, data) == pytest.approx(0.5 * 5 * 4)


def test_gaussian_nll_checks_sigma() -> None:
    with pytest.raises(ValueError, match="positive"):
        GaussianNll(0.0)
    data = pd.DataFrame({"x": [1.0], "y": [2.0]})
    with pytest.raises(ValueError, match="every observed"):
        GaussianNll(pd.Series({"x": 1.0}))(data, data + 1)
