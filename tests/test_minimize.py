import numpy as np
import pandas as pd

from mxlpy import fit


def mock_residual_fn(
    updates: dict[str, float],  # noqa: ARG001
) -> float:
    return 0.0


def test_default_minimizer() -> None:
    p_true = {"k1": 1.0, "k2": 2.0, "k3": 1.0}
    p_fit = fit.LocalScipyMinimizer()(
        mock_residual_fn,
        p_true,
        bounds={},
    ).unwrap()
    assert p_fit is not None
    assert np.allclose(pd.Series(p_fit.parameters), pd.Series(p_true), rtol=0.1)
