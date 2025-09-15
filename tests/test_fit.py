import numpy as np
import pandas as pd

from example_models import get_linear_chain_2v
from mxlpy import fit
from mxlpy.fit.abstract import _Settings
from mxlpy.minimizers.abstract import mock_minimizer
from mxlpy.model import Model


def mock_residual_proto(
    updates: dict[str, float],  # noqa: ARG001
    settings: _Settings,  # noqa: ARG001
) -> float:
    return 0.0


def test_fit_steady_state() -> None:
    p_true = {"k1": 1.0, "k2": 2.0, "k3": 1.0}
    data = pd.Series()
    p_fit = fit.steady_state(
        model=Model().add_parameters(p_true),
        p0=p_true,
        data=data,
        minimizer=mock_minimizer,
        residual_fn=mock_residual_proto,
    ).unwrap()
    assert p_fit is not None
    assert np.allclose(pd.Series(p_fit.best_pars), pd.Series(p_true), rtol=0.1)


def tets_fit_time_course() -> None:
    p_true = {"k1": 1.0, "k2": 2.0, "k3": 1.0}
    data = pd.DataFrame()
    p_fit = fit.time_course(
        model=Model(),
        p0=p_true,
        data=data,
        minimizer=mock_minimizer,
        residual_fn=mock_residual_proto,
    ).unwrap()
    assert p_fit is not None
    assert np.allclose(pd.Series(p_fit.best_pars), pd.Series(p_true), rtol=0.1)


if __name__ == "__main__":
    from mxlpy import Simulator

    model_fn = get_linear_chain_2v
    p_true = {"k1": 1.0, "k2": 2.0, "k3": 1.0}
    p_init = {"k1": 1.038, "k2": 1.87, "k3": 1.093}
    res = (
        Simulator(model_fn())
        .update_parameters(p_true)
        .simulate_time_course(np.linspace(0, 1, 11))
        .get_result()
        .unwrap()
    ).get_combined()

    p_fit = fit.steady_state(
        model_fn(),
        p0=p_init,
        data=res.iloc[-1],
        minimizer=fit.LocalScipyMinimizer(),
    ).unwrap()
    assert p_fit is not None
    assert np.allclose(pd.Series(p_fit.best_pars), pd.Series(p_true), rtol=0.1)

    p_fit = fit.time_course(
        model_fn(),
        p0=p_init,
        data=res,
        minimizer=fit.LocalScipyMinimizer(),
    ).unwrap()
    assert p_fit is not None
    assert np.allclose(pd.Series(p_fit.best_pars), pd.Series(p_true), rtol=0.1)
