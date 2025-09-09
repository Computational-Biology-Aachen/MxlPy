from collections.abc import Callable

import numpy as np
import pandas as pd

from example_models import get_linear_chain_2v
from mxlpy import fit
from mxlpy.fit import (
    Bounds,
    MinResult,
    ResidualFn,
    _Settings,
)
from mxlpy.model import Model
from mxlpy.types import Array, ArrayLike, unwrap


def mock_minimizer(
    residual_fn: ResidualFn,  # noqa: ARG001
    p0: dict[str, float],
    bounds: Bounds | None,  # noqa: ARG001
) -> MinResult | None:
    return MinResult(parameters=p0, residual=0.0)


def mock_residual_fn(
    updates: dict[str, float],  # noqa: ARG001
) -> float:
    return 0.0


def mock_residual_proto(
    updates: dict[str, float],  # noqa: ARG001
    settings: _Settings,  # noqa: ARG001
) -> float:
    return 0.0


class MockIntegrator:
    def __init__(
        self,
        rhs: Callable,  # noqa: ARG002
        y0: tuple[float, ...],
        jacobian: Callable | None = None,  # noqa: ARG002
    ) -> None:
        self.y0 = y0

    def reset(self) -> None:
        return

    def integrate(
        self,
        *,
        t_end: float,  # noqa: ARG002
        steps: int | None = None,  # noqa: ARG002
    ) -> tuple[Array | None, ArrayLike | None]:
        t = np.array([0.0])
        y = np.ones((1, len(self.y0)))
        return t, y

    def integrate_time_course(
        self,
        *,
        time_points: ArrayLike | None = None,  # noqa: ARG002
    ) -> tuple[Array | None, ArrayLike | None]:
        t = np.array([0.0])
        y = np.ones((1, len(self.y0)))
        return t, y

    def integrate_to_steady_state(
        self,
        *,
        tolerance: float,  # noqa: ARG002
        rel_norm: bool,  # noqa: ARG002
    ) -> tuple[float | None, ArrayLike | None]:
        t = 0.0
        y = np.ones(len(self.y0))
        return t, y


def test_default_minimizer() -> None:
    p_true = {"k1": 1.0, "k2": 2.0, "k3": 1.0}
    p_fit = fit.LocalScipyMinimizer()(
        mock_residual_fn,
        p_true,
        bounds={},
    )
    assert p_fit is not None
    assert np.allclose(pd.Series(p_fit.parameters), pd.Series(p_true), rtol=0.1)


def test_fit_steady_state() -> None:
    p_true = {"k1": 1.0, "k2": 2.0, "k3": 1.0}
    data = pd.Series()
    p_fit = fit.steady_state(
        model=Model().add_parameters(p_true),
        p0=p_true,
        data=data,
        minimizer=mock_minimizer,
        residual_fn=mock_residual_proto,
    )
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
    )
    assert p_fit is not None
    assert np.allclose(pd.Series(p_fit.best_pars), pd.Series(p_true), rtol=0.1)


if __name__ == "__main__":
    from mxlpy import Simulator

    model_fn = get_linear_chain_2v
    p_true = {"k1": 1.0, "k2": 2.0, "k3": 1.0}
    p_init = {"k1": 1.038, "k2": 1.87, "k3": 1.093}
    res = unwrap(
        Simulator(model_fn())
        .update_parameters(p_true)
        .simulate_time_course(np.linspace(0, 1, 11))
        .get_result()
    ).get_combined()

    p_fit = fit.steady_state(
        model_fn(),
        p0=p_init,
        data=res.iloc[-1],
        minimizer=fit.LocalScipyMinimizer(),
    )
    assert p_fit is not None
    assert np.allclose(pd.Series(p_fit.best_pars), pd.Series(p_true), rtol=0.1)

    p_fit = fit.time_course(
        model_fn(),
        p0=p_init,
        data=res,
        minimizer=fit.LocalScipyMinimizer(),
    )
    assert p_fit is not None
    assert np.allclose(pd.Series(p_fit.best_pars), pd.Series(p_true), rtol=0.1)
