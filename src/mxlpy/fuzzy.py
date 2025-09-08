"""Fuzzy / bayesian fitting methods."""

from __future__ import annotations

import multiprocessing
import sys
from dataclasses import dataclass, field
from functools import partial
from math import ceil
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd
import pebble
from tqdm import tqdm, trange

from mxlpy.simulator import Simulator

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mxlpy import Model

__all__ = ["ThompsonState", "thompson_sampling"]


@dataclass
class ThompsonState:
    """State of thompson sampling."""

    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    state: dict[str, pd.DataFrame] = field(default_factory=dict)

    @classmethod
    def from_parameter_values(cls, parameters: dict[str, Iterable[float]]) -> Self:
        """Create state from parameter values."""
        return cls(
            state={
                k: pd.DataFrame(
                    {
                        "x": v,
                        "success": np.ones_like(v, dtype=int),
                        "fail": np.ones_like(v, dtype=int),
                    }
                )
                for k, v in parameters.items()
            },
        )

    def sample(self) -> tuple[dict[str, int], dict[str, float]]:
        """Sample idxs and parameters."""
        idxs = {
            k: int(np.argmax(self.rng.beta(v["success"], v["fail"])))
            for k, v in self.state.items()
        }
        parameters = {k: v["x"][idxs[k]] for k, v in self.state.items()}
        return idxs, parameters

    def update(
        self,
        idxs: dict[str, int],
        pred: pd.DataFrame | None,
        data: pd.DataFrame,
        rtol: float,
    ) -> None:
        """Sample state."""
        accept: bool = (
            False if pred is None else np.sqrt(np.mean(np.square(pred - data))) < rtol
        )
        for k, v in self.state.items():
            v.loc[idxs[k], "success" if accept else "fail"] += 1  # type: ignore


def _thompson_worker(
    inp: tuple[dict[str, int], dict[str, float]],
    model: Model,
    data: pd.DataFrame,
) -> tuple[dict[str, int], pd.DataFrame | None]:
    idxs, parameters = inp
    if (
        res := (
            Simulator(model)
            .update_parameters(parameters)
            .simulate_time_course(data.index)
            .get_result()
        )
    ) is None:
        return idxs, None
    return idxs, res.get_variables()


def thompson_sampling(
    model: Model,
    data: pd.DataFrame,
    state: ThompsonState,
    rtol: float,
    n: int,
    *,
    max_workers: int | None = None,
    disable_tqdm: bool = False,
    timeout: float | None = None,
    parallel: bool = True,
) -> ThompsonState:
    """Perform thompson sampling."""
    if sys.platform in ["win32", "cygwin"]:
        parallel = False

    max_workers = multiprocessing.cpu_count() if max_workers is None else max_workers
    worker = partial(_thompson_worker, model=model, data=data)

    if not parallel:
        for _ in trange(n):
            idxs, pred = worker(state.sample())
            state.update(idxs, pred, data=data, rtol=rtol)
    else:
        # FIXME: think about whether this is ok to do. Thompson sampling is state-
        # dependent. We are breaking up that state a bit by chunking the approach
        # Is that fine to do?
        with (
            tqdm(total=n, disable=disable_tqdm) as pbar,
            pebble.ProcessPool(max_workers=max_workers) as pool,
        ):
            for _ in range(ceil(n / max_workers)):
                future = pool.map(
                    worker,
                    [state.sample() for _ in range(max_workers)],
                    timeout=timeout,
                )
                it = future.result()
                while True:
                    try:
                        idxs, pred = next(it)
                        state.update(idxs, pred, data=data, rtol=rtol)
                        pbar.update(1)
                    except StopIteration:
                        break
                    except TimeoutError:
                        pbar.update(1)
    return state
