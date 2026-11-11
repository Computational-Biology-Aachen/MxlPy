# Copyright 2025 Marvin van Aalst, Anna Matuszyńska

"""Bayesian posterior sampling of model parameters by MCMC.

The log posterior is a Gaussian log likelihood of the (unscaled) data with a
given noise level `sigma`, plus the log density of each parameter's prior.
Parameters whose prior lives on (0, inf) are sampled as their logarithm, which
mixes much better across orders of magnitude; all results are reported in
natural units.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, overload

import numpy as np
import pandas as pd
from scipy import stats
from wadler_lindig import pformat

from mxlpy._kinetic_builder import KineticModelBuilder
from mxlpy._ode_builder import OdeModelBuilder
from mxlpy._steady_state_builder import SteadyStateModelBuilder
from mxlpy.fit import losses
from mxlpy.fit._single import _check_y0
from mxlpy.fit.abstract import _Settings, _split_p0
from mxlpy.fit.residuals import _apply_updates
from mxlpy.parallel import parallelise
from mxlpy.runners import DynamicRunner, SteadyStateRunner, default_runner
from mxlpy.simulator import _normalise_protocol_index
from mxlpy.types import Array, Result

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    from mxlpy.distributions import Prior
    from mxlpy.fit.abstract import PostprocessFn
    from mxlpy.plot import FigAxs
    from mxlpy.simulation import Evaluation, OdeSimulation, Simulation

__all__ = [
    "AdaptiveMetropolis",
    "McmcChain",
    "McmcResult",
    "McmcSampler",
    "Sigma",
    "mcmc_protocol_time_course",
    "mcmc_steady_state",
    "mcmc_time_course",
]

type _Model = KineticModelBuilder | OdeModelBuilder | SteadyStateModelBuilder

# Noise standard deviation: one for all data, per column (time courses) or
# entry (steady states), or per data point
type Sigma = float | pd.Series | pd.DataFrame

# Relative spread of the chains' starting points around x0
_START_JITTER = 0.1
_START_ATTEMPTS = 100


###############################################################################
# Samplers
###############################################################################


@dataclass
class McmcChain:
    """One chain of an MCMC sampler, in sampling space."""

    samples: Array
    log_density: Array
    acceptance_rate: float


class McmcSampler(Protocol):
    """Draws `n_samples` from an (unnormalised) log density, starting at `z0`."""

    def __call__(
        self,
        log_density: Callable[[Array], float],
        z0: Array,
        n_samples: int,
        rng: np.random.Generator,
    ) -> McmcChain:
        """Run one chain."""
        ...


@dataclass
class AdaptiveMetropolis:
    """Adaptive Metropolis sampler (Haario et al. 2001).

    During `n_warmup` steps, the Gaussian proposal's covariance is set every
    `adaptation_interval` steps to `2.38² / d` times the covariance of the
    chain so far (regularised by `epsilon`). The warm-up is then discarded, and
    the samples are drawn with the proposal frozen, so they form a proper
    Markov chain.

    Parameters
    ----------
    n_warmup
        Number of adaptation steps, defaults to the number of samples
    adaptation_interval
        Steps between two covariance updates
    initial_scale
        Standard deviation of the initial proposal per dimension (in sampling
        space, i.e. relative for log-sampled parameters)
    epsilon
        Regularisation added to the diagonal of the adapted covariance

    """

    n_warmup: int | None = None
    adaptation_interval: int = 100
    initial_scale: float = 0.1
    epsilon: float = 1e-10

    def __call__(
        self,
        log_density: Callable[[Array], float],
        z0: Array,
        n_samples: int,
        rng: np.random.Generator,
    ) -> McmcChain:
        """Run one chain."""
        n_warmup = n_samples if self.n_warmup is None else self.n_warmup
        dim = len(z0)
        cov = np.eye(dim) * self.initial_scale**2
        chol = np.linalg.cholesky(cov)

        z, lp = np.asarray(z0, dtype=float), log_density(z0)
        warmup = np.empty((n_warmup, dim))
        samples = np.empty((n_samples, dim))
        densities = np.empty(n_samples)
        accepted = 0

        for step in range(n_warmup + n_samples):
            proposal = z + chol @ rng.standard_normal(dim)
            lp_proposal = log_density(proposal)
            is_accepted = np.log(rng.uniform()) < lp_proposal - lp
            if is_accepted:
                z, lp = proposal, lp_proposal

            if step < n_warmup:
                warmup[step] = z
                if (step + 1) % self.adaptation_interval == 0:
                    chol = self._adapt(warmup[: step + 1], chol)
            else:
                i = step - n_warmup
                samples[i] = z
                densities[i] = lp
                accepted += int(is_accepted)

        return McmcChain(
            samples=samples,
            log_density=densities,
            acceptance_rate=accepted / n_samples if n_samples else 0.0,
        )

    def _adapt(self, history: Array, chol: Array) -> Array:
        dim = history.shape[1]
        empirical = np.atleast_2d(np.cov(history, rowvar=False))
        if np.trace(empirical) <= self.epsilon * dim:
            # Nothing accepted yet: shrink the proposal instead
            return chol * 0.5
        cov = 2.38**2 / dim * (empirical + self.epsilon * np.eye(dim))
        try:
            return np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return chol


###############################################################################
# Result
###############################################################################


def _split_chains(x: Array) -> Array:
    """Split each chain (rows) into two halves, dropping a middle draw if odd."""
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, -half:]]) if half else x


def _rank_normalise(x: Array) -> Array:
    ranks = stats.rankdata(x, method="average").reshape(x.shape)
    return np.asarray(stats.norm.ppf((ranks - 3 / 8) / (x.size + 1 / 4)))


def _basic_r_hat(x: Array) -> float:
    n = x.shape[1]
    within = np.mean(np.var(x, axis=1, ddof=1))
    between = np.var(np.mean(x, axis=1), ddof=1)
    if within == 0:
        return np.nan
    return float(np.sqrt(((n - 1) / n * within + between) / within))


def _r_hat(x: Array) -> float:
    """Rank-normalised split R-hat (Vehtari et al. 2021) of chains (rows) of draws."""
    split = _split_chains(x)
    if split.shape[0] < 2 or split.shape[1] < 2:  # noqa: PLR2004
        return np.nan
    bulk = _basic_r_hat(_rank_normalise(split))
    folded = _basic_r_hat(_rank_normalise(np.abs(split - np.median(split))))
    return max(bulk, folded)


def _autocov(x: Array) -> Array:
    n = x.shape[1]
    centered = x - x.mean(axis=1, keepdims=True)
    size = 2 ** int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(centered, size, axis=1)
    return np.fft.irfft(f * np.conj(f), size, axis=1)[:, :n] / n


def _basic_ess(x: Array) -> float:
    """Effective sample size via Geyer's initial monotone sequence."""
    n_chain, n_draw = x.shape
    if n_draw < 4:  # noqa: PLR2004
        return np.nan
    acov = _autocov(x)
    mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1)
    var_plus = mean_var * (n_draw - 1) / n_draw
    if n_chain > 1:
        var_plus += np.var(np.mean(x, axis=1), ddof=1)
    if var_plus == 0:
        return np.nan

    rho = np.zeros(n_draw)
    rho[0] = rho_even = 1.0
    rho[1] = rho_odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
    t = 1
    while t < n_draw - 3 and rho_even + rho_odd > 0:
        rho_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if rho_even + rho_odd >= 0:
            rho[t + 1], rho[t + 2] = rho_even, rho_odd
        t += 2
    max_t = t - 2
    if rho_even > 0:
        rho[max_t + 1] = rho_even
    # Initial monotone sequence
    t = 1
    while t <= max_t - 2:
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = rho[t + 2] = (rho[t - 1] + rho[t]) / 2
        t += 2
    total = n_chain * n_draw
    tau = -1.0 + 2.0 * np.sum(rho[: max_t + 1]) + np.sum(rho[max_t + 1 : max_t + 2])
    return float(total / max(tau, 1 / np.log10(total)))


def _ess(x: Array) -> float:
    """Bulk effective sample size (Vehtari et al. 2021) of chains (rows) of draws."""
    return _basic_ess(_rank_normalise(_split_chains(x)))


@dataclass
class McmcResult:
    """Posterior samples of an MCMC run, in natural units.

    Attributes
    ----------
    parameter_names
        Names of the sampled parameters, the last axis of `chains`
    chains
        Samples, shape (n_chains, n_samples, n_parameters)
    log_posterior
        Unnormalised log posterior density of each sample, shape (n_chains, n_samples)
    acceptance_rates
        Fraction of accepted proposals per chain

    """

    parameter_names: list[str]
    chains: Array
    log_posterior: Array
    acceptance_rates: Array

    def __repr__(self) -> str:
        """Return default representation."""
        return pformat(self)

    def samples(self) -> pd.DataFrame:
        """All samples, one row per (chain, draw)."""
        n_chains, n_samples, n_pars = self.chains.shape
        return pd.DataFrame(
            self.chains.reshape(n_chains * n_samples, n_pars),
            index=pd.MultiIndex.from_product(
                [range(n_chains), range(n_samples)], names=["chain", "draw"]
            ),
            columns=self.parameter_names,
        )

    def credible_intervals(self, level: float = 0.95) -> pd.DataFrame:
        """Equal-tailed credible intervals containing `level` of the posterior mass.

        Returns
        -------
        pd.DataFrame
            One row per parameter, with columns `lower`, `median` and `upper`

        """
        tail = (1 - level) / 2
        q = self.samples().quantile([tail, 0.5, 1 - tail]).T
        q.columns = pd.Index(["lower", "median", "upper"])
        return q

    def r_hat(self) -> pd.Series:
        """Rank-normalised split R-hat per parameter; below ~1.01 suggests convergence."""
        return pd.Series(
            [_r_hat(self.chains[:, :, i]) for i in range(len(self.parameter_names))],
            index=self.parameter_names,
        )

    def ess(self) -> pd.Series:
        """Bulk effective sample size per parameter."""
        return pd.Series(
            [_ess(self.chains[:, :, i]) for i in range(len(self.parameter_names))],
            index=self.parameter_names,
        )

    def plot_traces(self) -> FigAxs:
        """Plot the samples of every chain over the draws, one axes per parameter."""
        from mxlpy.plot import grid_layout  # noqa: PLC0415

        fig, axs = grid_layout(len(self.parameter_names), n_cols=1, col_width=8)
        for i, (name, ax) in enumerate(zip(self.parameter_names, axs, strict=False)):
            for chain in range(self.chains.shape[0]):
                ax.plot(self.chains[chain, :, i], lw=0.5, alpha=0.8)
            ax.set_ylabel(name)
        list(axs)[len(self.parameter_names) - 1].set_xlabel("draw")
        return fig, axs

    def plot_marginals(self, bins: int = 50) -> FigAxs:
        """Plot a histogram of the samples of all chains, one axes per parameter."""
        from mxlpy.plot import grid_layout  # noqa: PLC0415

        fig, axs = grid_layout(len(self.parameter_names), sharex=False)
        samples = self.samples()
        for name, ax in zip(self.parameter_names, axs, strict=False):
            ax.hist(samples[name], bins=bins, density=True)
            ax.set_xlabel(name)
        return fig, axs


###############################################################################
# Posterior
###############################################################################


def _sigma_array(sigma: Sigma, data: pd.Series | pd.DataFrame) -> Array:
    """`sigma` broadcast to the shape of `data`."""
    if isinstance(sigma, pd.DataFrame):
        if isinstance(data, pd.Series):
            msg = "sigma per data point (a DataFrame) needs time course data"
            raise ValueError(msg)  # noqa: TRY004, invalid input like the others
        values = sigma.reindex(index=data.index, columns=data.columns).to_numpy()
    elif isinstance(sigma, pd.Series):
        keys = data.index if isinstance(data, pd.Series) else data.columns
        values = np.broadcast_to(sigma.reindex(keys).to_numpy(), data.shape)
    else:
        values = np.full(data.shape, float(sigma))
    values = np.asarray(values, dtype=float)
    observed = np.isfinite(data.to_numpy(dtype=float))
    if not (np.isfinite(values[observed]).all() and (values[observed] > 0).all()):
        msg = "sigma has to be positive and finite for every observed data point"
        raise ValueError(msg)
    return values


@dataclass
class _Posterior:
    """Log posterior density in sampling space; picklable for parallel chains."""

    settings: _Settings[Any]
    simulate: Callable[[_Settings[Any]], Result[Any]]
    names: list[str]
    priors: list[Prior]
    log_scaled: NDArray[np.bool_]
    observed: NDArray[np.bool_]
    data: Array
    inv_sigma: Array
    log_norm: float

    def to_natural(self, z: Array) -> Array:
        theta = np.array(z, dtype=float)
        theta[..., self.log_scaled] = np.exp(theta[..., self.log_scaled])
        return theta

    def to_sampling(self, theta: Array) -> Array:
        z = np.array(theta, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z[..., self.log_scaled] = np.log(z[..., self.log_scaled])
        return z

    def log_prior(self, theta: Array) -> float:
        return float(
            sum(p.log_pdf(float(t)) for p, t in zip(self.priors, theta, strict=True))
        )

    def log_likelihood(self, theta: Array) -> float:
        _apply_updates(
            dict(zip(self.names, theta.tolist(), strict=True)), self.settings
        )
        match res := self.simulate(self.settings).value:
            case Exception():
                return -np.inf
            case _:
                prediction = res.get_combined()
        if (postprocess := self.settings.postprocess_fn) is not None:
            prediction = postprocess(prediction)
        # Aligned by label, like the fit residuals; anything the prediction
        # lacks becomes NaN and makes the likelihood -inf
        data = self.settings.data
        if isinstance(data, pd.Series):
            pred = prediction.iloc[-1].reindex(data.index).to_numpy(dtype=float)
        else:
            pred = prediction.reindex(index=data.index, columns=data.columns).to_numpy(
                dtype=float
            )
        residual = (pred - self.data)[self.observed] * self.inv_sigma[self.observed]
        if not np.isfinite(residual).all():
            return -np.inf
        return self.log_norm - 0.5 * float(np.sum(residual**2))

    def log_posterior(self, theta: Array) -> float:
        if not np.isfinite(lp := self.log_prior(theta)):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def __call__(self, z: Array) -> float:
        """Log posterior of the natural parameters, plus the log-transform's Jacobian."""
        theta = self.to_natural(z)
        if not np.isfinite(lp := self.log_posterior(theta)):
            return -np.inf
        return lp + float(np.sum(z[self.log_scaled]))


def _make_posterior(
    settings: _Settings[Any],
    simulate: Callable[[_Settings[Any]], Result[Any]],
    prior: Mapping[str, Prior],
    sigma: Sigma,
) -> _Posterior:
    data = settings.data
    values = data.to_numpy(dtype=float)
    observed = np.isfinite(values)
    sigma_values = _sigma_array(sigma, data)
    return _Posterior(
        settings=settings,
        simulate=simulate,
        names=list(prior),
        priors=list(prior.values()),
        log_scaled=np.array([p.positive_support for p in prior.values()]),
        observed=observed,
        data=values,
        inv_sigma=np.where(observed, 1 / np.where(observed, sigma_values, 1.0), 0.0),
        log_norm=float(
            -np.sum(np.log(sigma_values[observed]))
            - 0.5 * observed.sum() * np.log(2 * np.pi)
        ),
    )


def _run_chain(
    inp: tuple[Array, np.random.SeedSequence],
    *,
    posterior: _Posterior,
    sampler: McmcSampler,
    n_samples: int,
) -> McmcChain:
    z0, seed = inp
    rng = np.random.default_rng(seed)
    # Spread the starts, so that R-hat can detect chains that don't mix
    start = z0
    for _ in range(_START_ATTEMPTS):
        candidate = z0 + _START_JITTER * rng.standard_normal(len(z0)) * np.where(
            posterior.log_scaled, 1.0, np.maximum(np.abs(z0), 1e-12)
        )
        if np.isfinite(posterior(candidate)):
            start = candidate
            break
    return sampler(posterior, start, n_samples, rng)


def _sample(
    settings: _Settings[Any],
    simulate: Callable[[_Settings[Any]], Result[Any]],
    *,
    prior: Mapping[str, Prior],
    sigma: Sigma,
    x0: dict[str, float] | None,
    n_samples: int,
    n_chains: int,
    sampler: McmcSampler | None,
    seed: int | None,
    parallel: bool,
    max_workers: int | None,
    timeout: float | None,
) -> Result[McmcResult]:
    if not prior:
        return Result(ValueError("prior is empty, so there is nothing to sample"))
    if unknown := set(x0 or {}) - set(prior):
        return Result(ValueError(f"x0 has entries without a prior: {sorted(unknown)}"))

    model = settings.model
    current = model.get_parameter_values() | (
        {}
        if isinstance(model, SteadyStateModelBuilder)
        else dict(model.get_initial_conditions()) | (settings.y0 or {})
    )
    try:
        theta0 = np.array([(x0 or {}).get(k, current[k]) for k in prior], dtype=float)
    except KeyError as e:
        return Result(
            ValueError(f"{e.args[0]!r} is neither a parameter nor a variable")
        )

    try:
        posterior = _make_posterior(settings, simulate, prior, sigma)
        z0 = posterior.to_sampling(theta0)
        start_density = posterior(z0)
    except Exception as e:  # noqa: BLE001
        return Result(e)
    if not np.isfinite(start_density):
        msg = (
            "The log posterior at the starting point isn't finite: x0 is outside a "
            "prior's support (or on the boundary of a log-sampled one), the model "
            "can't be simulated there, or its output lacks data entries"
        )
        return Result(ValueError(msg))

    seeds = np.random.SeedSequence(seed).spawn(n_chains)
    chains = [
        chain
        for _, chain in sorted(
            parallelise(
                partial(
                    _run_chain,
                    posterior=posterior,
                    sampler=AdaptiveMetropolis() if sampler is None else sampler,
                    n_samples=n_samples,
                ),
                [(i, (z0, s)) for i, s in enumerate(seeds)],
                parallel=parallel,
                max_workers=max_workers,
                timeout=timeout,
                disable_tqdm=True,
            ),
            key=lambda x: x[0],
        )
    ]
    if not chains:
        return Result(TimeoutError("No chain finished within the timeout"))

    jacobian = np.array(
        [c.samples[:, posterior.log_scaled].sum(axis=1) for c in chains]
    )
    return Result(
        McmcResult(
            parameter_names=posterior.names,
            chains=np.array([posterior.to_natural(c.samples) for c in chains]),
            log_posterior=np.array([c.log_density for c in chains]) - jacobian,
            acceptance_rates=np.array([c.acceptance_rate for c in chains]),
        )
    )


def _settings[M: _Model](
    model: M,
    *,
    data: pd.Series | pd.DataFrame,
    prior: Mapping[str, Prior],
    y0: dict[str, float] | None,
    runner: SteadyStateRunner[M, Any] | None,
    postprocess_fn: PostprocessFn | None,
    protocol: pd.DataFrame | None,
    as_deepcopy: bool,
) -> _Settings[M]:
    if as_deepcopy:
        model = deepcopy(model)
    p_names, v_names = _split_p0(model, dict.fromkeys(prior, 0.0))
    return _Settings(
        model=model,
        data=data,
        y0=y0,
        runner=default_runner(model) if runner is None else runner,
        loss_fn=losses.rmse,  # unused: the likelihood replaces the loss
        p_names=p_names,
        v_names=v_names,
        standard_scale=False,
        protocol=protocol,
        postprocess_fn=postprocess_fn,
    )


def _simulate_steady_state(settings: _Settings[Any]) -> Result[Any]:
    return settings.runner.steady_state(settings.model)


def _simulate_time_course(settings: _Settings[Any]) -> Result[Any]:
    runner: DynamicRunner[Any, Any] = settings.runner  # type: ignore[assignment]
    return runner.time_course(settings.model, list(settings.data.index))


def _simulate_protocol_time_course(settings: _Settings[Any]) -> Result[Any]:
    runner: DynamicRunner[Any, Any] = settings.runner  # type: ignore[assignment]
    return runner.protocol_time_course(
        settings.model,
        settings.protocol,  # type: ignore[arg-type]
        settings.data.index,
    )


@overload
def mcmc_steady_state(
    model: KineticModelBuilder,
    *,
    data: pd.Series,
    prior: Mapping[str, Prior],
    sigma: Sigma,
    n_samples: int = 1000,
    n_chains: int = 4,
    sampler: McmcSampler | None = None,
    x0: dict[str, float] | None = None,
    y0: dict[str, float] | None = None,
    runner: SteadyStateRunner[KineticModelBuilder, Simulation] | None = None,
    postprocess_fn: PostprocessFn | None = None,
    seed: int | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    timeout: float | None = None,
    as_deepcopy: bool = True,
) -> Result[McmcResult]: ...


@overload
def mcmc_steady_state(
    model: OdeModelBuilder,
    *,
    data: pd.Series,
    prior: Mapping[str, Prior],
    sigma: Sigma,
    n_samples: int = 1000,
    n_chains: int = 4,
    sampler: McmcSampler | None = None,
    x0: dict[str, float] | None = None,
    y0: dict[str, float] | None = None,
    runner: SteadyStateRunner[OdeModelBuilder, OdeSimulation] | None = None,
    postprocess_fn: PostprocessFn | None = None,
    seed: int | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    timeout: float | None = None,
    as_deepcopy: bool = True,
) -> Result[McmcResult]: ...


@overload
def mcmc_steady_state(
    model: SteadyStateModelBuilder,
    *,
    data: pd.Series,
    prior: Mapping[str, Prior],
    sigma: Sigma,
    n_samples: int = 1000,
    n_chains: int = 4,
    sampler: McmcSampler | None = None,
    x0: dict[str, float] | None = None,
    runner: SteadyStateRunner[SteadyStateModelBuilder, Evaluation] | None = None,
    postprocess_fn: PostprocessFn | None = None,
    seed: int | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    timeout: float | None = None,
    as_deepcopy: bool = True,
) -> Result[McmcResult]: ...


def mcmc_steady_state[M: _Model](
    model: M,
    *,
    data: pd.Series,
    prior: Mapping[str, Prior],
    sigma: Sigma,
    n_samples: int = 1000,
    n_chains: int = 4,
    sampler: McmcSampler | None = None,
    x0: dict[str, float] | None = None,
    y0: dict[str, float] | None = None,
    runner: SteadyStateRunner[M, Any] | None = None,
    postprocess_fn: PostprocessFn | None = None,
    seed: int | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    timeout: float | None = None,
    as_deepcopy: bool = True,
) -> Result[McmcResult]:
    """Sample the posterior of model parameters given steady-state data.

    Parameters
    ----------
    model
        Model whose parameters (or initial conditions) are sampled
    data
        Experimental data. Missing values (NaN) are ignored.
    prior
        Prior distribution of every sampled parameter (or variable), e.g.
        `{"k1": LogNormal(0, 1)}`. Its keys are the sampled names.
    sigma
        Standard deviation of the Gaussian measurement noise: one for all data,
        a Series per column (time courses) or entry (steady states), or a
        DataFrame per data point
    n_samples
        Number of samples per chain, after the sampler's warm-up
    n_chains
        Number of independent chains
    sampler
        MCMC sampler, defaults to `AdaptiveMetropolis()`
    x0
        Starting point of the chains, defaults to the model's current values
    y0
        Initial conditions as {variable_name: value}
    runner
        How the model is simulated, see `mxlpy.runners`
    postprocess_fn
        Applied to the combined simulation result before it is compared with
        the data. Its output has to contain the data columns.
    seed
        Seed of the chains' random number generators, for reproducible samples
    parallel
        Whether to run the chains in parallel
    max_workers
        Maximum number of parallel workers
    timeout
        Timeout in seconds for each parallel worker; chains that don't finish
        are dropped. Ignored if not `parallel`.
    as_deepcopy
        Whether to copy the model to avoid overwriting its state

    Returns
    -------
        Result containing the posterior samples

    """
    _check_y0(model, y0)
    return _sample(
        _settings(
            model,
            data=data,
            prior=prior,
            y0=y0,
            runner=runner,
            postprocess_fn=postprocess_fn,
            protocol=None,
            as_deepcopy=as_deepcopy,
        ),
        _simulate_steady_state,
        prior=prior,
        sigma=sigma,
        x0=x0,
        n_samples=n_samples,
        n_chains=n_chains,
        sampler=sampler,
        seed=seed,
        parallel=parallel,
        max_workers=max_workers,
        timeout=timeout,
    )


def mcmc_time_course[M: KineticModelBuilder | OdeModelBuilder](
    model: M,
    *,
    data: pd.DataFrame,
    prior: Mapping[str, Prior],
    sigma: Sigma,
    n_samples: int = 1000,
    n_chains: int = 4,
    sampler: McmcSampler | None = None,
    x0: dict[str, float] | None = None,
    y0: dict[str, float] | None = None,
    runner: DynamicRunner[M, Any] | None = None,
    postprocess_fn: PostprocessFn | None = None,
    seed: int | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    timeout: float | None = None,
    as_deepcopy: bool = True,
) -> Result[McmcResult]:
    """Sample the posterior of model parameters given time course data.

    Examples
    --------
        >>> res = fit.mcmc_time_course(
        ...     model,
        ...     data=data,
        ...     prior={"k1": LogNormal(0, 1)},
        ...     sigma=0.1,
        ... ).unwrap_or_err()
        >>> res.credible_intervals()

    Parameters
    ----------
    model
        Model whose parameters (or initial conditions) are sampled
    data
        Experimental data. Missing values (NaN) are ignored.
    prior
        Prior distribution of every sampled parameter (or variable), e.g.
        `{"k1": LogNormal(0, 1)}`. Its keys are the sampled names.
    sigma
        Standard deviation of the Gaussian measurement noise: one for all data,
        a Series per column (time courses) or entry (steady states), or a
        DataFrame per data point
    n_samples
        Number of samples per chain, after the sampler's warm-up
    n_chains
        Number of independent chains
    sampler
        MCMC sampler, defaults to `AdaptiveMetropolis()`
    x0
        Starting point of the chains, defaults to the model's current values
    y0
        Initial conditions as {variable_name: value}
    runner
        How the model is simulated, see `mxlpy.runners`
    postprocess_fn
        Applied to the combined simulation result before it is compared with
        the data. Its output has to contain the data columns.
    seed
        Seed of the chains' random number generators, for reproducible samples
    parallel
        Whether to run the chains in parallel
    max_workers
        Maximum number of parallel workers
    timeout
        Timeout in seconds for each parallel worker; chains that don't finish
        are dropped. Ignored if not `parallel`.
    as_deepcopy
        Whether to copy the model to avoid overwriting its state

    Returns
    -------
        Result containing the posterior samples

    """
    return _sample(
        _settings(
            model,
            data=data,
            prior=prior,
            y0=y0,
            runner=runner,
            postprocess_fn=postprocess_fn,
            protocol=None,
            as_deepcopy=as_deepcopy,
        ),
        _simulate_time_course,
        prior=prior,
        sigma=sigma,
        x0=x0,
        n_samples=n_samples,
        n_chains=n_chains,
        sampler=sampler,
        seed=seed,
        parallel=parallel,
        max_workers=max_workers,
        timeout=timeout,
    )


def mcmc_protocol_time_course[M: KineticModelBuilder | OdeModelBuilder](
    model: M,
    *,
    data: pd.DataFrame,
    protocol: pd.DataFrame,
    prior: Mapping[str, Prior],
    sigma: Sigma,
    n_samples: int = 1000,
    n_chains: int = 4,
    sampler: McmcSampler | None = None,
    x0: dict[str, float] | None = None,
    y0: dict[str, float] | None = None,
    runner: DynamicRunner[M, Any] | None = None,
    postprocess_fn: PostprocessFn | None = None,
    seed: int | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
    timeout: float | None = None,
    as_deepcopy: bool = True,
) -> Result[McmcResult]:
    """Sample the posterior of model parameters given protocol time course data.

    Time points of the protocol time course are taken from the data.

    Parameters
    ----------
    model
        Model whose parameters (or initial conditions) are sampled
    data
        Experimental data. Missing values (NaN) are ignored.
    protocol
        Protocol of parameter changes over time
    prior
        Prior distribution of every sampled parameter (or variable), e.g.
        `{"k1": LogNormal(0, 1)}`. Its keys are the sampled names.
    sigma
        Standard deviation of the Gaussian measurement noise: one for all data,
        a Series per column (time courses) or entry (steady states), or a
        DataFrame per data point
    n_samples
        Number of samples per chain, after the sampler's warm-up
    n_chains
        Number of independent chains
    sampler
        MCMC sampler, defaults to `AdaptiveMetropolis()`
    x0
        Starting point of the chains, defaults to the model's current values
    y0
        Initial conditions as {variable_name: value}
    runner
        How the model is simulated, see `mxlpy.runners`
    postprocess_fn
        Applied to the combined simulation result before it is compared with
        the data. Its output has to contain the data columns.
    seed
        Seed of the chains' random number generators, for reproducible samples
    parallel
        Whether to run the chains in parallel
    max_workers
        Maximum number of parallel workers
    timeout
        Timeout in seconds for each parallel worker; chains that don't finish
        are dropped. Ignored if not `parallel`.
    as_deepcopy
        Whether to copy the model to avoid overwriting its state

    Returns
    -------
        Result containing the posterior samples

    """
    return _sample(
        _settings(
            model,
            data=data,
            prior=prior,
            y0=y0,
            runner=runner,
            postprocess_fn=postprocess_fn,
            protocol=_normalise_protocol_index(protocol),
            as_deepcopy=as_deepcopy,
        ),
        _simulate_protocol_time_course,
        prior=prior,
        sigma=sigma,
        x0=x0,
        n_samples=n_samples,
        n_chains=n_chains,
        sampler=sampler,
        seed=seed,
        parallel=parallel,
        max_workers=max_workers,
        timeout=timeout,
    )
