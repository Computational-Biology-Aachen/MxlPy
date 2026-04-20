"""Utility fns for integrators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.signal import find_peaks  # type: ignore[import-untyped]

from mxlpy.types import OscillationDetected

__all__ = [
    "DetectOscillations",
    "DetectOscillationsScipY",
    "NoOscillationDetection",
    "OscillationDetector",
]

# Minimum number of elements in the autocorrelation search window required to
# identify a local maximum: the sliding-window comparison `search[1:-1] > …`
# needs at least one interior element, so search must contain at least 3 items.
_MIN_PEAK_SEARCH_LEN = 3
# Minimum number of peaks required to estimate a period: need at least two
# consecutive peaks to compute an inter-peak interval.
_MIN_PEAKS_FOR_PERIOD = 2


class OscillationDetector(Protocol):
    """Protocol for oscillation detection callables.

    Any callable matching this signature can be passed as an oscillation
    detector to integrators and simulators.

    Examples
    --------
    Use the built-in autocorrelation detector (default)::

        Simulator(model).simulate_to_steady_state(
            oscillation_detector=DetectOscillations()
        )

    Customise detection sensitivity::

        Simulator(model).simulate_to_steady_state(
            oscillation_detector=DetectOscillations(autocorr_threshold=0.5)
        )

    Disable detection entirely::

        Simulator(model).simulate_to_steady_state(
            oscillation_detector=NoOscillationDetection()
        )

    Use a scipy peak-finding detector::

        Simulator(model).simulate_to_steady_state(
            oscillation_detector=DetectOscillationsScipY()
        )

    """

    def __call__(
        self,
        history: np.ndarray,
        variable_names: list[str],
        *,
        times: np.ndarray | None = None,
    ) -> OscillationDetected | None:
        """Detect oscillations in a trajectory.

        Parameters
        ----------
        history
            Array of shape ``(N, n_vars)`` containing variable values at *N*
            consecutive time points.
        variable_names
            Names of the *n_vars* variables.
        times
            Optional 1-D array of length *N* giving the simulation time at
            each sample.

        Returns
        -------
        OscillationDetected | None
            An :class:`~mxlpy.types.OscillationDetected` exception if
            oscillation is detected, otherwise ``None``.

        """
        ...


@dataclass(frozen=True)
class NoOscillationDetection:
    """No-op oscillation detector that never reports oscillations.

    Use this variant when oscillation detection is not desired, e.g. for
    fitting routines where the model is known to converge or where the
    overhead of detection should be avoided.

    Examples
    --------
    >>> Simulator(model).simulate_to_steady_state(
    ...     oscillation_detector=NoOscillationDetection()
    ... )

    """

    def __call__(
        self,
        history: np.ndarray,  # noqa: ARG002
        variable_names: list[str],  # noqa: ARG002
        *,
        times: np.ndarray | None = None,  # noqa: ARG002
    ) -> OscillationDetected | None:
        """Return ``None`` unconditionally."""
        return None


@dataclass(frozen=True)
class DetectOscillations:
    """Autocorrelation-based oscillation detector.

    Analyses a trajectory by computing the normalised autocorrelation of each
    variable.  A significant secondary peak (above *autocorr_threshold*) at
    lag >= *min_lag* is treated as evidence of sustained oscillation.

    Attributes
    ----------
    min_lag
        Minimum lag (in samples) at which a secondary autocorrelation peak is
        considered.  Small values prevent spurious detections caused by
        step-to-step correlation.  Default: 2.
    autocorr_threshold
        Minimum normalised autocorrelation value for a secondary peak to be
        counted as evidence of oscillation (0 < threshold < 1).  Default: 0.3.
    min_samples
        Minimum number of samples required to attempt detection.  Trajectories
        shorter than this are considered insufficient to distinguish oscillation
        from noise and ``None`` is returned.  Default: 10.
    flat_signal_threshold
        Minimum standard deviation a signal must have to be analysed.
        Variables whose standard deviation falls below this value are treated
        as constant and skipped.  Default: 1e-10.

    Examples
    --------
    >>> detector = DetectOscillations(autocorr_threshold=0.5)
    >>> Simulator(model).simulate_to_steady_state(
    ...     oscillation_detector=detector
    ... )

    """

    min_lag: int = 2
    autocorr_threshold: float = 0.3
    min_samples: int = 10
    flat_signal_threshold: float = 1e-6

    def __call__(
        self,
        history: np.ndarray,
        variable_names: list[str],
        *,
        times: np.ndarray | None = None,
    ) -> OscillationDetected | None:
        """Detect oscillatory behaviour in a trajectory of variable values.

        Parameters
        ----------
        history
            Array of shape ``(N, n_vars)`` containing variable values at *N*
            consecutive time points.
        variable_names
            Names of the *n_vars* variables.  These are reported directly in
            the returned exception.
        times
            Optional 1-D array of length *N* giving the simulation time at
            each sample.  When provided, the oscillation period is estimated
            in physical time units; otherwise it is expressed in samples.

        Returns
        -------
        OscillationDetected | None
            An :class:`~mxlpy.types.OscillationDetected` exception if at
            least one variable is found to oscillate, otherwise ``None``.

        """
        n = len(history)
        if n < self.min_samples:
            return None

        oscillating: list[str] = []
        periods: list[float] = []

        for i, name in enumerate(variable_names):
            y = history[:, i].astype(float)
            std = float(np.std(y))
            if std < self.flat_signal_threshold:
                continue  # flat signal - not oscillating

            y_c = (y - float(np.mean(y))) / std  # zero-mean, unit-variance

            # Full autocorrelation; keep only non-negative lags
            ac_full = np.correlate(y_c, y_c, mode="full")
            ac = ac_full[n - 1 :]  # shape (N,)
            if ac[0] == 0:
                continue
            ac = ac / ac[0]  # normalise: lag-0 == 1

            # Search for a secondary peak at lag >= min_lag
            search = ac[self.min_lag :]
            if len(search) < _MIN_PEAK_SEARCH_LEN:
                continue

            is_local_max = (search[1:-1] > search[:-2]) & (search[1:-1] > search[2:])
            peak_idx = np.where(is_local_max)[0] + self.min_lag + 1  # index into `ac`

            if len(peak_idx) == 0:
                continue

            best_lag = int(peak_idx[np.argmax(ac[peak_idx])])
            if ac[best_lag] >= self.autocorr_threshold:
                oscillating.append(name)
                if times is not None and len(times) == n and n > 1:
                    dt = float(times[-1] - times[0]) / (n - 1)
                    periods.append(best_lag * dt)

        if not oscillating:
            return None

        period: float | None = float(np.mean(periods)) if periods else None
        return OscillationDetected(oscillating_species=oscillating, period=period)


@dataclass(frozen=True)
class DetectOscillationsScipY:
    """Peak-finding oscillation detector using ``scipy.signal.find_peaks``.

    Identifies local maxima in each variable's time series.  A variable is
    considered oscillating when it has at least *min_peaks* prominent peaks.
    Peak prominence is required to exceed *min_prominence_factor* times the
    signal's standard deviation, making the threshold scale-invariant.

    Attributes
    ----------
    min_peaks
        Minimum number of peaks required to classify a variable as
        oscillating.  Default: 2.
    min_prominence_factor
        Minimum peak prominence expressed as a fraction of the signal's
        standard deviation.  Prevents noise spikes from triggering
        detection.  Default: 0.1.
    min_samples
        Minimum number of samples required to attempt detection.  Default: 10.
    flat_signal_threshold
        Minimum standard deviation a signal must have to be analysed.
        Default: 1e-10.

    Examples
    --------
    >>> detector = DetectOscillationsScipY(min_peaks=3)
    >>> Simulator(model).simulate_to_steady_state(
    ...     oscillation_detector=detector
    ... )

    """

    min_peaks: int = 2
    min_prominence_factor: float = 0.1
    min_samples: int = 10
    flat_signal_threshold: float = 1e-6

    def __call__(
        self,
        history: np.ndarray,
        variable_names: list[str],
        *,
        times: np.ndarray | None = None,
    ) -> OscillationDetected | None:
        """Detect oscillatory behaviour using ``scipy.signal.find_peaks``.

        Parameters
        ----------
        history
            Array of shape ``(N, n_vars)`` containing variable values at *N*
            consecutive time points.
        variable_names
            Names of the *n_vars* variables.  These are reported directly in
            the returned exception.
        times
            Optional 1-D array of length *N* giving the simulation time at
            each sample.  When provided, the oscillation period is estimated
            as the mean inter-peak interval in physical time units.

        Returns
        -------
        OscillationDetected | None
            An :class:`~mxlpy.types.OscillationDetected` exception if at
            least one variable is found to oscillate, otherwise ``None``.

        """
        n = len(history)
        if n < self.min_samples:
            return None

        oscillating: list[str] = []
        periods: list[float] = []

        for i, name in enumerate(variable_names):
            y = history[:, i].astype(float)
            std = float(np.std(y))
            if std < self.flat_signal_threshold:
                continue  # flat signal - not oscillating

            prominence = self.min_prominence_factor * std
            peaks, _ = find_peaks(y, prominence=prominence)
            if len(peaks) < self.min_peaks:
                continue

            oscillating.append(name)
            if times is not None and len(peaks) >= _MIN_PEAKS_FOR_PERIOD:
                peak_times = times[peaks]
                periods.append(float(np.mean(np.diff(peak_times))))

        if not oscillating:
            return None

        period: float | None = float(np.mean(periods)) if periods else None
        return OscillationDetected(oscillating_species=oscillating, period=period)


# Module-level instances for backward compatibility and use as default values
# in function signatures throughout the codebase.
no_oscillation_detection = NoOscillationDetection()
detect_oscillations = DetectOscillations()
detect_oscillations_scipy = DetectOscillationsScipY()
