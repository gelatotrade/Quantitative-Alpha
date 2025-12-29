"""
Lyapunov Exponent Analysis for Financial Time Series

The Lyapunov exponent quantifies the rate of separation of infinitesimally
close trajectories in phase space. For chaotic systems, the largest Lyapunov
exponent (LLE) is positive, indicating sensitive dependence on initial conditions.

Financial Interpretation:
    - Positive LLE: Chaotic dynamics, limited predictability horizon
    - Near-zero LLE: Quasi-periodic behavior, longer predictability
    - Negative LLE: Stable dynamics, strong mean reversion

The predictability horizon can be estimated as τ ≈ 1/λ where λ is the LLE.
"""

import numpy as np
from scipy.linalg import qr
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class LyapunovSpectrum:
    """
    Complete Lyapunov spectrum of a dynamical system.

    Attributes:
        exponents: Array of Lyapunov exponents (ordered decreasing)
        dimension_kaplan_yorke: Kaplan-Yorke dimension
        entropy_kolmogorov_sinai: KS entropy (sum of positive exponents)
        predictability_horizon: Characteristic prediction time scale
    """
    exponents: np.ndarray
    dimension_kaplan_yorke: float
    entropy_kolmogorov_sinai: float
    predictability_horizon: float


class LyapunovExponent:
    """
    Lyapunov Exponent Calculator for Time Series and Dynamical Systems.

    Provides multiple methods for estimating Lyapunov exponents:
    1. Direct method for known ODEs (using variational equations)
    2. Wolf algorithm for time series
    3. Rosenstein algorithm for time series

    Example:
        >>> lyap = LyapunovExponent()
        >>> lle = lyap.largest_from_series(returns, dt=1/252)
        >>> print(f"Predictability horizon: {1/lle:.1f} days")
    """

    def __init__(self, dt: float = 1.0):
        """
        Initialize the calculator.

        Args:
            dt: Time step between observations
        """
        self.dt = dt

    def from_ode(
        self,
        equations: callable,
        jacobian: callable,
        initial_state: np.ndarray,
        t_total: float = 1000.0,
        t_transient: float = 100.0,
        n_steps: int = 10000
    ) -> LyapunovSpectrum:
        """
        Calculate full Lyapunov spectrum for a known ODE system.

        Uses QR decomposition method with variational equations.

        Args:
            equations: Function f(x, t) returning dx/dt
            jacobian: Function J(x) returning Jacobian matrix
            initial_state: Initial condition
            t_total: Total integration time
            t_transient: Transient time to discard
            n_steps: Number of integration steps

        Returns:
            LyapunovSpectrum with full spectrum
        """
        n = len(initial_state)
        dt = t_total / n_steps

        # Initialize
        x = initial_state.copy()
        Q = np.eye(n)  # Orthonormal frame
        lyap_sum = np.zeros(n)

        # Discard transient
        n_transient = int(t_transient / dt)
        for _ in range(n_transient):
            k1 = equations(x, 0)
            k2 = equations(x + 0.5*dt*k1, 0)
            k3 = equations(x + 0.5*dt*k2, 0)
            k4 = equations(x + dt*k3, 0)
            x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        # Main loop
        n_main = n_steps - n_transient
        for i in range(n_main):
            # Evolve state
            k1 = equations(x, 0)
            k2 = equations(x + 0.5*dt*k1, 0)
            k3 = equations(x + 0.5*dt*k2, 0)
            k4 = equations(x + dt*k3, 0)
            x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

            # Evolve tangent vectors
            J = jacobian(x)
            Q = Q + dt * (J @ Q)

            # Reorthonormalize periodically
            if i % 10 == 0:
                Q, R = qr(Q)
                lyap_sum += np.log(np.abs(np.diag(R)))

        # Average exponents
        exponents = lyap_sum / (n_main * dt)
        exponents = np.sort(exponents)[::-1]

        return self._create_spectrum(exponents)

    def largest_from_series(
        self,
        series: np.ndarray,
        embedding_dim: int = 3,
        delay: int = 1,
        min_neighbors: int = 10,
        mean_period: int = None
    ) -> float:
        """
        Estimate largest Lyapunov exponent using Rosenstein's algorithm.

        This is the most robust method for noisy time series.

        Args:
            series: Input time series
            embedding_dim: Embedding dimension
            delay: Time delay for embedding
            min_neighbors: Minimum number of neighbors to consider
            mean_period: Mean orbital period (for excluding temporal neighbors)

        Returns:
            Largest Lyapunov exponent estimate
        """
        n = len(series)

        # Create embedding
        n_vectors = n - (embedding_dim - 1) * delay
        embedded = np.zeros((n_vectors, embedding_dim))
        for i in range(embedding_dim):
            embedded[:, i] = series[i*delay:i*delay + n_vectors]

        # Estimate mean period if not provided
        if mean_period is None:
            # Use autocorrelation zero-crossing
            autocorr = np.correlate(series - np.mean(series),
                                   series - np.mean(series), mode='full')
            autocorr = autocorr[n-1:] / autocorr[n-1]
            zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
            mean_period = zero_crossings[0] if len(zero_crossings) > 0 else 10

        # For each point, find nearest neighbor (excluding temporal neighbors)
        divergence = []
        max_iter = min(n_vectors - mean_period * 10, n_vectors // 2)

        for i in range(max_iter):
            # Distances to all points
            distances = np.linalg.norm(embedded - embedded[i], axis=1)

            # Exclude temporal neighbors
            temporal_mask = np.abs(np.arange(n_vectors) - i) <= mean_period
            distances[temporal_mask] = np.inf

            # Find nearest neighbor
            j = np.argmin(distances)
            if distances[j] == np.inf:
                continue

            # Track divergence
            div_curve = []
            for k in range(min(mean_period * 10, n_vectors - max(i, j) - 1)):
                if i + k >= n_vectors or j + k >= n_vectors:
                    break
                d = np.linalg.norm(embedded[i+k] - embedded[j+k])
                if d > 0:
                    div_curve.append(np.log(d))
                else:
                    break

            if len(div_curve) > mean_period:
                divergence.append(div_curve)

        if not divergence:
            return np.nan

        # Average divergence curves
        max_len = max(len(d) for d in divergence)
        avg_divergence = np.zeros(max_len)
        counts = np.zeros(max_len)

        for div in divergence:
            avg_divergence[:len(div)] += div
            counts[:len(div)] += 1

        avg_divergence = avg_divergence / (counts + 1e-10)

        # Linear fit to find slope (Lyapunov exponent)
        # Use the linear region (typically early times)
        fit_range = min(mean_period * 2, len(avg_divergence) // 2)
        if fit_range < 5:
            return np.nan

        t = np.arange(fit_range) * self.dt
        slope, _ = np.polyfit(t, avg_divergence[:fit_range], 1)

        return slope

    def wolf_algorithm(
        self,
        series: np.ndarray,
        embedding_dim: int = 3,
        delay: int = 1,
        evolve_time: int = 10,
        threshold_min: float = None,
        threshold_max: float = None
    ) -> Tuple[float, List[float]]:
        """
        Wolf algorithm for Lyapunov exponent estimation.

        Tracks divergence of nearby trajectories and replaces
        the reference trajectory when divergence exceeds threshold.

        Args:
            series: Input time series
            embedding_dim: Embedding dimension
            delay: Time delay
            evolve_time: Steps before replacement
            threshold_min: Minimum distance for neighbor search
            threshold_max: Maximum distance for replacement

        Returns:
            Lyapunov exponent and list of local estimates
        """
        n = len(series)

        # Create embedding
        n_vectors = n - (embedding_dim - 1) * delay
        embedded = np.zeros((n_vectors, embedding_dim))
        for i in range(embedding_dim):
            embedded[:, i] = series[i*delay:i*delay + n_vectors]

        # Set thresholds
        if threshold_min is None:
            threshold_min = np.std(embedded) * 0.01
        if threshold_max is None:
            threshold_max = np.std(embedded) * 0.1

        # Initialize
        current_idx = 0
        local_exponents = []

        while current_idx < n_vectors - evolve_time - 1:
            # Find nearest neighbor
            distances = np.linalg.norm(embedded - embedded[current_idx], axis=1)
            distances[max(0, current_idx-1):min(n_vectors, current_idx+2)] = np.inf

            # Find neighbor in distance range
            valid = (distances > threshold_min) & (distances < threshold_max)
            if not np.any(valid):
                current_idx += 1
                continue

            neighbor_idx = np.where(valid)[0][np.argmin(distances[valid])]
            d0 = distances[neighbor_idx]

            # Evolve
            new_current = current_idx + evolve_time
            new_neighbor = neighbor_idx + evolve_time

            if new_current >= n_vectors or new_neighbor >= n_vectors:
                break

            d1 = np.linalg.norm(embedded[new_current] - embedded[new_neighbor])

            if d1 > 0 and d0 > 0:
                local_exp = np.log(d1 / d0) / (evolve_time * self.dt)
                local_exponents.append(local_exp)

            current_idx = new_current

        if not local_exponents:
            return np.nan, []

        return np.mean(local_exponents), local_exponents

    def _create_spectrum(self, exponents: np.ndarray) -> LyapunovSpectrum:
        """Create LyapunovSpectrum from exponent array."""
        # Kaplan-Yorke dimension
        cumsum = np.cumsum(exponents)
        j = np.searchsorted(-cumsum, 0)  # First index where cumsum becomes negative
        if j > 0 and j < len(exponents):
            d_ky = j + cumsum[j-1] / abs(exponents[j])
        else:
            d_ky = len(exponents) if cumsum[-1] > 0 else 0

        # Kolmogorov-Sinai entropy
        h_ks = np.sum(exponents[exponents > 0])

        # Predictability horizon
        if exponents[0] > 0:
            tau = 1 / exponents[0]
        else:
            tau = np.inf

        return LyapunovSpectrum(
            exponents=exponents,
            dimension_kaplan_yorke=d_ky,
            entropy_kolmogorov_sinai=h_ks,
            predictability_horizon=tau
        )


class StabilityAnalyzer:
    """
    Market Stability Analyzer using Lyapunov-based methods.

    Provides real-time stability monitoring and regime detection
    based on local Lyapunov exponent estimation.

    Trading Implications:
        - High positive local LE: Increased volatility expected, reduce positions
        - Near-zero LE: Stable regime, trend-following strategies appropriate
        - Transition from negative to positive: Regime change imminent

    Example:
        >>> analyzer = StabilityAnalyzer(window=50)
        >>> stability = analyzer.rolling_stability(returns)
        >>> trading_signal = analyzer.generate_signal(stability)
    """

    def __init__(self, window: int = 50, embedding_dim: int = 3):
        """
        Initialize the stability analyzer.

        Args:
            window: Rolling window size
            embedding_dim: Embedding dimension for LE estimation
        """
        self.window = window
        self.embedding_dim = embedding_dim
        self.le_calculator = LyapunovExponent()

    def rolling_stability(
        self,
        series: np.ndarray,
        step: int = 1
    ) -> dict:
        """
        Calculate rolling stability metrics.

        Args:
            series: Input time series
            step: Step size for rolling calculation

        Returns:
            Dictionary with rolling metrics
        """
        n = len(series)
        n_windows = (n - self.window) // step + 1

        lyapunov = np.full(n, np.nan)
        volatility = np.full(n, np.nan)
        stability_index = np.full(n, np.nan)

        for i in range(n_windows):
            start = i * step
            end = start + self.window
            idx = end - 1  # Assign to last point of window

            window_data = series[start:end]

            # Local Lyapunov exponent
            try:
                le, _ = self.le_calculator.wolf_algorithm(
                    window_data,
                    embedding_dim=min(self.embedding_dim, self.window // 10)
                )
                lyapunov[idx] = le
            except Exception:
                lyapunov[idx] = np.nan

            # Rolling volatility
            volatility[idx] = np.std(window_data)

            # Stability index (combination of LE and vol)
            if not np.isnan(le):
                # Normalized stability: lower is more stable
                vol_norm = volatility[idx] / (np.nanmean(volatility[:idx+1]) + 1e-10)
                stability_index[idx] = le * vol_norm

        return {
            'lyapunov': lyapunov,
            'volatility': volatility,
            'stability_index': stability_index,
            'regime': self._classify_regime(lyapunov)
        }

    def _classify_regime(
        self,
        lyapunov: np.ndarray,
        threshold_chaos: float = 0.1,
        threshold_stable: float = -0.05
    ) -> np.ndarray:
        """
        Classify market regime based on Lyapunov exponent.

        Returns:
            0: Stable (mean-reverting)
            1: Transition (uncertain)
            2: Chaotic (trending/volatile)
        """
        regime = np.ones(len(lyapunov), dtype=int)  # Default: transition
        regime[lyapunov > threshold_chaos] = 2  # Chaotic
        regime[lyapunov < threshold_stable] = 0  # Stable
        regime[np.isnan(lyapunov)] = 1  # Unknown -> transition

        return regime

    def predictability_horizon(
        self,
        series: np.ndarray,
        confidence: float = 0.9
    ) -> float:
        """
        Estimate predictability horizon in time steps.

        The horizon is the time after which prediction error
        grows to a significant fraction of the system's variability.

        Args:
            series: Input time series
            confidence: Confidence level (determines threshold)

        Returns:
            Predictability horizon in time steps
        """
        le = self.le_calculator.largest_from_series(
            series,
            embedding_dim=self.embedding_dim
        )

        if np.isnan(le) or le <= 0:
            return np.inf

        # Horizon where error grows by factor of 1/confidence
        horizon = -np.log(1 - confidence) / le

        return horizon

    def generate_signal(
        self,
        stability_metrics: dict,
        lookback: int = 5
    ) -> np.ndarray:
        """
        Generate trading signals from stability metrics.

        Signal interpretation:
            +1: Increase exposure (stable regime)
             0: Neutral / reduce exposure (transition)
            -1: Defensive / hedging (chaotic regime)

        Args:
            stability_metrics: Output from rolling_stability
            lookback: Smoothing window

        Returns:
            Trading signal array
        """
        regime = stability_metrics['regime']
        n = len(regime)
        signal = np.zeros(n)

        for i in range(lookback, n):
            window = regime[i-lookback:i]
            stable_count = np.sum(window == 0)
            chaotic_count = np.sum(window == 2)

            if stable_count > lookback * 0.6:
                signal[i] = 1  # Stable - increase exposure
            elif chaotic_count > lookback * 0.6:
                signal[i] = -1  # Chaotic - defensive
            else:
                signal[i] = 0  # Transition - neutral

        return signal
