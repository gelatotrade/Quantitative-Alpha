"""
Feynman Path Integrals for Financial Modeling

The path integral formulation, borrowed from quantum mechanics, provides
an alternative approach to option pricing and risk assessment. Instead of
solving differential equations, we sum over all possible price paths.

Key Concept:
    P(S_T | S_0) = ∫ D[S(t)] exp(-S[S(t)])

where S[S(t)] is the "action" of the path - analogous to the Lagrangian
in physics.

Quantum Finance Analogy:
    - Volatility σ ↔ Planck constant ℏ
    - Low volatility: Classical limit, deterministic paths dominate
    - High volatility: Quantum fluctuations, all paths contribute

Applications:
    - Path-dependent options (Asian, barriers, lookback)
    - Complex derivative pricing
    - Risk assessment under multiple scenarios
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import factorial
from typing import Tuple, Callable, Optional, List
from dataclasses import dataclass


@dataclass
class PathAction:
    """
    Action functional for a price path.

    The action generalizes the idea of "probability weight" for a path.
    Lower action = more probable path (in the classical limit).

    Attributes:
        kinetic: "Kinetic energy" - path roughness/variance
        potential: "Potential energy" - cost of deviating from drift
        total: Total action
    """
    kinetic: float
    potential: float
    total: float


class FeynmanPathIntegral:
    """
    Feynman Path Integral Methods for Option Pricing.

    Implements Monte Carlo path integration with importance sampling
    for accurate pricing of path-dependent derivatives.

    The transition amplitude is:
        K(S_T, T; S_0, 0) = ∫ D[S] exp(-S[S]/σ²)

    where the action S[S] encodes the dynamics.

    Example:
        >>> fpi = FeynmanPathIntegral(S0=100, T=1.0, sigma=0.2)
        >>> asian_price = fpi.price_asian_option(strike=100)
        >>> barrier_price = fpi.price_barrier_option(strike=100, barrier=90)
    """

    def __init__(
        self,
        S0: float,
        T: float,
        r: float = 0.05,
        sigma: float = 0.2,
        n_steps: int = 252
    ):
        """
        Initialize path integral calculator.

        Args:
            S0: Initial price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            n_steps: Number of time steps for discretization
        """
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = T / n_steps

    def generate_paths(
        self,
        n_paths: int = 10000,
        antithetic: bool = True
    ) -> np.ndarray:
        """
        Generate price paths for Monte Carlo integration.

        Uses geometric Brownian motion with optional antithetic variates.

        Args:
            n_paths: Number of paths
            antithetic: Use antithetic variates for variance reduction

        Returns:
            Price paths (n_paths, n_steps + 1)
        """
        dt = self.dt
        n_steps = self.n_steps

        if antithetic:
            n_half = n_paths // 2
            dW = np.random.randn(n_half, n_steps) * np.sqrt(dt)
            dW = np.vstack([dW, -dW])  # Antithetic pairs
        else:
            dW = np.random.randn(n_paths, n_steps) * np.sqrt(dt)

        # Exact simulation of GBM
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * dW

        log_returns = drift + diffusion
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices = np.hstack([
            np.zeros((n_paths if not antithetic else n_half * 2, 1)),
            log_prices
        ])

        paths = self.S0 * np.exp(log_prices)
        return paths

    def compute_action(
        self,
        path: np.ndarray,
        drift: Callable[[float, float], float] = None
    ) -> PathAction:
        """
        Compute action (Lagrangian integral) for a path.

        Action = ∫ L dt where L = T - V (kinetic - potential)

        For GBM: L = (1/2σ²) * (dlog S/dt - μ)²

        Args:
            path: Price path
            drift: Drift function μ(S, t)

        Returns:
            PathAction with kinetic, potential, and total action
        """
        if drift is None:
            drift = lambda S, t: self.r

        log_path = np.log(path)
        d_log_S = np.diff(log_path)

        dt = self.dt
        times = np.arange(len(path) - 1) * dt

        # Kinetic term: (dlog S/dt)²
        kinetic = np.sum(d_log_S**2 / dt) / (2 * self.sigma**2)

        # Potential term: deviation from expected drift
        expected_drift = np.array([drift(path[i], times[i]) for i in range(len(times))])
        potential = np.sum((d_log_S/dt - expected_drift + 0.5*self.sigma**2)**2 * dt) / (2 * self.sigma**2)

        return PathAction(
            kinetic=kinetic,
            potential=potential,
            total=kinetic + potential
        )

    def price_european(
        self,
        strike: float,
        option_type: str = 'call',
        n_paths: int = 100000
    ) -> Tuple[float, float]:
        """
        Price European option using path integral Monte Carlo.

        Args:
            strike: Strike price
            option_type: 'call' or 'put'
            n_paths: Number of paths

        Returns:
            (price, standard_error) tuple
        """
        paths = self.generate_paths(n_paths)
        S_T = paths[:, -1]

        if option_type == 'call':
            payoffs = np.maximum(S_T - strike, 0)
        else:
            payoffs = np.maximum(strike - S_T, 0)

        discount = np.exp(-self.r * self.T)
        price = discount * np.mean(payoffs)
        stderr = discount * np.std(payoffs) / np.sqrt(n_paths)

        return price, stderr

    def price_asian_option(
        self,
        strike: float,
        option_type: str = 'call',
        averaging: str = 'arithmetic',
        n_paths: int = 100000
    ) -> Tuple[float, float]:
        """
        Price Asian option (path-dependent averaging).

        Asian options have payoff based on average price over the path,
        making them inherently path-dependent and suited for path integrals.

        Args:
            strike: Strike price
            option_type: 'call' or 'put'
            averaging: 'arithmetic' or 'geometric'
            n_paths: Number of paths

        Returns:
            (price, standard_error) tuple
        """
        paths = self.generate_paths(n_paths)

        if averaging == 'arithmetic':
            avg_price = np.mean(paths, axis=1)
        else:  # geometric
            avg_price = np.exp(np.mean(np.log(paths), axis=1))

        if option_type == 'call':
            payoffs = np.maximum(avg_price - strike, 0)
        else:
            payoffs = np.maximum(strike - avg_price, 0)

        discount = np.exp(-self.r * self.T)
        price = discount * np.mean(payoffs)
        stderr = discount * np.std(payoffs) / np.sqrt(n_paths)

        return price, stderr

    def price_barrier_option(
        self,
        strike: float,
        barrier: float,
        barrier_type: str = 'down-and-out',
        option_type: str = 'call',
        n_paths: int = 100000
    ) -> Tuple[float, float]:
        """
        Price barrier option (path-dependent knockout/knockin).

        Barrier options become worthless (knock-out) or activated (knock-in)
        if the price crosses a barrier level.

        Args:
            strike: Strike price
            barrier: Barrier level
            barrier_type: 'down-and-out', 'up-and-out', 'down-and-in', 'up-and-in'
            option_type: 'call' or 'put'
            n_paths: Number of paths

        Returns:
            (price, standard_error) tuple
        """
        paths = self.generate_paths(n_paths)
        S_T = paths[:, -1]

        # Check barrier crossing
        if 'down' in barrier_type:
            crossed = np.any(paths <= barrier, axis=1)
        else:  # up
            crossed = np.any(paths >= barrier, axis=1)

        # Determine which paths pay off
        if 'out' in barrier_type:
            active = ~crossed  # Knock-out: pay if NOT crossed
        else:  # in
            active = crossed  # Knock-in: pay if crossed

        if option_type == 'call':
            payoffs = np.maximum(S_T - strike, 0) * active
        else:
            payoffs = np.maximum(strike - S_T, 0) * active

        discount = np.exp(-self.r * self.T)
        price = discount * np.mean(payoffs)
        stderr = discount * np.std(payoffs) / np.sqrt(n_paths)

        return price, stderr

    def price_lookback_option(
        self,
        option_type: str = 'call',
        n_paths: int = 100000
    ) -> Tuple[float, float]:
        """
        Price lookback option (payoff depends on extreme values).

        Lookback call: S_T - min(S_t)
        Lookback put: max(S_t) - S_T

        Args:
            option_type: 'call' or 'put'
            n_paths: Number of paths

        Returns:
            (price, standard_error) tuple
        """
        paths = self.generate_paths(n_paths)
        S_T = paths[:, -1]

        if option_type == 'call':
            S_min = np.min(paths, axis=1)
            payoffs = S_T - S_min
        else:
            S_max = np.max(paths, axis=1)
            payoffs = S_max - S_T

        discount = np.exp(-self.r * self.T)
        price = discount * np.mean(payoffs)
        stderr = discount * np.std(payoffs) / np.sqrt(n_paths)

        return price, stderr


class QuantumFinance:
    """
    Quantum Finance: Volatility as the Planck Constant.

    This class explores the analogy between quantum mechanics and finance:
        - σ (volatility) ↔ ℏ (Planck constant)
        - Price path ↔ Particle trajectory
        - Option price ↔ Quantum amplitude

    In the "classical limit" (σ → 0):
        - Prices follow deterministic paths (least action)
        - Markets are predictable

    In the "quantum regime" (high σ):
        - All paths contribute significantly
        - Uncertainty dominates

    This framework provides insight into:
        - When markets are predictable vs chaotic
        - The role of volatility in breaking determinism
        - Phase transitions between market regimes

    Example:
        >>> qf = QuantumFinance(S0=100)
        >>> classical = qf.classical_path(T=1.0)
        >>> quantum_cloud = qf.uncertainty_cloud(T=1.0, sigma=0.3)
    """

    def __init__(self, S0: float, r: float = 0.05):
        """
        Initialize quantum finance framework.

        Args:
            S0: Initial price
            r: Risk-free rate (fundamental drift)
        """
        self.S0 = S0
        self.r = r

    def classical_path(
        self,
        T: float,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate classical (deterministic) price path.

        In the limit σ → 0, this is the path of least action.

        Args:
            T: Time horizon
            n_points: Number of points

        Returns:
            (times, prices) tuple
        """
        times = np.linspace(0, T, n_points)
        prices = self.S0 * np.exp(self.r * times)

        return times, prices

    def uncertainty_cloud(
        self,
        T: float,
        sigma: float,
        confidence_levels: List[float] = [0.5, 0.9, 0.99],
        n_points: int = 100
    ) -> dict:
        """
        Calculate quantum uncertainty cloud around classical path.

        The "uncertainty principle" in finance: the future price
        is not a point but a probability distribution.

        Args:
            T: Time horizon
            sigma: Volatility ("Planck constant")
            confidence_levels: Confidence intervals to compute
            n_points: Number of time points

        Returns:
            Dictionary with classical path and confidence bands
        """
        times = np.linspace(0, T, n_points)

        # Classical path
        classical = self.S0 * np.exp(self.r * times)

        # Standard deviation of log-price at each time
        # Var[log S_t] = σ²t
        log_std = sigma * np.sqrt(times)

        result = {
            'times': times,
            'classical': classical,
            'bands': {}
        }

        for conf in confidence_levels:
            z = norm_ppf((1 + conf) / 2)
            upper = classical * np.exp(z * log_std)
            lower = classical * np.exp(-z * log_std)
            result['bands'][conf] = {'upper': upper, 'lower': lower}

        return result

    def decoherence_time(
        self,
        sigma: float,
        precision: float = 0.1
    ) -> float:
        """
        Estimate decoherence time (predictability horizon).

        This is the time at which quantum fluctuations grow to
        the size of the precision requirement.

        Decoherence occurs when σ√t ≈ precision

        Args:
            sigma: Volatility
            precision: Required prediction precision (e.g., 10% = 0.1)

        Returns:
            Decoherence time (predictability horizon)
        """
        return (precision / sigma) ** 2

    def effective_planck_constant(
        self,
        returns: np.ndarray,
        dt: float = 1/252
    ) -> float:
        """
        Estimate effective "Planck constant" from market data.

        This is essentially the realized volatility, but framed
        in quantum terms as the measure of market "quantumness".

        Args:
            returns: Return series
            dt: Time step

        Returns:
            Effective volatility / Planck constant
        """
        # Annualized volatility
        return np.std(returns) * np.sqrt(1 / dt)

    def classical_vs_quantum_ratio(
        self,
        sigma: float,
        T: float
    ) -> float:
        """
        Calculate ratio of quantum to classical uncertainty.

        When this ratio >> 1, the market is in "quantum regime"
        (highly unpredictable). When << 1, classical/deterministic.

        Args:
            sigma: Volatility
            T: Time horizon

        Returns:
            Ratio of quantum to classical uncertainty
        """
        # Classical uncertainty: just the mean return uncertainty
        # Quantum uncertainty: σ√T
        classical_uncertainty = 0.01  # Baseline 1% precision in drift estimate
        quantum_uncertainty = sigma * np.sqrt(T)

        return quantum_uncertainty / classical_uncertainty


def norm_ppf(p: float) -> float:
    """Standard normal percent point function (inverse CDF)."""
    from scipy.stats import norm
    return norm.ppf(p)
