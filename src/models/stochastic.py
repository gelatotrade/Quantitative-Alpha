"""
Stochastic Physics Models for Financial Markets

This module implements stochastic differential equations (SDEs) and their
solutions for modeling market dynamics. Key components:

1. Langevin Dynamics: Models price evolution with friction (market efficiency)
   and noise (information arrival).

2. Fokker-Planck Equation: Evolution of probability density functions,
   enabling prediction of entire price distributions.

3. Path Integral Methods: Feynman-Kac formulation for option pricing
   and risk assessment.

Physical Interpretation:
- γ (friction): Market efficiency / arbitrage force
- σ (noise): Information arrival / volatility
- μ (drift): Fundamental trend / risk premium
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional, Callable, Dict
from dataclasses import dataclass


@dataclass
class SDEParameters:
    """
    Parameters for Stochastic Differential Equations.

    Attributes:
        mu: Drift coefficient (expected return)
        sigma: Diffusion coefficient (volatility)
        gamma: Friction/damping coefficient (mean reversion)
        theta: Long-term mean (for Ornstein-Uhlenbeck)
    """
    mu: float = 0.0
    sigma: float = 0.2
    gamma: float = 0.1
    theta: float = 0.0


class LangevinDynamics:
    """
    Langevin Equation for Market Microstructure.

    The Langevin equation models Brownian motion with friction:
        dp/dt = -γp(t) + ξ(t) + K(x)

    where:
        -γp(t): Dissipation/friction (market efficiency, arbitrage)
        ξ(t): Stochastic force (information arrival, noise)
        K(x): Deterministic force (fundamental pressure)

    Financial Interpretation:
        - High γ: Efficient market, strong mean reversion
        - Low γ: Momentum persists, trend-following profitable
        - The half-life of trends: τ = ln(2)/γ

    Example:
        >>> langevin = LangevinDynamics(gamma=0.1, sigma=0.2)
        >>> paths = langevin.simulate_paths(S0=100, T=1.0, n_paths=1000)
        >>> print(f"Half-life of mean reversion: {langevin.half_life():.2f}")
    """

    def __init__(
        self,
        gamma: float = 0.1,
        sigma: float = 0.2,
        mu: float = 0.0,
        theta: float = None
    ):
        """
        Initialize Langevin dynamics.

        Args:
            gamma: Friction coefficient (mean reversion rate)
            sigma: Volatility / noise amplitude
            mu: Drift / expected return
            theta: Long-term mean (defaults to initial price if None)
        """
        self.gamma = gamma
        self.sigma = sigma
        self.mu = mu
        self.theta = theta

    def half_life(self) -> float:
        """Calculate half-life of mean reversion."""
        if self.gamma <= 0:
            return np.inf
        return np.log(2) / self.gamma

    def characteristic_time(self) -> float:
        """Calculate characteristic relaxation time τ = 1/γ."""
        if self.gamma <= 0:
            return np.inf
        return 1 / self.gamma

    def simulate_paths(
        self,
        S0: float,
        T: float,
        n_steps: int = 252,
        n_paths: int = 1000,
        method: str = 'euler'
    ) -> np.ndarray:
        """
        Simulate price paths using Langevin dynamics.

        Args:
            S0: Initial price
            T: Time horizon (years)
            n_steps: Number of time steps
            n_paths: Number of Monte Carlo paths
            method: 'euler' or 'milstein'

        Returns:
            Price paths array (n_paths, n_steps + 1)
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        # Initialize paths
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        # Long-term mean
        theta = self.theta if self.theta is not None else S0

        # Generate all random numbers at once
        dW = np.random.randn(n_paths, n_steps) * sqrt_dt

        if method == 'euler':
            for t in range(n_steps):
                S = paths[:, t]
                # Ornstein-Uhlenbeck with log-price
                log_S = np.log(S)
                log_theta = np.log(theta)

                dlog_S = (
                    self.gamma * (log_theta - log_S) +
                    (self.mu - 0.5 * self.sigma**2)
                ) * dt + self.sigma * dW[:, t]

                paths[:, t + 1] = S * np.exp(dlog_S)

        elif method == 'milstein':
            for t in range(n_steps):
                S = paths[:, t]
                log_S = np.log(S)
                log_theta = np.log(theta)

                drift = self.gamma * (log_theta - log_S) + self.mu - 0.5 * self.sigma**2
                diffusion = self.sigma

                # Milstein correction
                dlog_S = (
                    drift * dt +
                    diffusion * dW[:, t] +
                    0.5 * diffusion**2 * (dW[:, t]**2 - dt)
                )

                paths[:, t + 1] = S * np.exp(dlog_S)

        return paths

    def transition_density(
        self,
        x0: float,
        x1: float,
        t: float
    ) -> float:
        """
        Analytical transition density for OU process.

        P(x1, t | x0, 0) for Ornstein-Uhlenbeck process.

        Args:
            x0: Initial state
            x1: Final state
            t: Time

        Returns:
            Transition probability density
        """
        theta = self.theta if self.theta is not None else x0

        # Mean and variance of transition distribution
        mean = x0 * np.exp(-self.gamma * t) + theta * (1 - np.exp(-self.gamma * t))
        var = (self.sigma**2 / (2 * self.gamma)) * (1 - np.exp(-2 * self.gamma * t))

        if var <= 0:
            return 0.0

        # Gaussian density
        return np.exp(-(x1 - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)

    def stationary_distribution(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Stationary distribution of the OU process.

        Args:
            x: Points at which to evaluate density

        Returns:
            Stationary density values
        """
        theta = self.theta if self.theta is not None else 0
        var = self.sigma**2 / (2 * self.gamma)

        return np.exp(-(x - theta)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)

    def estimate_from_data(
        self,
        returns: np.ndarray,
        dt: float = 1/252
    ) -> Dict[str, float]:
        """
        Estimate Langevin parameters from return data.

        Uses maximum likelihood estimation for OU process.

        Args:
            returns: Return series
            dt: Time step between observations

        Returns:
            Dictionary of estimated parameters
        """
        n = len(returns)

        # Log-price levels (cumulative returns)
        log_prices = np.cumsum(returns)

        # Regression: dX = γ(θ - X)dt + σdW
        # X_{t+1} - X_t = γ(θ - X_t)dt + ε
        y = np.diff(log_prices)
        X = log_prices[:-1]

        # OLS: y = a + b*X + ε  where b = -γdt, a = γθdt
        X_matrix = np.column_stack([np.ones(len(X)), X])
        coeffs = np.linalg.lstsq(X_matrix, y, rcond=None)[0]

        gamma_est = -coeffs[1] / dt
        theta_est = coeffs[0] / (gamma_est * dt) if gamma_est != 0 else np.mean(log_prices)

        # Estimate sigma from residuals
        residuals = y - X_matrix @ coeffs
        sigma_est = np.std(residuals) / np.sqrt(dt)

        # Drift
        mu_est = np.mean(returns) / dt

        return {
            'gamma': max(gamma_est, 1e-6),
            'theta': theta_est,
            'sigma': sigma_est,
            'mu': mu_est,
            'half_life': np.log(2) / max(gamma_est, 1e-6)
        }


class FokkerPlanck:
    """
    Fokker-Planck Equation Solver for Probability Density Evolution.

    The Fokker-Planck equation (Kolmogorov forward equation) describes
    the time evolution of the probability density function:

        ∂P/∂t = -∂/∂x[D¹(x)P] + ∂²/∂x²[D²(x)P]

    where:
        D¹(x): Drift coefficient (local expected return)
        D²(x): Diffusion coefficient (local volatility)

    Application:
        - Project entire probability clouds of future prices
        - Detect bimodal distributions (regime uncertainty)
        - Superior to point forecasts for risk management

    Example:
        >>> fp = FokkerPlanck(drift=lambda x: -0.1*x, diffusion=lambda x: 0.2)
        >>> P_t = fp.evolve(P0, t=1.0)
        >>> risk_metrics = fp.tail_risk(P_t, threshold=-0.1)
    """

    def __init__(
        self,
        drift: Callable[[np.ndarray], np.ndarray] = None,
        diffusion: Callable[[np.ndarray], np.ndarray] = None,
        x_range: Tuple[float, float] = (-3, 3),
        n_grid: int = 200
    ):
        """
        Initialize Fokker-Planck solver.

        Args:
            drift: Drift coefficient function D¹(x)
            diffusion: Diffusion coefficient function D²(x)
            x_range: Spatial domain (min, max)
            n_grid: Number of grid points
        """
        self.drift = drift if drift else lambda x: np.zeros_like(x)
        self.diffusion = diffusion if diffusion else lambda x: 0.2 * np.ones_like(x)

        self.x_min, self.x_max = x_range
        self.n_grid = n_grid
        self.x = np.linspace(self.x_min, self.x_max, n_grid)
        self.dx = self.x[1] - self.x[0]

    def initial_gaussian(
        self,
        mean: float = 0.0,
        std: float = 0.1
    ) -> np.ndarray:
        """
        Create Gaussian initial condition.

        Args:
            mean: Mean of distribution
            std: Standard deviation

        Returns:
            Initial PDF on the grid
        """
        P0 = np.exp(-(self.x - mean)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))
        # Normalize
        P0 = P0 / (np.sum(P0) * self.dx)
        return P0

    def initial_delta(self, x0: float) -> np.ndarray:
        """
        Create approximate delta function initial condition.

        Args:
            x0: Location of delta

        Returns:
            Approximate delta PDF on the grid
        """
        idx = np.argmin(np.abs(self.x - x0))
        P0 = np.zeros(self.n_grid)
        P0[idx] = 1.0 / self.dx
        return P0

    def build_operator(self) -> np.ndarray:
        """
        Build Fokker-Planck operator matrix.

        Uses finite differences for spatial derivatives.

        Returns:
            Sparse operator matrix
        """
        D1 = self.drift(self.x)  # Drift
        D2 = self.diffusion(self.x)**2 / 2  # Diffusion (note: D² = σ²/2)

        dx = self.dx
        n = self.n_grid

        # First derivative: central difference for drift term
        # -∂/∂x[D¹P] ≈ -D¹_{i+1}P_{i+1} + D¹_{i-1}P_{i-1}) / (2dx)
        diag_p1 = -D1[:-1] / (2 * dx)
        diag_m1 = D1[1:] / (2 * dx)

        # Second derivative: central difference for diffusion term
        # ∂²/∂x²[D²P] ≈ (D²_{i+1}P_{i+1} - 2D²_iP_i + D²_{i-1}P_{i-1}) / dx²
        diag_0 = -2 * D2 / dx**2
        diag_p1_diff = D2[:-1] / dx**2
        diag_m1_diff = D2[1:] / dx**2

        # Combine
        diag_p1_total = diag_p1 + diag_p1_diff
        diag_m1_total = diag_m1 + diag_m1_diff

        # Build matrix
        L = (diags(diag_0, 0, shape=(n, n)) +
             diags(diag_p1_total, 1, shape=(n, n)) +
             diags(diag_m1_total, -1, shape=(n, n)))

        # Boundary conditions: zero flux
        L = L.toarray()
        L[0, :] = 0
        L[-1, :] = 0
        L[0, 0] = -1
        L[-1, -1] = -1

        return L

    def evolve(
        self,
        P0: np.ndarray,
        t_final: float,
        dt: float = None,
        method: str = 'crank-nicolson'
    ) -> np.ndarray:
        """
        Evolve the probability density forward in time.

        Args:
            P0: Initial probability density
            t_final: Final time
            dt: Time step (auto if None)
            method: 'euler', 'crank-nicolson', or 'rk4'

        Returns:
            Final probability density
        """
        L = self.build_operator()

        if dt is None:
            # CFL condition
            D2_max = np.max(self.diffusion(self.x)**2 / 2)
            dt = 0.4 * self.dx**2 / D2_max

        n_steps = int(np.ceil(t_final / dt))
        dt = t_final / n_steps

        P = P0.copy()

        if method == 'euler':
            for _ in range(n_steps):
                P = P + dt * (L @ P)
                P = np.maximum(P, 0)  # Ensure non-negative
                P = P / (np.sum(P) * self.dx)  # Normalize

        elif method == 'crank-nicolson':
            I = np.eye(self.n_grid)
            A = I - 0.5 * dt * L
            B = I + 0.5 * dt * L

            for _ in range(n_steps):
                P = np.linalg.solve(A, B @ P)
                P = np.maximum(P, 0)
                P = P / (np.sum(P) * self.dx)

        elif method == 'rk4':
            for _ in range(n_steps):
                k1 = L @ P
                k2 = L @ (P + 0.5 * dt * k1)
                k3 = L @ (P + 0.5 * dt * k2)
                k4 = L @ (P + dt * k3)
                P = P + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
                P = np.maximum(P, 0)
                P = P / (np.sum(P) * self.dx)

        return P

    def evolve_trajectory(
        self,
        P0: np.ndarray,
        t_final: float,
        n_snapshots: int = 10,
        dt: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve and return snapshots of the PDF over time.

        Args:
            P0: Initial probability density
            t_final: Final time
            n_snapshots: Number of time snapshots
            dt: Time step

        Returns:
            (times, PDFs) tuple where PDFs is (n_snapshots, n_grid)
        """
        t_values = np.linspace(0, t_final, n_snapshots)
        PDFs = np.zeros((n_snapshots, self.n_grid))
        PDFs[0] = P0

        P = P0.copy()
        t_current = 0

        for i in range(1, n_snapshots):
            P = self.evolve(P, t_values[i] - t_current, dt=dt)
            PDFs[i] = P
            t_current = t_values[i]

        return t_values, PDFs

    def tail_risk(
        self,
        P: np.ndarray,
        threshold: float,
        direction: str = 'left'
    ) -> float:
        """
        Calculate tail probability.

        Args:
            P: Probability density
            threshold: Threshold value
            direction: 'left' (below threshold) or 'right' (above)

        Returns:
            Tail probability
        """
        if direction == 'left':
            mask = self.x < threshold
        else:
            mask = self.x > threshold

        return np.sum(P[mask]) * self.dx

    def expected_value(self, P: np.ndarray) -> float:
        """Calculate expected value E[X]."""
        return np.sum(self.x * P) * self.dx

    def variance(self, P: np.ndarray) -> float:
        """Calculate variance Var[X]."""
        mean = self.expected_value(P)
        return np.sum((self.x - mean)**2 * P) * self.dx

    def quantile(self, P: np.ndarray, q: float) -> float:
        """
        Calculate quantile (inverse CDF).

        Args:
            P: Probability density
            q: Quantile (0 to 1)

        Returns:
            Quantile value
        """
        cdf = np.cumsum(P) * self.dx
        idx = np.searchsorted(cdf, q)
        return self.x[min(idx, len(self.x) - 1)]

    def value_at_risk(self, P: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk.

        Args:
            P: Probability density of returns
            confidence: Confidence level

        Returns:
            VaR (positive number represents loss)
        """
        return -self.quantile(P, 1 - confidence)

    def expected_shortfall(
        self,
        P: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (CVaR).

        Args:
            P: Probability density of returns
            confidence: Confidence level

        Returns:
            ES (positive number represents expected loss in tail)
        """
        var = self.value_at_risk(P, confidence)
        threshold = -var

        mask = self.x < threshold
        if np.sum(P[mask]) == 0:
            return var

        tail_prob = np.sum(P[mask]) * self.dx
        expected_tail = np.sum(self.x[mask] * P[mask]) * self.dx / tail_prob

        return -expected_tail


class SDESimulator:
    """
    General SDE Simulator with Multiple Schemes.

    Solves SDEs of the form:
        dX = μ(X,t)dt + σ(X,t)dW

    Supports various discretization schemes:
        - Euler-Maruyama
        - Milstein
        - Heun (predictor-corrector)
        - Stochastic Runge-Kutta

    Example:
        >>> # Geometric Brownian Motion
        >>> sde = SDESimulator(
        ...     drift=lambda x, t: 0.05 * x,
        ...     diffusion=lambda x, t: 0.2 * x
        ... )
        >>> paths = sde.simulate(X0=100, T=1.0, n_paths=10000)
    """

    def __init__(
        self,
        drift: Callable[[np.ndarray, float], np.ndarray],
        diffusion: Callable[[np.ndarray, float], np.ndarray],
        diffusion_derivative: Callable[[np.ndarray, float], np.ndarray] = None
    ):
        """
        Initialize SDE simulator.

        Args:
            drift: Drift function μ(X, t)
            diffusion: Diffusion function σ(X, t)
            diffusion_derivative: dσ/dX for Milstein scheme
        """
        self.mu = drift
        self.sigma = diffusion
        self.sigma_prime = diffusion_derivative

    def simulate(
        self,
        X0: float,
        T: float,
        n_steps: int = 252,
        n_paths: int = 1000,
        method: str = 'euler'
    ) -> np.ndarray:
        """
        Simulate SDE paths.

        Args:
            X0: Initial condition
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths
            method: 'euler', 'milstein', 'heun', or 'srk'

        Returns:
            Paths array (n_paths, n_steps + 1)
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        X = np.zeros((n_paths, n_steps + 1))
        X[:, 0] = X0

        dW = np.random.randn(n_paths, n_steps) * sqrt_dt

        for i in range(n_steps):
            t = i * dt
            Xt = X[:, i]

            if method == 'euler':
                X[:, i+1] = (
                    Xt +
                    self.mu(Xt, t) * dt +
                    self.sigma(Xt, t) * dW[:, i]
                )

            elif method == 'milstein':
                if self.sigma_prime is None:
                    raise ValueError("Milstein requires diffusion derivative")

                mu_t = self.mu(Xt, t)
                sigma_t = self.sigma(Xt, t)
                sigma_prime_t = self.sigma_prime(Xt, t)

                X[:, i+1] = (
                    Xt +
                    mu_t * dt +
                    sigma_t * dW[:, i] +
                    0.5 * sigma_t * sigma_prime_t * (dW[:, i]**2 - dt)
                )

            elif method == 'heun':
                # Predictor
                X_pred = Xt + self.mu(Xt, t) * dt + self.sigma(Xt, t) * dW[:, i]

                # Corrector
                X[:, i+1] = Xt + 0.5 * (
                    self.mu(Xt, t) + self.mu(X_pred, t + dt)
                ) * dt + 0.5 * (
                    self.sigma(Xt, t) + self.sigma(X_pred, t + dt)
                ) * dW[:, i]

            elif method == 'srk':
                # Stochastic Runge-Kutta (order 1.0)
                sigma_t = self.sigma(Xt, t)
                X_hat = Xt + sigma_t * sqrt_dt

                X[:, i+1] = (
                    Xt +
                    self.mu(Xt, t) * dt +
                    sigma_t * dW[:, i] +
                    0.5 * (self.sigma(X_hat, t) - sigma_t) * (dW[:, i]**2 / sqrt_dt - sqrt_dt)
                )

        return X

    def expectation(
        self,
        X0: float,
        T: float,
        function: Callable[[np.ndarray], np.ndarray] = None,
        n_paths: int = 10000
    ) -> Tuple[float, float]:
        """
        Monte Carlo estimation of E[f(X_T)].

        Args:
            X0: Initial condition
            T: Time horizon
            function: Payoff function (identity if None)
            n_paths: Number of paths

        Returns:
            (mean, std_error) tuple
        """
        paths = self.simulate(X0, T, n_paths=n_paths)
        X_T = paths[:, -1]

        if function is None:
            values = X_T
        else:
            values = function(X_T)

        mean = np.mean(values)
        std_error = np.std(values) / np.sqrt(n_paths)

        return mean, std_error
