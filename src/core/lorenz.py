"""
Lorenz System Implementation for Financial Market Dynamics

The Lorenz system, originally developed by Edward Lorenz in 1963 for atmospheric
convection modeling, serves as the prototypical example of deterministic chaos.
This implementation applies the system to financial time series analysis.

Key Parameter: LORENZ_sigma_13
    The explicit calibration of sigma=13 (vs standard sigma=10) represents
    higher market friction and delayed information processing, characterizing
    markets with lower liquidity or higher transaction costs.
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import warnings


@dataclass
class LorenzParameters:
    """
    Lorenz system parameters with financial interpretation.

    Attributes:
        sigma (float): Prandtl number - Price/Momentum reaction speed
                       Higher values = more inert price adjustment, stronger trends
        rho (float): Rayleigh number - Liquidity/Speculation degree
                     Controls transition from stability to chaos
        beta (float): Geometric factor - Mean reversion rate of volatility
                      Higher values = faster return to mean
    """
    sigma: float = 13.0  # LORENZ_sigma_13 - Financial calibration
    rho: float = 28.0    # Standard chaotic regime parameter
    beta: float = 8/3    # Standard geometric factor

    def validate(self) -> bool:
        """Validate parameters for chaotic behavior."""
        # Chaos requires rho > rho_critical where rho_critical ≈ 24.74
        rho_critical = self.sigma * (self.sigma + self.beta + 3) / (self.sigma - self.beta - 1)
        return self.rho > rho_critical and self.sigma > 0 and self.beta > 0


class LorenzSystem:
    """
    Lorenz Dynamical System for Market Microstructure Analysis.

    The system models market dynamics through three coupled nonlinear ODEs:
        dx/dt = σ(y - x)       : Order flow dynamics
        dy/dt = x(ρ - z) - y   : Price deviation from equilibrium
        dz/dt = xy - βz        : Market volatility (system energy)

    Financial Interpretation:
        x(t) - Order Flow / Market Momentum (convection rate)
        y(t) - Price deviation from fundamental value
        z(t) - Volatility / Distance from equilibrium

    Example:
        >>> lorenz = LorenzSystem(sigma=13.0)  # LORENZ_sigma_13 calibration
        >>> trajectory = lorenz.integrate([1.0, 1.0, 1.0], t_span=(0, 100))
        >>> regime = lorenz.detect_regime(trajectory)
    """

    def __init__(
        self,
        sigma: float = 13.0,
        rho: float = 28.0,
        beta: float = 8/3,
        params: Optional[LorenzParameters] = None
    ):
        """
        Initialize Lorenz system with financial calibration.

        Args:
            sigma: Prandtl number (default 13 for financial markets)
            rho: Rayleigh number (drives chaos)
            beta: Geometric factor (mean reversion)
            params: Optional LorenzParameters dataclass
        """
        if params is not None:
            self.params = params
        else:
            self.params = LorenzParameters(sigma=sigma, rho=rho, beta=beta)

        if not self.params.validate():
            warnings.warn(
                f"Parameters may not produce chaotic behavior. "
                f"Current rho={self.params.rho} may be below critical threshold."
            )

    def equations(self, state: np.ndarray, t: float = 0) -> np.ndarray:
        """
        Lorenz system differential equations.

        Args:
            state: Current state vector [x, y, z]
            t: Time (unused but required for ODE solvers)

        Returns:
            Derivatives [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state
        sigma, rho, beta = self.params.sigma, self.params.rho, self.params.beta

        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z

        return np.array([dx_dt, dy_dt, dz_dt])

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for stability analysis.

        Args:
            state: Current state vector [x, y, z]

        Returns:
            3x3 Jacobian matrix
        """
        x, y, z = state
        sigma, rho, beta = self.params.sigma, self.params.rho, self.params.beta

        J = np.array([
            [-sigma, sigma, 0],
            [rho - z, -1, -x],
            [y, x, -beta]
        ])
        return J

    def integrate(
        self,
        initial_state: np.ndarray,
        t_span: Tuple[float, float] = (0, 100),
        n_points: int = 10000,
        method: str = 'RK45'
    ) -> dict:
        """
        Integrate the Lorenz system numerically.

        Args:
            initial_state: Initial conditions [x0, y0, z0]
            t_span: Time interval (start, end)
            n_points: Number of output points
            method: Integration method ('RK45', 'DOP853', 'Radau')

        Returns:
            Dictionary containing:
                - t: Time array
                - x, y, z: State trajectories
                - trajectory: Full state array
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        solution = solve_ivp(
            lambda t, y: self.equations(y, t),
            t_span,
            initial_state,
            method=method,
            t_eval=t_eval,
            dense_output=True
        )

        return {
            't': solution.t,
            'x': solution.y[0],
            'y': solution.y[1],
            'z': solution.y[2],
            'trajectory': solution.y.T,
            'success': solution.success
        }

    def find_fixed_points(self) -> list:
        """
        Calculate fixed points of the Lorenz system.

        Returns:
            List of fixed points as numpy arrays
        """
        sigma, rho, beta = self.params.sigma, self.params.rho, self.params.beta

        # Origin is always a fixed point
        fixed_points = [np.array([0, 0, 0])]

        # Additional fixed points exist for rho > 1
        if rho > 1:
            c = np.sqrt(beta * (rho - 1))
            fixed_points.append(np.array([c, c, rho - 1]))
            fixed_points.append(np.array([-c, -c, rho - 1]))

        return fixed_points

    def stability_analysis(self, fixed_point: np.ndarray) -> dict:
        """
        Analyze stability of a fixed point.

        Args:
            fixed_point: Point to analyze

        Returns:
            Dictionary with eigenvalues, eigenvectors, and stability type
        """
        J = self.jacobian(fixed_point)
        eigenvalues, eigenvectors = np.linalg.eig(J)

        # Classify stability
        real_parts = np.real(eigenvalues)
        if np.all(real_parts < 0):
            stability = "stable_node"
        elif np.all(real_parts > 0):
            stability = "unstable_node"
        else:
            stability = "saddle_point"

        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'stability': stability,
            'lyapunov_spectrum': np.sort(real_parts)[::-1]
        }

    def detect_regime(
        self,
        trajectory: np.ndarray,
        z_threshold: float = None
    ) -> np.ndarray:
        """
        Detect market regime based on z-coordinate (volatility proxy).

        The z-coordinate indicates distance from equilibrium:
            - Low z: Stable orbital, trend-following regime
            - High z: Near saddle point, mean-reversion/volatility regime

        Args:
            trajectory: State trajectory (n_points, 3)
            z_threshold: Threshold for regime classification

        Returns:
            Array of regime labels (0: trend, 1: volatile, 2: transition)
        """
        if z_threshold is None:
            z_threshold = self.params.rho - 1  # Fixed point z-coordinate

        z = trajectory[:, 2] if trajectory.ndim == 2 else trajectory

        regimes = np.zeros(len(z), dtype=int)
        regimes[z > z_threshold * 1.2] = 1  # High volatility
        regimes[z < z_threshold * 0.8] = 0  # Low volatility / trending
        regimes[(z >= z_threshold * 0.8) & (z <= z_threshold * 1.2)] = 2  # Transition

        return regimes


class LorenzAttractor:
    """
    Strange Attractor Analysis for Financial Time Series.

    Provides methods for attractor reconstruction, dimension estimation,
    and Lyapunov exponent calculation from market data.
    """

    def __init__(self, lorenz_system: LorenzSystem = None):
        """
        Initialize attractor analyzer.

        Args:
            lorenz_system: Optional LorenzSystem instance
        """
        self.lorenz = lorenz_system or LorenzSystem()

    def generate_attractor(
        self,
        n_points: int = 50000,
        transient: int = 5000
    ) -> np.ndarray:
        """
        Generate points on the strange attractor.

        Args:
            n_points: Number of points to generate
            transient: Initial transient to discard

        Returns:
            Attractor points (n_points, 3)
        """
        # Random initial condition near the attractor
        initial = np.array([1.0, 1.0, 1.0]) + np.random.randn(3) * 0.1

        total_points = n_points + transient
        result = self.lorenz.integrate(
            initial,
            t_span=(0, total_points * 0.01),
            n_points=total_points
        )

        # Discard transient
        return result['trajectory'][transient:]

    def correlation_dimension(
        self,
        trajectory: np.ndarray,
        r_values: np.ndarray = None
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate correlation dimension using Grassberger-Procaccia algorithm.

        Args:
            trajectory: Points on the attractor
            r_values: Radii for correlation sum

        Returns:
            Estimated dimension and (r, C(r)) arrays
        """
        n = len(trajectory)

        if r_values is None:
            # Estimate appropriate r range
            distances = np.linalg.norm(
                trajectory[::10, None] - trajectory[None, ::10],
                axis=2
            )
            r_min = np.percentile(distances[distances > 0], 1)
            r_max = np.percentile(distances, 50)
            r_values = np.logspace(np.log10(r_min), np.log10(r_max), 20)

        # Correlation sum
        C = np.zeros(len(r_values))
        for i, r in enumerate(r_values):
            count = 0
            sample_size = min(1000, n)
            indices = np.random.choice(n, sample_size, replace=False)
            for j in indices:
                distances = np.linalg.norm(trajectory - trajectory[j], axis=1)
                count += np.sum(distances < r) - 1
            C[i] = count / (sample_size * (n - 1))

        # Linear regression in log-log space
        valid = (C > 0) & (C < 1)
        if np.sum(valid) > 2:
            log_r = np.log(r_values[valid])
            log_C = np.log(C[valid])
            dimension = np.polyfit(log_r, log_C, 1)[0]
        else:
            dimension = np.nan

        return dimension, (r_values, C)

    def lyapunov_exponent_estimate(
        self,
        trajectory: np.ndarray,
        dt: float = 0.01
    ) -> float:
        """
        Estimate largest Lyapunov exponent from trajectory.

        Positive exponent indicates chaos (sensitive dependence on initial conditions).

        Args:
            trajectory: State trajectory
            dt: Time step

        Returns:
            Estimated largest Lyapunov exponent
        """
        n = len(trajectory)

        # Find pairs of initially close points
        epsilon = 1e-3
        divergence_rates = []

        for i in range(0, n - 100, 100):
            distances = np.linalg.norm(trajectory[i+1:i+50] - trajectory[i], axis=1)
            close_idx = np.where((distances > epsilon/2) & (distances < epsilon))[0]

            if len(close_idx) > 0:
                j = i + 1 + close_idx[0]
                # Track divergence
                for k in range(1, min(50, n - max(i, j))):
                    d0 = np.linalg.norm(trajectory[i] - trajectory[j])
                    d1 = np.linalg.norm(trajectory[i+k] - trajectory[j+k])
                    if d0 > 0 and d1 > 0:
                        divergence_rates.append(np.log(d1/d0) / (k * dt))

        return np.mean(divergence_rates) if divergence_rates else np.nan


def transform_prices_to_lorenz(
    prices: np.ndarray,
    window: int = 20,
    sigma: float = 13.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform price series to Lorenz-like state space coordinates.

    This creates financial analogs of the Lorenz state variables:
        x: Momentum (rate of change)
        y: Price deviation from moving average
        z: Volatility (rolling standard deviation)

    Args:
        prices: Price time series
        window: Rolling window for calculations
        sigma: Scaling factor (LORENZ_sigma_13)

    Returns:
        Tuple of (x, y, z) coordinate arrays
    """
    returns = np.diff(np.log(prices))

    # x: Momentum / Order flow proxy
    x = np.zeros(len(returns))
    x[window:] = np.convolve(returns, np.ones(window)/window, mode='valid')

    # y: Price deviation from equilibrium (moving average)
    ma = np.convolve(prices, np.ones(window)/window, mode='valid')
    y = np.zeros(len(prices))
    y[window-1:] = (prices[window-1:] - ma) / ma * sigma
    y = y[1:]  # Align with returns

    # z: Volatility (rolling standard deviation)
    z = np.zeros(len(returns))
    for i in range(window, len(returns)):
        z[i] = np.std(returns[i-window:i]) * np.sqrt(252) * sigma

    return x, y, z
