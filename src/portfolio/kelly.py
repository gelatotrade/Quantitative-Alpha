"""
Kelly Criterion for Optimal Position Sizing

The Kelly Criterion determines the optimal fraction of capital to bet
that maximizes the expected geometric growth rate of wealth.

Key Formula:
    f* = μ / σ²    (for continuous returns)
    f* = (bp - q) / b    (for discrete bets)

where:
    μ = expected return
    σ = standard deviation
    b = odds (payoff ratio)
    p = probability of winning
    q = 1 - p

Important:
    - Full Kelly is often too aggressive (high volatility)
    - Practitioners use Fractional Kelly (typically 1/2 or 1/4)
    - Kelly assumes accurate probability estimates (sensitive to errors)

Integration with HMM:
    - Increase Kelly fraction in high-confidence bull regimes
    - Reduce fraction during uncertain/bear regimes
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class KellyResult:
    """
    Results from Kelly optimization.

    Attributes:
        optimal_fraction: Optimal betting fraction
        expected_growth: Expected log growth rate
        volatility: Expected volatility of growth
        leverage: Implied leverage (f* > 1 means borrowing)
        half_kelly: Conservative half-Kelly fraction
    """
    optimal_fraction: float
    expected_growth: float
    volatility: float
    leverage: float
    half_kelly: float


class KellyCriterion:
    """
    Kelly Criterion Calculator for Position Sizing.

    Provides both discrete and continuous formulations:
        - Discrete: For binary bets (gambling, binary options)
        - Continuous: For continuous returns (stocks, forex)
        - Multi-asset: Optimal allocation across multiple assets

    Example:
        >>> kelly = KellyCriterion()
        >>> result = kelly.continuous(mu=0.10, sigma=0.20)
        >>> print(f"Optimal fraction: {result.optimal_fraction:.2%}")
        >>> print(f"Half-Kelly: {result.half_kelly:.2%}")
    """

    def discrete(
        self,
        p: float,
        b: float,
        a: float = 1.0
    ) -> KellyResult:
        """
        Kelly criterion for discrete (binary) bets.

        f* = (bp - q) / b

        For even odds (b=1): f* = p - q = 2p - 1

        Args:
            p: Probability of winning
            b: Payout ratio (win amount / bet amount)
            a: Loss ratio (typically 1 = lose entire bet)

        Returns:
            KellyResult with optimal fraction
        """
        q = 1 - p

        # Generalized Kelly
        optimal_f = (p / a) - (q / b)

        # Expected growth rate
        if optimal_f > 0:
            growth = p * np.log(1 + b * optimal_f) + q * np.log(1 - a * optimal_f)
        else:
            growth = 0

        # Volatility of log returns
        if optimal_f > 0:
            var = p * (np.log(1 + b * optimal_f))**2 + q * (np.log(1 - a * optimal_f))**2 - growth**2
            volatility = np.sqrt(max(var, 0))
        else:
            volatility = 0

        return KellyResult(
            optimal_fraction=max(optimal_f, 0),
            expected_growth=growth,
            volatility=volatility,
            leverage=max(optimal_f, 0),
            half_kelly=max(optimal_f / 2, 0)
        )

    def continuous(
        self,
        mu: float,
        sigma: float,
        r: float = 0
    ) -> KellyResult:
        """
        Kelly criterion for continuous returns (Gaussian).

        For log-normal returns:
            f* = (μ - r) / σ²

        This is also the tangent portfolio in mean-variance optimization.

        Args:
            mu: Expected annual return
            sigma: Annual volatility
            r: Risk-free rate

        Returns:
            KellyResult with optimal fraction
        """
        excess_return = mu - r

        if sigma <= 0:
            return KellyResult(
                optimal_fraction=0,
                expected_growth=0,
                volatility=0,
                leverage=0,
                half_kelly=0
            )

        optimal_f = excess_return / (sigma ** 2)

        # Expected geometric growth rate
        # g = r + f*(μ - r) - f²σ²/2
        growth = r + optimal_f * excess_return - 0.5 * (optimal_f * sigma) ** 2

        # Volatility of portfolio
        vol = abs(optimal_f) * sigma

        return KellyResult(
            optimal_fraction=optimal_f,
            expected_growth=growth,
            volatility=vol,
            leverage=optimal_f,
            half_kelly=optimal_f / 2
        )

    def multi_asset(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        r: float = 0
    ) -> np.ndarray:
        """
        Multi-asset Kelly (optimal leverage for each asset).

        The optimal allocation is:
            f* = Σ⁻¹(μ - r)

        This is the tangent portfolio scaled by risk tolerance.

        Args:
            mu: Expected returns vector
            cov: Covariance matrix
            r: Risk-free rate

        Returns:
            Optimal allocation vector
        """
        excess_returns = mu - r
        cov_inv = np.linalg.inv(cov)
        optimal_f = cov_inv @ excess_returns

        return optimal_f

    def from_historical(
        self,
        returns: np.ndarray,
        annualization: float = 252
    ) -> KellyResult:
        """
        Estimate Kelly fraction from historical returns.

        Args:
            returns: Array of periodic returns
            annualization: Factor to annualize (252 for daily)

        Returns:
            KellyResult based on historical statistics
        """
        mu = np.mean(returns) * annualization
        sigma = np.std(returns) * np.sqrt(annualization)

        return self.continuous(mu, sigma)


class FractionalKelly:
    """
    Fractional Kelly Position Sizing.

    Full Kelly is often too aggressive because:
        1. Estimates of μ and σ have error
        2. Returns are not truly Gaussian (fat tails)
        3. Drawdowns at full Kelly can be severe

    Fractional Kelly reduces the optimal fraction:
        f_fractional = fraction × f_kelly

    Common choices:
        - Half Kelly (0.5): Good balance of growth and risk
        - Quarter Kelly (0.25): Conservative, suitable for uncertainty
        - Dynamic: Adjust fraction based on regime confidence

    Example:
        >>> fk = FractionalKelly(fraction=0.5)
        >>> position = fk.size_position(account=100000, kelly_f=0.25)
    """

    def __init__(
        self,
        fraction: float = 0.5,
        max_leverage: float = 2.0,
        min_position: float = 0.01
    ):
        """
        Initialize fractional Kelly sizer.

        Args:
            fraction: Kelly fraction (0.5 = half-Kelly)
            max_leverage: Maximum allowed leverage
            min_position: Minimum position size (as fraction of account)
        """
        self.fraction = fraction
        self.max_leverage = max_leverage
        self.min_position = min_position

    def size_position(
        self,
        account_value: float,
        kelly_f: float,
        regime_confidence: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate position size with Kelly and regime adjustment.

        Args:
            account_value: Current account value
            kelly_f: Full Kelly fraction
            regime_confidence: Regime detection confidence (0 to 1)

        Returns:
            Dictionary with position sizing details
        """
        # Apply fractional Kelly
        adjusted_f = kelly_f * self.fraction

        # Scale by regime confidence
        adjusted_f = adjusted_f * regime_confidence

        # Apply leverage limits
        if adjusted_f > self.max_leverage:
            adjusted_f = self.max_leverage
        elif adjusted_f < -self.max_leverage:
            adjusted_f = -self.max_leverage

        # Minimum position threshold
        if abs(adjusted_f) < self.min_position:
            adjusted_f = 0

        position_value = account_value * adjusted_f

        return {
            'kelly_full': kelly_f,
            'kelly_fractional': kelly_f * self.fraction,
            'regime_adjusted': adjusted_f,
            'position_value': position_value,
            'position_pct': adjusted_f,
            'leverage': abs(adjusted_f)
        }

    def dynamic_fraction(
        self,
        base_fraction: float,
        regime: int,
        regime_confidence: float,
        sharpe_ratio: float = None
    ) -> float:
        """
        Dynamically adjust Kelly fraction based on regime and confidence.

        Higher fraction when:
            - In favorable regime (bull market)
            - High regime confidence
            - High Sharpe ratio

        Args:
            base_fraction: Base Kelly fraction
            regime: Current regime (0=bull, 1=bear, 2=sideways)
            regime_confidence: Confidence in regime detection
            sharpe_ratio: Optional Sharpe ratio for scaling

        Returns:
            Adjusted Kelly fraction
        """
        # Regime multiplier
        regime_multipliers = {
            0: 1.5,   # Bull: more aggressive
            1: 0.5,   # Bear: defensive
            2: 0.75,  # Sideways: moderate
        }
        regime_mult = regime_multipliers.get(regime, 1.0)

        # Confidence adjustment
        confidence_adj = 0.5 + 0.5 * regime_confidence  # Range: 0.5 to 1.0

        # Sharpe ratio adjustment
        if sharpe_ratio is not None:
            if sharpe_ratio > 2:
                sharpe_mult = 1.2
            elif sharpe_ratio > 1:
                sharpe_mult = 1.0
            elif sharpe_ratio > 0:
                sharpe_mult = 0.8
            else:
                sharpe_mult = 0.5
        else:
            sharpe_mult = 1.0

        adjusted = base_fraction * regime_mult * confidence_adj * sharpe_mult

        # Clamp to reasonable range
        return np.clip(adjusted, 0.1, 1.0)

    def optimal_growth_simulation(
        self,
        returns: np.ndarray,
        fractions: np.ndarray = None,
        n_periods: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate growth for different Kelly fractions.

        Helps visualize the tradeoff between growth and risk
        at different Kelly levels.

        Args:
            returns: Historical returns
            fractions: Kelly fractions to test
            n_periods: Number of periods to simulate

        Returns:
            Dictionary with simulation results
        """
        if fractions is None:
            fractions = np.linspace(0.1, 2.0, 20)

        if n_periods is None:
            n_periods = len(returns)

        results = {
            'fractions': fractions,
            'final_wealth': np.zeros(len(fractions)),
            'max_drawdown': np.zeros(len(fractions)),
            'volatility': np.zeros(len(fractions)),
            'growth_rate': np.zeros(len(fractions))
        }

        for i, f in enumerate(fractions):
            # Simulate wealth evolution
            portfolio_returns = f * returns
            wealth = np.cumprod(1 + portfolio_returns)

            results['final_wealth'][i] = wealth[-1]
            results['growth_rate'][i] = np.mean(np.log(1 + portfolio_returns))
            results['volatility'][i] = np.std(portfolio_returns)

            # Max drawdown
            peak = np.maximum.accumulate(wealth)
            drawdown = (peak - wealth) / peak
            results['max_drawdown'][i] = np.max(drawdown)

        return results
