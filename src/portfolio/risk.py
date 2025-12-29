"""
Risk Management and Tail Risk Analysis

Comprehensive risk management toolkit for quantitative portfolios:
    - Value at Risk (VaR) and Expected Shortfall (ES/CVaR)
    - Drawdown analysis
    - Tail risk metrics (skewness, kurtosis)
    - Risk decomposition

The goal is to move beyond simple volatility measures to capture
the full distribution of returns, especially tail events.
"""

import numpy as np
from scipy.stats import norm, t, skew, kurtosis
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class RiskMetrics:
    """
    Comprehensive risk metrics for a portfolio.

    Attributes:
        volatility: Annualized standard deviation
        var_95: 95% Value at Risk (daily)
        var_99: 99% Value at Risk (daily)
        cvar_95: 95% Conditional VaR (Expected Shortfall)
        max_drawdown: Maximum historical drawdown
        skewness: Return distribution skewness
        kurtosis: Return distribution excess kurtosis
        sharpe_ratio: Annualized Sharpe ratio
        sortino_ratio: Sortino ratio (downside risk only)
        calmar_ratio: Return / Max Drawdown
    """
    volatility: float
    var_95: float
    var_99: float
    cvar_95: float
    max_drawdown: float
    skewness: float
    kurtosis: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float


class RiskManager:
    """
    Portfolio Risk Manager.

    Provides real-time risk monitoring and position limit enforcement.
    Integrates with HMM regime detection for dynamic risk limits.

    Example:
        >>> rm = RiskManager(max_var=0.02, max_drawdown=0.10)
        >>> risk_check = rm.check_position(current_portfolio)
        >>> if not risk_check['approved']:
        ...     reduce_positions()
    """

    def __init__(
        self,
        max_var: float = 0.02,
        max_drawdown: float = 0.10,
        var_confidence: float = 0.95,
        lookback_days: int = 252
    ):
        """
        Initialize risk manager.

        Args:
            max_var: Maximum allowed daily VaR
            max_drawdown: Maximum allowed drawdown
            var_confidence: VaR confidence level
            lookback_days: Days for historical calculation
        """
        self.max_var = max_var
        self.max_drawdown = max_drawdown
        self.var_confidence = var_confidence
        self.lookback_days = lookback_days

    def compute_all_metrics(
        self,
        returns: np.ndarray,
        risk_free: float = 0.02
    ) -> RiskMetrics:
        """
        Compute comprehensive risk metrics.

        Args:
            returns: Return series
            risk_free: Annual risk-free rate

        Returns:
            RiskMetrics dataclass
        """
        # Basic statistics
        ann_vol = np.std(returns) * np.sqrt(252)
        ann_return = np.mean(returns) * 252

        # VaR (parametric, assuming normal)
        var_95 = -np.percentile(returns, 5)
        var_99 = -np.percentile(returns, 1)

        # Expected Shortfall (CVaR)
        cvar_95 = -np.mean(returns[returns <= np.percentile(returns, 5)])

        # Drawdown
        cum_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (peak - cum_returns) / peak
        max_dd = np.max(drawdown)

        # Higher moments
        ret_skew = skew(returns)
        ret_kurt = kurtosis(returns)  # Excess kurtosis

        # Risk-adjusted metrics
        daily_rf = risk_free / 252
        sharpe = (ann_return - risk_free) / ann_vol if ann_vol > 0 else 0

        # Sortino (downside deviation)
        downside = returns[returns < 0]
        downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = (ann_return - risk_free) / downside_vol if downside_vol > 0 else 0

        # Calmar
        calmar = ann_return / max_dd if max_dd > 0 else np.inf

        return RiskMetrics(
            volatility=ann_vol,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            max_drawdown=max_dd,
            skewness=ret_skew,
            kurtosis=ret_kurt,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar
        )

    def var_historical(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Historical VaR (percentile method).

        Args:
            returns: Return series
            confidence: Confidence level

        Returns:
            VaR value (positive = loss)
        """
        return -np.percentile(returns, (1 - confidence) * 100)

    def var_parametric(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Parametric VaR (Gaussian assumption).

        Args:
            returns: Return series
            confidence: Confidence level

        Returns:
            VaR value (positive = loss)
        """
        mu = np.mean(returns)
        sigma = np.std(returns)
        return -(mu + norm.ppf(1 - confidence) * sigma)

    def var_cornish_fisher(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Cornish-Fisher VaR (adjusts for skewness and kurtosis).

        More accurate for non-normal distributions.

        Args:
            returns: Return series
            confidence: Confidence level

        Returns:
            VaR value adjusted for fat tails
        """
        mu = np.mean(returns)
        sigma = np.std(returns)
        s = skew(returns)
        k = kurtosis(returns)

        z = norm.ppf(1 - confidence)

        # Cornish-Fisher expansion
        z_cf = (z +
                (z**2 - 1) * s / 6 +
                (z**3 - 3*z) * k / 24 -
                (2*z**3 - 5*z) * s**2 / 36)

        return -(mu + z_cf * sigma)

    def expected_shortfall(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """
        Expected Shortfall (CVaR / Average VaR).

        The expected loss given that loss exceeds VaR.
        More coherent risk measure than VaR.

        Args:
            returns: Return series
            confidence: Confidence level

        Returns:
            Expected Shortfall (positive = loss)
        """
        var = self.var_historical(returns, confidence)
        tail_returns = returns[returns <= -var]

        if len(tail_returns) == 0:
            return var

        return -np.mean(tail_returns)

    def marginal_var(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        confidence: float = 0.95
    ) -> np.ndarray:
        """
        Marginal VaR: sensitivity of VaR to position changes.

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix
            confidence: VaR confidence level

        Returns:
            Marginal VaR for each asset
        """
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_var)

        z = norm.ppf(1 - confidence)
        var = -z * portfolio_std

        # Marginal contribution
        marginal = np.dot(cov_matrix, weights) / portfolio_std * (-z)

        return marginal

    def check_position(
        self,
        current_var: float,
        current_drawdown: float
    ) -> Dict[str, bool]:
        """
        Check if current risk is within limits.

        Args:
            current_var: Current portfolio VaR
            current_drawdown: Current drawdown

        Returns:
            Dictionary with limit checks
        """
        return {
            'var_ok': current_var <= self.max_var,
            'drawdown_ok': current_drawdown <= self.max_drawdown,
            'approved': current_var <= self.max_var and current_drawdown <= self.max_drawdown,
            'var_utilization': current_var / self.max_var,
            'dd_utilization': current_drawdown / self.max_drawdown
        }


class TailRiskAnalyzer:
    """
    Tail Risk Analysis: Beyond Normal Distribution.

    Focuses on extreme events (fat tails) that standard models miss.
    Uses Extreme Value Theory (EVT) and regime-dependent analysis.

    Key Insight:
        Market returns have fat tails (kurtosis > 3).
        Normal VaR underestimates tail risk by 50%+ during crises.

    Example:
        >>> tra = TailRiskAnalyzer()
        >>> tail_metrics = tra.analyze_tails(returns)
        >>> if tail_metrics['left_tail_ratio'] > 2:
        ...     print("Warning: Significant left tail risk")
    """

    def __init__(self, threshold_percentile: float = 5.0):
        """
        Initialize tail risk analyzer.

        Args:
            threshold_percentile: Percentile defining tail (5 = bottom 5%)
        """
        self.threshold_pct = threshold_percentile

    def analyze_tails(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive tail analysis.

        Args:
            returns: Return series

        Returns:
            Dictionary with tail risk metrics
        """
        n = len(returns)
        threshold_low = np.percentile(returns, self.threshold_pct)
        threshold_high = np.percentile(returns, 100 - self.threshold_pct)

        left_tail = returns[returns <= threshold_low]
        right_tail = returns[returns >= threshold_high]

        # Expected tail under normal assumption
        mu, sigma = np.mean(returns), np.std(returns)
        expected_tail_size = norm.ppf(self.threshold_pct / 100) * sigma

        # Actual vs expected
        left_tail_ratio = abs(np.mean(left_tail)) / abs(expected_tail_size)

        # Tail asymmetry
        tail_asymmetry = abs(np.mean(left_tail)) - np.mean(right_tail)

        # Hill estimator for tail index
        sorted_abs = np.sort(np.abs(returns))[::-1]
        k = int(n * self.threshold_pct / 100)
        if k > 1:
            hill_est = 1 / np.mean(np.log(sorted_abs[:k] / sorted_abs[k]))
        else:
            hill_est = np.nan

        return {
            'left_tail_mean': np.mean(left_tail),
            'right_tail_mean': np.mean(right_tail),
            'left_tail_ratio': left_tail_ratio,  # > 1 means fatter than normal
            'tail_asymmetry': tail_asymmetry,
            'hill_estimator': hill_est,  # Tail heaviness
            'skewness': skew(returns),
            'excess_kurtosis': kurtosis(returns),
            'n_left_extremes': len(left_tail),
            'n_right_extremes': len(right_tail)
        }

    def regime_conditional_tails(
        self,
        returns: np.ndarray,
        regimes: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        Analyze tails conditional on regime.

        Args:
            returns: Return series
            regimes: Regime labels

        Returns:
            Dictionary mapping regime to tail metrics
        """
        results = {}

        for regime in np.unique(regimes):
            mask = regimes == regime
            regime_returns = returns[mask]

            if len(regime_returns) > 10:
                results[regime] = self.analyze_tails(regime_returns)
            else:
                results[regime] = {'error': 'insufficient data'}

        return results

    def stress_test(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        stress_scenarios: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Stress test portfolio under extreme scenarios.

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix
            stress_scenarios: Dict of scenario name -> asset returns

        Returns:
            Portfolio return under each scenario
        """
        results = {}

        for name, scenario in stress_scenarios.items():
            portfolio_return = np.dot(weights, scenario)
            results[name] = portfolio_return

        return results

    def generate_stress_scenarios(
        self,
        returns: np.ndarray,
        n_assets: int
    ) -> Dict[str, np.ndarray]:
        """
        Generate standard stress scenarios.

        Args:
            returns: Historical returns matrix (n_samples, n_assets)
            n_assets: Number of assets

        Returns:
            Dictionary of stress scenarios
        """
        scenarios = {}

        # Worst historical day
        daily_portfolio = np.mean(returns, axis=1)
        worst_day = np.argmin(daily_portfolio)
        scenarios['worst_historical'] = returns[worst_day]

        # 3-sigma event
        scenarios['3_sigma_down'] = np.mean(returns, axis=0) - 3 * np.std(returns, axis=0)

        # Correlation spike (all assets down together)
        scenarios['correlation_crisis'] = np.full(n_assets, -0.05)

        # Sector rotation (half up, half down)
        rotation = np.zeros(n_assets)
        rotation[:n_assets//2] = 0.05
        rotation[n_assets//2:] = -0.05
        scenarios['sector_rotation'] = rotation

        return scenarios


class DrawdownAnalyzer:
    """
    Drawdown Analysis and Recovery Time Estimation.

    Tracks underwater periods and estimates time to recovery
    based on historical patterns and current regime.
    """

    def __init__(self):
        """Initialize drawdown analyzer."""
        pass

    def compute_drawdowns(
        self,
        returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute drawdown series.

        Args:
            returns: Return series

        Returns:
            (drawdown_series, peak_series) tuple
        """
        cum_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (peak - cum_returns) / peak

        return drawdown, peak

    def drawdown_periods(
        self,
        returns: np.ndarray
    ) -> List[Dict]:
        """
        Identify distinct drawdown periods.

        Args:
            returns: Return series

        Returns:
            List of drawdown period dictionaries
        """
        drawdown, peak = self.compute_drawdowns(returns)

        periods = []
        in_drawdown = False
        start = 0
        max_dd = 0

        for i, dd in enumerate(drawdown):
            if dd > 0 and not in_drawdown:
                in_drawdown = True
                start = i
                max_dd = dd
            elif dd > 0 and in_drawdown:
                max_dd = max(max_dd, dd)
            elif dd == 0 and in_drawdown:
                periods.append({
                    'start': start,
                    'end': i,
                    'duration': i - start,
                    'max_drawdown': max_dd
                })
                in_drawdown = False
                max_dd = 0

        # Handle ongoing drawdown
        if in_drawdown:
            periods.append({
                'start': start,
                'end': len(drawdown) - 1,
                'duration': len(drawdown) - start,
                'max_drawdown': max_dd,
                'ongoing': True
            })

        return periods

    def expected_recovery_time(
        self,
        current_drawdown: float,
        expected_return: float,
        volatility: float
    ) -> float:
        """
        Estimate time to recover from current drawdown.

        Uses simple drift model: T ≈ -ln(1-dd) / μ

        Args:
            current_drawdown: Current drawdown (e.g., 0.10 = 10%)
            expected_return: Expected annual return
            volatility: Annual volatility

        Returns:
            Expected recovery time in years
        """
        if expected_return <= 0:
            return np.inf

        # Recovery return needed
        recovery_needed = np.log(1 / (1 - current_drawdown))

        # Expected time assuming drift only
        drift_time = recovery_needed / expected_return

        # Add uncertainty buffer based on volatility
        # Higher vol = longer expected recovery
        vol_adjustment = 1 + volatility

        return drift_time * vol_adjustment
