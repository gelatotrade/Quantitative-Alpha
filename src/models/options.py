"""
Options Pricing, Greeks, and Implied Volatility Surfaces

This module provides comprehensive tools for options analysis:

1. Black-Scholes Model: Analytical pricing and Greeks
2. Implied Volatility Surface: 3D visualization of market fear topology
3. Greeks Surfaces: Multi-dimensional risk exposure visualization

The IV Surface is particularly important:
    - Skew: OTM puts have higher IV (crash insurance / tail risk)
    - Term Structure: Short-term vs long-term volatility expectations
    - Smile: Deviations from Black-Scholes assumptions

Greeks Interpretation:
    - Delta (Δ): Price sensitivity to underlying
    - Gamma (Γ): Delta's rate of change (convexity)
    - Vega (ν): Sensitivity to volatility
    - Theta (Θ): Time decay
    - Rho (ρ): Interest rate sensitivity
    - Vanna: Cross-derivative dΔ/dσ
    - Volga: Vega convexity dν/dσ
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import RectBivariateSpline, griddata
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum


class OptionType(Enum):
    """Option type enumeration."""
    CALL = 'call'
    PUT = 'put'


@dataclass
class OptionContract:
    """
    Option contract specification.

    Attributes:
        S: Underlying price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield
        option_type: Call or Put
    """
    S: float
    K: float
    T: float
    r: float = 0.05
    sigma: float = 0.2
    q: float = 0.0
    option_type: OptionType = OptionType.CALL


class BlackScholes:
    """
    Black-Scholes Option Pricing Model with Complete Greeks.

    The model assumes:
        - Log-normal price distribution
        - Constant volatility
        - No arbitrage, continuous trading

    Provides analytical formulas for all Greeks including
    second and third order sensitivities.

    Example:
        >>> bs = BlackScholes()
        >>> price = bs.price(S=100, K=100, T=1, r=0.05, sigma=0.2)
        >>> greeks = bs.all_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2)
        >>> print(f"Delta: {greeks['delta']:.4f}, Gamma: {greeks['gamma']:.4f}")
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate d1 parameter."""
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate d2 parameter."""
        return BlackScholes.d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0,
        option_type: OptionType = OptionType.CALL
    ) -> float:
        """
        Calculate option price.

        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            option_type: Call or Put

        Returns:
            Option price
        """
        if T <= 0:
            # At expiration
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = self.d1(S, K, T, r, sigma, q)
        d2 = self.d2(S, K, T, r, sigma, q)

        if option_type == OptionType.CALL:
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        return price

    def delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0,
        option_type: OptionType = OptionType.CALL
    ) -> float:
        """
        Calculate Delta (∂V/∂S).

        Interpretation: Position equivalent in underlying shares.
        """
        if T <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1 = self.d1(S, K, T, r, sigma, q)

        if option_type == OptionType.CALL:
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return -np.exp(-q * T) * norm.cdf(-d1)

    def gamma(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Calculate Gamma (∂²V/∂S²).

        Interpretation: Rate of delta change. High gamma near ATM = pin risk.
        Same for calls and puts (put-call parity).
        """
        if T <= 0:
            return 0.0

        d1 = self.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def vega(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Calculate Vega (∂V/∂σ).

        Interpretation: Sensitivity to volatility changes.
        Returns vega per 1% change in volatility (not per 100%).
        """
        if T <= 0:
            return 0.0

        d1 = self.d1(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100

    def theta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0,
        option_type: OptionType = OptionType.CALL
    ) -> float:
        """
        Calculate Theta (∂V/∂t).

        Interpretation: Time decay per day (returned as negative for long positions).
        """
        if T <= 0:
            return 0.0

        d1 = self.d1(S, K, T, r, sigma, q)
        d2 = self.d2(S, K, T, r, sigma, q)

        term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

        if option_type == OptionType.CALL:
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)

        # Return daily theta (divide by 365)
        return (term1 + term2 + term3) / 365

    def rho(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0,
        option_type: OptionType = OptionType.CALL
    ) -> float:
        """
        Calculate Rho (∂V/∂r).

        Interpretation: Sensitivity to interest rate changes.
        Returns rho per 1% change in rate.
        """
        if T <= 0:
            return 0.0

        d2 = self.d2(S, K, T, r, sigma, q)

        if option_type == OptionType.CALL:
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    def vanna(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Calculate Vanna (∂Δ/∂σ = ∂ν/∂S).

        Interpretation: How delta changes with volatility.
        Critical for skew trading.
        """
        if T <= 0:
            return 0.0

        d1 = self.d1(S, K, T, r, sigma, q)
        d2 = self.d2(S, K, T, r, sigma, q)

        return -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma

    def volga(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0
    ) -> float:
        """
        Calculate Volga (∂²V/∂σ² = ∂ν/∂σ).

        Interpretation: Vega convexity. Positive volga profits from vol-of-vol.
        """
        if T <= 0:
            return 0.0

        d1 = self.d1(S, K, T, r, sigma, q)
        d2 = self.d2(S, K, T, r, sigma, q)

        vega_val = self.vega(S, K, T, r, sigma, q) * 100  # Unnormalized
        return vega_val * d1 * d2 / sigma

    def charm(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0,
        option_type: OptionType = OptionType.CALL
    ) -> float:
        """
        Calculate Charm (∂Δ/∂t = ∂Θ/∂S).

        Interpretation: How delta changes with time passage.
        """
        if T <= 0:
            return 0.0

        d1 = self.d1(S, K, T, r, sigma, q)
        d2 = self.d2(S, K, T, r, sigma, q)

        term = (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))

        if option_type == OptionType.CALL:
            return -np.exp(-q * T) * (norm.pdf(d1) * term + q * norm.cdf(d1))
        else:
            return -np.exp(-q * T) * (norm.pdf(d1) * term - q * norm.cdf(-d1))

    def all_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0,
        option_type: OptionType = OptionType.CALL
    ) -> Dict[str, float]:
        """
        Calculate all Greeks at once.

        Returns:
            Dictionary with all Greek values
        """
        return {
            'price': self.price(S, K, T, r, sigma, q, option_type),
            'delta': self.delta(S, K, T, r, sigma, q, option_type),
            'gamma': self.gamma(S, K, T, r, sigma, q),
            'vega': self.vega(S, K, T, r, sigma, q),
            'theta': self.theta(S, K, T, r, sigma, q, option_type),
            'rho': self.rho(S, K, T, r, sigma, q, option_type),
            'vanna': self.vanna(S, K, T, r, sigma, q),
            'volga': self.volga(S, K, T, r, sigma, q),
            'charm': self.charm(S, K, T, r, sigma, q, option_type)
        }


class ImpliedVolatilitySurface:
    """
    Implied Volatility Surface: The Topology of Market Fear.

    The IV surface shows implied volatility as a function of:
        - Strike (K) or Moneyness (K/S)
        - Time to Expiration (T)

    Key Features:
        - Smile/Skew: IV varies with strike (fat tails, jump risk)
        - Term Structure: IV varies with maturity
        - Surface Dynamics: How the surface moves over time

    Applications:
        - Detect mispricing opportunities
        - Quantify tail risk (skew premium)
        - Volatility arbitrage (RV vs IV)

    Example:
        >>> surface = ImpliedVolatilitySurface()
        >>> surface.fit(strikes, maturities, iv_data)
        >>> iv = surface.get_iv(moneyness=0.95, maturity=0.25)
        >>> metrics = surface.analyze_skew(maturity=0.25)
    """

    def __init__(self):
        """Initialize the IV surface."""
        self.bs = BlackScholes()
        self.surface = None
        self.strikes = None
        self.maturities = None
        self.iv_grid = None

    def calculate_iv(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0,
        option_type: OptionType = OptionType.CALL
    ) -> float:
        """
        Calculate implied volatility from option price.

        Uses Brent's method for root finding.

        Args:
            price: Observed option price
            S, K, T, r, q: Option parameters
            option_type: Call or Put

        Returns:
            Implied volatility
        """
        def objective(sigma):
            return self.bs.price(S, K, T, r, sigma, q, option_type) - price

        try:
            iv = brentq(objective, 0.001, 5.0)
        except ValueError:
            # No solution found
            iv = np.nan

        return iv

    def generate_sabr_surface(
        self,
        S: float = 100,
        r: float = 0.05,
        alpha: float = 0.2,
        beta: float = 0.5,
        rho: float = -0.3,
        nu: float = 0.4,
        n_strikes: int = 21,
        n_maturities: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic IV surface using SABR model.

        The SABR model produces realistic smile/skew patterns:
            - beta controls backbone
            - rho controls skew
            - nu controls smile curvature

        Args:
            S: Spot price
            r: Risk-free rate
            alpha: Initial volatility
            beta: CEV exponent (0 to 1)
            rho: Correlation (-1 to 1)
            nu: Vol-of-vol

        Returns:
            (strikes, maturities, iv_grid) tuple
        """
        # Strike range: 70% to 130% of spot
        self.strikes = np.linspace(0.7 * S, 1.3 * S, n_strikes)
        self.maturities = np.linspace(0.1, 2.0, n_maturities)

        self.iv_grid = np.zeros((n_maturities, n_strikes))

        for i, T in enumerate(self.maturities):
            for j, K in enumerate(self.strikes):
                # SABR approximation for implied volatility
                F = S * np.exp(r * T)  # Forward
                if abs(F - K) < 1e-10:
                    # ATM case
                    iv = alpha * (1 + ((1 - beta)**2 / 24 * alpha**2 / F**(2*(1-beta)) +
                                       0.25 * rho * beta * nu * alpha / F**(1-beta) +
                                       (2 - 3 * rho**2) / 24 * nu**2) * T)
                else:
                    # General case
                    log_fk = np.log(F / K)
                    fk_mid = (F * K)**((1 - beta) / 2)
                    z = nu / alpha * fk_mid * log_fk
                    x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))

                    denom = fk_mid * (1 + (1 - beta)**2 / 24 * log_fk**2 +
                                     (1 - beta)**4 / 1920 * log_fk**4)

                    term1 = alpha / denom
                    term2 = z / x_z if abs(x_z) > 1e-10 else 1

                    correction = 1 + ((1 - beta)**2 / 24 * alpha**2 / fk_mid**2 +
                                     0.25 * rho * beta * nu * alpha / fk_mid +
                                     (2 - 3 * rho**2) / 24 * nu**2) * T

                    iv = term1 * term2 * correction

                self.iv_grid[i, j] = max(iv, 0.01)

        return self.strikes, self.maturities, self.iv_grid

    def fit(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        iv_data: np.ndarray
    ):
        """
        Fit surface interpolator to observed IV data.

        Args:
            strikes: Array of strikes
            maturities: Array of maturities
            iv_data: 2D array of IVs (maturities x strikes)
        """
        self.strikes = strikes
        self.maturities = maturities
        self.iv_grid = iv_data

        # Bivariate spline interpolation
        self.surface = RectBivariateSpline(maturities, strikes, iv_data)

    def get_iv(
        self,
        strike: float = None,
        maturity: float = None,
        moneyness: float = None,
        S: float = 100
    ) -> float:
        """
        Get interpolated IV at given point.

        Args:
            strike: Strike price (or use moneyness)
            maturity: Time to expiration
            moneyness: K/S ratio (alternative to strike)
            S: Spot price (needed if using moneyness)

        Returns:
            Interpolated implied volatility
        """
        if moneyness is not None:
            strike = moneyness * S

        if self.surface is None:
            raise ValueError("Surface not fitted. Call fit() or generate_sabr_surface() first.")

        return float(self.surface(maturity, strike)[0, 0])

    def analyze_skew(self, maturity: float, S: float = 100) -> Dict[str, float]:
        """
        Analyze volatility skew at given maturity.

        Returns various skew metrics:
            - atm_vol: ATM implied volatility
            - skew_25d: 25-delta put IV - 25-delta call IV
            - butterfly_25d: (25d put + 25d call) / 2 - ATM (smile curvature)
            - risk_reversal: 25d call - 25d put (directional skew)

        Args:
            maturity: Time to expiration
            S: Spot price

        Returns:
            Dictionary of skew metrics
        """
        # Define key strikes
        atm_strike = S
        otm_put_strike = 0.9 * S  # Approx 25-delta put
        otm_call_strike = 1.1 * S  # Approx 25-delta call

        atm_vol = self.get_iv(strike=atm_strike, maturity=maturity)
        put_vol = self.get_iv(strike=otm_put_strike, maturity=maturity)
        call_vol = self.get_iv(strike=otm_call_strike, maturity=maturity)

        return {
            'atm_vol': atm_vol,
            'put_25d_vol': put_vol,
            'call_25d_vol': call_vol,
            'skew_25d': put_vol - call_vol,
            'butterfly_25d': (put_vol + call_vol) / 2 - atm_vol,
            'risk_reversal': call_vol - put_vol,
            'skew_slope': (put_vol - call_vol) / (0.2 * S)  # Per unit moneyness
        }

    def term_structure(
        self,
        strike: float = None,
        moneyness: float = 1.0,
        S: float = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract term structure at given strike/moneyness.

        Args:
            strike: Strike price
            moneyness: K/S ratio
            S: Spot price

        Returns:
            (maturities, ivs) tuple
        """
        if strike is None:
            strike = moneyness * S

        ivs = np.array([self.get_iv(strike=strike, maturity=T) for T in self.maturities])

        return self.maturities, ivs


class GreeksSurface:
    """
    Greeks Surface Generator for 3D Risk Visualization.

    Creates 3D surfaces of option Greeks across strike and maturity,
    enabling comprehensive risk visualization.

    Key Insights:
        - Gamma Surface: Pin risk concentration (peak near ATM, short maturity)
        - Vega Surface: Volatility exposure distribution
        - Vanna Surface: Skew risk exposure
        - Volga Surface: Vol-of-vol exposure

    Example:
        >>> gs = GreeksSurface()
        >>> K, T, gamma = gs.gamma_surface(S=100)
        >>> K, T, vanna = gs.vanna_surface(S=100)
    """

    def __init__(
        self,
        S: float = 100,
        r: float = 0.05,
        sigma: float = 0.2,
        n_strikes: int = 50,
        n_maturities: int = 50
    ):
        """
        Initialize Greeks surface generator.

        Args:
            S: Spot price
            r: Risk-free rate
            sigma: Baseline volatility
            n_strikes: Number of strike points
            n_maturities: Number of maturity points
        """
        self.S = S
        self.r = r
        self.sigma = sigma
        self.bs = BlackScholes()

        # Grid setup
        self.strikes = np.linspace(0.7 * S, 1.3 * S, n_strikes)
        self.maturities = np.linspace(0.02, 2.0, n_maturities)

        # Create meshgrid
        self.K_grid, self.T_grid = np.meshgrid(self.strikes, self.maturities)

    def _compute_surface(self, greek_func) -> np.ndarray:
        """Compute Greek surface using vectorized operations."""
        surface = np.zeros_like(self.K_grid)

        for i in range(len(self.maturities)):
            for j in range(len(self.strikes)):
                surface[i, j] = greek_func(
                    self.S, self.strikes[j], self.maturities[i], self.r, self.sigma
                )

        return surface

    def delta_surface(
        self,
        option_type: OptionType = OptionType.CALL
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate Delta surface."""
        delta_func = lambda S, K, T, r, sig: self.bs.delta(S, K, T, r, sig, 0, option_type)
        surface = self._compute_surface(delta_func)
        return self.K_grid, self.T_grid, surface

    def gamma_surface(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate Gamma surface.

        Shows concentration of convexity risk.
        Peak at ATM with short maturity = maximum pin risk.
        """
        surface = self._compute_surface(self.bs.gamma)
        return self.K_grid, self.T_grid, surface

    def vega_surface(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate Vega surface.

        Shows volatility exposure across strike/maturity space.
        Longest maturities have highest vega exposure.
        """
        surface = self._compute_surface(self.bs.vega)
        return self.K_grid, self.T_grid, surface

    def theta_surface(
        self,
        option_type: OptionType = OptionType.CALL
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate Theta surface."""
        theta_func = lambda S, K, T, r, sig: self.bs.theta(S, K, T, r, sig, 0, option_type)
        surface = self._compute_surface(theta_func)
        return self.K_grid, self.T_grid, surface

    def vanna_surface(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate Vanna surface (∂Δ/∂σ).

        Critical for understanding how hedging changes with volatility.
        Positive vanna = delta increases with volatility.
        """
        surface = self._compute_surface(self.bs.vanna)
        return self.K_grid, self.T_grid, surface

    def volga_surface(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate Volga surface (∂²V/∂σ²).

        Shows vega convexity. Positive volga profits from vol-of-vol.
        Key for volatility trading strategies.
        """
        surface = self._compute_surface(self.bs.volga)
        return self.K_grid, self.T_grid, surface

    def all_surfaces(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate all Greek surfaces.

        Returns:
            Dictionary mapping Greek name to (K_grid, T_grid, values) tuple
        """
        return {
            'delta_call': self.delta_surface(OptionType.CALL),
            'delta_put': self.delta_surface(OptionType.PUT),
            'gamma': self.gamma_surface(),
            'vega': self.vega_surface(),
            'theta_call': self.theta_surface(OptionType.CALL),
            'theta_put': self.theta_surface(OptionType.PUT),
            'vanna': self.vanna_surface(),
            'volga': self.volga_surface()
        }


def generate_smile_from_skew(
    S: float,
    T: float,
    atm_vol: float,
    skew: float,
    convexity: float,
    n_strikes: int = 21
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate volatility smile from skew parameters.

    Uses quadratic approximation:
        σ(K) = σ_ATM + skew * (K/S - 1) + convexity * (K/S - 1)²

    Args:
        S: Spot price
        T: Maturity
        atm_vol: ATM volatility
        skew: Linear skew coefficient (negative = normal skew)
        convexity: Quadratic coefficient (positive = smile)
        n_strikes: Number of strike points

    Returns:
        (strikes, ivs) tuple
    """
    strikes = np.linspace(0.7 * S, 1.3 * S, n_strikes)
    moneyness = strikes / S - 1

    ivs = atm_vol + skew * moneyness + convexity * moneyness**2

    return strikes, ivs
