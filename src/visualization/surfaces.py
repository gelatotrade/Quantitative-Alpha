"""
Surface Plotting Utilities for Quantitative Finance.

Specialized 3D surface visualizations for:
    - Implied Volatility surfaces
    - Greeks surfaces
    - Risk-Return optimization surfaces
    - Kelly Criterion growth surfaces
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, Dict, List


class SurfacePlotter:
    """
    Advanced 3D Surface Plotter for Financial Data.

    Provides interactive and publication-quality surface plots
    with financial annotations and interpretations.

    Example:
        >>> plotter = SurfacePlotter()
        >>> plotter.plot_kelly_growth_surface(returns, sigmas, kelly_f)
        >>> plotter.plot_var_surface(confidence, horizons, var_values)
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 10),
        colormap: str = 'viridis'
    ):
        """
        Initialize surface plotter.

        Args:
            figsize: Default figure size
            colormap: Default colormap
        """
        self.figsize = figsize
        self.colormap = colormap

    def plot_kelly_growth_surface(
        self,
        mu_range: np.ndarray,
        sigma_range: np.ndarray,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot Kelly growth rate surface as function of return and volatility.

        The surface shows the expected geometric growth rate:
            g(f) = μ - σ²f²/2

        At optimal Kelly: g* = μ²/(2σ²)

        Args:
            mu_range: Range of expected returns
            sigma_range: Range of volatilities
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=self.figsize, dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        MU, SIGMA = np.meshgrid(mu_range, sigma_range)

        # Optimal Kelly fraction
        F_KELLY = MU / (SIGMA ** 2)

        # Growth rate at optimal Kelly
        GROWTH = MU ** 2 / (2 * SIGMA ** 2)

        # Cap extreme values for visualization
        GROWTH = np.clip(GROWTH, 0, 1)

        surf = ax.plot_surface(
            SIGMA * 100, MU * 100, GROWTH * 100,
            cmap=cm.RdYlGn,
            alpha=0.9,
            linewidth=0.1,
            antialiased=True
        )

        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Optimal Growth Rate (%)', fontsize=10)

        ax.set_xlabel('Volatility σ (%)', fontsize=11)
        ax.set_ylabel('Expected Return μ (%)', fontsize=11)
        ax.set_zlabel('Growth Rate g* (%)', fontsize=11)
        ax.set_title('Kelly Criterion: Optimal Growth Surface\ng* = μ²/(2σ²)',
                    fontsize=14, fontweight='bold')

        # Add contour projection
        ax.contour(
            SIGMA * 100, MU * 100, GROWTH * 100,
            zdir='z', offset=0, cmap=cm.RdYlGn, alpha=0.5
        )

        ax.view_init(elev=25, azim=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_var_surface(
        self,
        confidences: np.ndarray,
        horizons: np.ndarray,
        returns: np.ndarray,
        sigma: float = 0.2,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot VaR surface across confidence levels and time horizons.

        VaR(α, T) = -σ√T × Φ⁻¹(1-α)

        Args:
            confidences: Confidence levels (e.g., 0.90 to 0.99)
            horizons: Time horizons in days
            returns: Historical returns for estimation
            sigma: Annualized volatility
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        from scipy.stats import norm

        fig = plt.figure(figsize=self.figsize, dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        CONF, HORIZ = np.meshgrid(confidences, horizons)

        # VaR calculation (parametric)
        z_scores = norm.ppf(1 - CONF)
        daily_sigma = sigma / np.sqrt(252)
        VAR = -z_scores * daily_sigma * np.sqrt(HORIZ)

        surf = ax.plot_surface(
            CONF * 100, HORIZ, VAR * 100,
            cmap=cm.Reds,
            alpha=0.9,
            linewidth=0.1,
            antialiased=True
        )

        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Value at Risk (%)', fontsize=10)

        ax.set_xlabel('Confidence Level (%)', fontsize=11)
        ax.set_ylabel('Horizon (days)', fontsize=11)
        ax.set_zlabel('VaR (%)', fontsize=11)
        ax.set_title('Value at Risk Surface\nParametric VaR = σ√T × z_α',
                    fontsize=14, fontweight='bold')

        ax.view_init(elev=20, azim=135)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_sharpe_surface(
        self,
        returns_range: np.ndarray,
        volatility_range: np.ndarray,
        risk_free: float = 0.02,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot Sharpe ratio surface.

        Sharpe = (μ - r_f) / σ

        Args:
            returns_range: Range of returns
            volatility_range: Range of volatilities
            risk_free: Risk-free rate
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=self.figsize, dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        MU, SIGMA = np.meshgrid(returns_range, volatility_range)
        SHARPE = (MU - risk_free) / SIGMA

        # Cap for visualization
        SHARPE = np.clip(SHARPE, -3, 3)

        surf = ax.plot_surface(
            SIGMA * 100, MU * 100, SHARPE,
            cmap=cm.RdYlGn,
            alpha=0.9,
            linewidth=0.1,
            antialiased=True
        )

        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Sharpe Ratio', fontsize=10)

        ax.set_xlabel('Volatility σ (%)', fontsize=11)
        ax.set_ylabel('Return μ (%)', fontsize=11)
        ax.set_zlabel('Sharpe Ratio', fontsize=11)
        ax.set_title(f'Sharpe Ratio Surface (r_f = {risk_free*100:.1f}%)\n'
                    f'Sharpe = (μ - r_f) / σ',
                    fontsize=14, fontweight='bold')

        # Add zero plane
        xx, yy = np.meshgrid(volatility_range * 100, returns_range * 100)
        ax.plot_surface(xx, yy, np.zeros_like(xx),
                       alpha=0.3, color='gray')

        ax.view_init(elev=25, azim=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_correlation_surface(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        window_range: np.ndarray,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot rolling correlation surface between two assets.

        Shows how correlation evolves over different lookback windows.

        Args:
            returns1: Returns of first asset
            returns2: Returns of second asset
            window_range: Range of rolling windows
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=self.figsize, dpi=150)
        ax = fig.add_subplot(111, projection='3d')

        n_samples = len(returns1)
        time_indices = np.arange(max(window_range), n_samples)

        TIME, WINDOW = np.meshgrid(time_indices, window_range)
        CORR = np.zeros_like(TIME, dtype=float)

        for i, w in enumerate(window_range):
            for j, t in enumerate(time_indices):
                if t >= w:
                    r1 = returns1[t-w:t]
                    r2 = returns2[t-w:t]
                    CORR[i, j] = np.corrcoef(r1, r2)[0, 1]

        surf = ax.plot_surface(
            TIME, WINDOW, CORR,
            cmap=cm.coolwarm,
            alpha=0.9,
            linewidth=0.1,
            antialiased=True
        )

        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Correlation ρ', fontsize=10)

        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Window Size (days)', fontsize=11)
        ax.set_zlabel('Correlation', fontsize=11)
        ax.set_title('Rolling Correlation Surface\nCorrelation Dynamics Across Time Scales',
                    fontsize=14, fontweight='bold')

        ax.view_init(elev=25, azim=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_hrp_dendrogram_3d(
        self,
        linkage_matrix: np.ndarray,
        weights: np.ndarray,
        labels: List[str] = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        3D visualization of HRP hierarchical clustering with weights.

        Args:
            linkage_matrix: Scipy linkage matrix
            weights: Optimized weights
            labels: Asset labels
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        from scipy.cluster.hierarchy import dendrogram

        fig = plt.figure(figsize=self.figsize, dpi=150)

        # 2D dendrogram
        ax1 = fig.add_subplot(121)
        dend = dendrogram(linkage_matrix, labels=labels, ax=ax1,
                         leaf_rotation=45, leaf_font_size=8)
        ax1.set_title('HRP Hierarchical Clustering', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Distance (Correlation-based)')

        # 3D weights visualization
        ax2 = fig.add_subplot(122, projection='3d')

        n_assets = len(weights)
        x = np.arange(n_assets)
        y = np.zeros(n_assets)

        # Bar heights are weights
        ax2.bar3d(x, y, np.zeros(n_assets), 0.8, 0.8, weights,
                 color=cm.viridis(weights / weights.max()), alpha=0.8)

        if labels:
            ax2.set_xticks(x + 0.4)
            ax2.set_xticklabels(labels, rotation=45, fontsize=8)

        ax2.set_ylabel('', fontsize=11)
        ax2.set_zlabel('Weight', fontsize=11)
        ax2.set_title('HRP Optimal Weights', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
