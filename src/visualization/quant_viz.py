"""
Quantitative Finance Visualization Suite

Professional-grade 3D visualizations for physics-based financial modeling.
Designed for quantitative researchers and portfolio managers.

Key Visualizations:
    1. Lorenz Attractor: Market phase space dynamics
    2. IV Surface: Topology of market fear
    3. Greeks Surfaces: Multi-dimensional risk exposure
    4. Risk-Return-Time: 3D efficient frontier evolution
    5. Fokker-Planck: Probability density evolution
    6. Regime Dynamics: HMM state transitions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from typing import Tuple, Dict, Optional, List
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)


class Arrow3D(FancyArrowPatch):
    """3D arrow for vector field visualization."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class QuantVisualization:
    """
    Comprehensive Visualization Suite for Quantitative Finance.

    Provides publication-quality 3D plots for physics-based financial models.

    Example:
        >>> viz = QuantVisualization(figsize=(14, 10), style='dark')
        >>> viz.plot_lorenz_attractor(trajectory)
        >>> viz.plot_iv_surface(strikes, maturities, iv_grid)
        >>> viz.plot_risk_return_time(portfolios)
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 10),
        style: str = 'dark',
        dpi: int = 150
    ):
        """
        Initialize visualization suite.

        Args:
            figsize: Default figure size
            style: 'dark' or 'light' theme
            dpi: Figure resolution
        """
        self.figsize = figsize
        self.style = style
        self.dpi = dpi

        # Set style
        if style == 'dark':
            plt.style.use('dark_background')
            self.bg_color = '#1a1a2e'
            self.text_color = '#ffffff'
            self.grid_color = '#333355'
            self.accent_colors = ['#00d4ff', '#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3']
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
            self.bg_color = '#ffffff'
            self.text_color = '#000000'
            self.grid_color = '#cccccc'
            self.accent_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    def plot_lorenz_attractor(
        self,
        trajectory: np.ndarray,
        title: str = "Lorenz Strange Attractor\nMarket Phase Space Dynamics",
        color_by_z: bool = True,
        show_fixed_points: bool = True,
        sigma: float = 13.0,
        rho: float = 28.0,
        beta: float = 8/3,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot 3D Lorenz attractor with financial interpretation.

        Args:
            trajectory: State trajectory (n_points, 3)
            title: Plot title
            color_by_z: Color trajectory by z-coordinate (volatility)
            show_fixed_points: Mark fixed points of the system
            sigma, rho, beta: Lorenz parameters
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')

        x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

        if color_by_z:
            # Color by z-coordinate (volatility proxy)
            points = trajectory.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Normalize z for coloring
            norm = plt.Normalize(z.min(), z.max())
            colors = cm.plasma(norm(z))

            for i in range(len(segments)):
                ax.plot(
                    [segments[i, 0, 0], segments[i, 1, 0]],
                    [segments[i, 0, 1], segments[i, 1, 1]],
                    [segments[i, 0, 2], segments[i, 1, 2]],
                    color=colors[i],
                    alpha=0.7,
                    linewidth=0.5
                )

            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label('z (Volatility Proxy)', color=self.text_color)
        else:
            ax.plot(x, y, z, color=self.accent_colors[0], alpha=0.7, linewidth=0.5)

        # Show fixed points
        if show_fixed_points:
            c = np.sqrt(beta * (rho - 1))
            fixed_points = [
                (0, 0, 0),
                (c, c, rho - 1),
                (-c, -c, rho - 1)
            ]
            for i, fp in enumerate(fixed_points):
                ax.scatter(*fp, s=100, c=self.accent_colors[1], marker='o',
                          edgecolors='white', linewidths=2, zorder=5)
                ax.text(fp[0], fp[1], fp[2] + 2,
                       f'FP{i+1}', fontsize=10, color=self.text_color)

        # Labels and formatting
        ax.set_xlabel('x (Momentum / Order Flow)', fontsize=11, color=self.text_color)
        ax.set_ylabel('y (Price Deviation)', fontsize=11, color=self.text_color)
        ax.set_zlabel('z (Volatility / System Energy)', fontsize=11, color=self.text_color)
        ax.set_title(title, fontsize=14, fontweight='bold', color=self.text_color, pad=20)

        # Add parameter annotation
        param_text = f'σ={sigma:.1f} (LORENZ_sigma_13)  ρ={rho:.1f}  β={beta:.2f}'
        ax.text2D(0.02, 0.98, param_text, transform=ax.transAxes,
                 fontsize=9, color=self.accent_colors[0], verticalalignment='top')

        # Interpretation legend
        interp_text = (
            "Interpretation:\n"
            "• Low z: Stable regime (trend-following)\n"
            "• High z: Turbulent regime (mean-reversion)\n"
            "• Wings: Market oscillation patterns"
        )
        ax.text2D(0.02, 0.15, interp_text, transform=ax.transAxes,
                 fontsize=8, color=self.text_color, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        ax.set_facecolor(self.bg_color)
        fig.patch.set_facecolor(self.bg_color)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')

        return fig

    def plot_iv_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        iv_grid: np.ndarray,
        spot: float = 100,
        title: str = "Implied Volatility Surface\nTopology of Market Fear",
        show_atm: bool = True,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot 3D implied volatility surface.

        Args:
            strikes: Strike prices
            maturities: Time to expiration
            iv_grid: IV values (maturities x strikes)
            spot: Current spot price
            title: Plot title
            show_atm: Highlight ATM line
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')

        # Create meshgrid
        K, T = np.meshgrid(strikes, maturities)

        # Convert strikes to moneyness
        moneyness = K / spot

        # Plot surface
        surf = ax.plot_surface(
            moneyness, T, iv_grid * 100,  # IV in percentage
            cmap=cm.RdYlBu_r,
            alpha=0.9,
            linewidth=0.2,
            antialiased=True,
            edgecolors='white'
        )

        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Implied Volatility (%)', color=self.text_color)

        # ATM line
        if show_atm:
            atm_ivs = np.array([iv_grid[i, len(strikes)//2] * 100 for i in range(len(maturities))])
            ax.plot(
                np.ones(len(maturities)),  # Moneyness = 1
                maturities,
                atm_ivs,
                color=self.accent_colors[1],
                linewidth=3,
                label='ATM IV',
                zorder=10
            )

        # Skew annotation
        skew_25d = iv_grid[:, len(strikes)//4] - iv_grid[:, 3*len(strikes)//4]
        avg_skew = np.mean(skew_25d) * 100

        ax.set_xlabel('Moneyness (K/S)', fontsize=11, color=self.text_color)
        ax.set_ylabel('Time to Expiration (years)', fontsize=11, color=self.text_color)
        ax.set_zlabel('Implied Volatility (%)', fontsize=11, color=self.text_color)
        ax.set_title(title, fontsize=14, fontweight='bold', color=self.text_color, pad=20)

        # Interpretation
        interp_text = (
            f"Surface Analysis:\n"
            f"• Avg 25Δ Skew: {avg_skew:.1f}%\n"
            f"• Skew > 0: Put protection premium\n"
            f"• Smile: Fat tail compensation"
        )
        ax.text2D(0.02, 0.15, interp_text, transform=ax.transAxes,
                 fontsize=8, color=self.text_color, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        ax.set_facecolor(self.bg_color)
        fig.patch.set_facecolor(self.bg_color)
        ax.view_init(elev=25, azim=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')

        return fig

    def plot_greeks_surfaces(
        self,
        K_grid: np.ndarray,
        T_grid: np.ndarray,
        greeks: Dict[str, np.ndarray],
        spot: float = 100,
        title: str = "Black-Scholes Greek Surfaces\n3D Risk Exposure Mapping",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot multiple Greeks surfaces in a grid.

        Args:
            K_grid: Strike meshgrid
            T_grid: Maturity meshgrid
            greeks: Dict mapping Greek name to surface values
            spot: Spot price for moneyness calculation
            title: Main title
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        n_greeks = len(greeks)
        n_cols = min(3, n_greeks)
        n_rows = (n_greeks + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] * n_rows / 2), dpi=self.dpi)

        moneyness = K_grid / spot

        greek_cmaps = {
            'delta': cm.RdYlGn,
            'gamma': cm.hot,
            'vega': cm.YlOrRd,
            'theta': cm.Blues_r,
            'vanna': cm.PuOr,
            'volga': cm.PRGn
        }

        greek_labels = {
            'delta': 'Δ: Price Sensitivity',
            'gamma': 'Γ: Delta Convexity',
            'vega': 'ν: Volatility Exposure',
            'theta': 'Θ: Time Decay',
            'vanna': '∂Δ/∂σ: Cross Sensitivity',
            'volga': '∂ν/∂σ: Vega Convexity'
        }

        for i, (greek_name, values) in enumerate(greeks.items()):
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')

            cmap = greek_cmaps.get(greek_name.lower(), cm.viridis)

            surf = ax.plot_surface(
                moneyness, T_grid, values,
                cmap=cmap,
                alpha=0.9,
                linewidth=0.1,
                antialiased=True
            )

            ax.set_xlabel('Moneyness', fontsize=9, color=self.text_color)
            ax.set_ylabel('Maturity (y)', fontsize=9, color=self.text_color)
            ax.set_zlabel(greek_name, fontsize=9, color=self.text_color)

            label = greek_labels.get(greek_name.lower(), greek_name)
            ax.set_title(label, fontsize=11, color=self.text_color)

            ax.set_facecolor(self.bg_color)
            ax.view_init(elev=30, azim=45)

        fig.suptitle(title, fontsize=14, fontweight='bold', color=self.text_color, y=1.02)
        fig.patch.set_facecolor(self.bg_color)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')

        return fig

    def plot_risk_return_time(
        self,
        returns: np.ndarray,
        volatilities: np.ndarray,
        times: np.ndarray,
        labels: List[str] = None,
        title: str = "Risk-Return-Time 3D Coordinate System\nPortfolio Evolution in Space-Time",
        show_efficient_frontier: bool = True,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot 3D Risk-Return-Time coordinate system.

        This is a key visualization showing how portfolios evolve through
        the risk-return space over time.

        Args:
            returns: Expected returns array
            volatilities: Volatility (risk) array
            times: Time points
            labels: Optional point labels
            title: Plot title
            show_efficient_frontier: Show efficient frontier at each time
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')

        # Create time-color gradient
        norm = plt.Normalize(times.min(), times.max())
        colors = cm.viridis(norm(times))

        # Plot trajectories
        if returns.ndim == 2:
            # Multiple portfolios
            n_portfolios = returns.shape[0]
            for p in range(n_portfolios):
                ax.plot(
                    volatilities[p], returns[p], times,
                    color=self.accent_colors[p % len(self.accent_colors)],
                    linewidth=2,
                    alpha=0.8,
                    label=labels[p] if labels else f'Portfolio {p+1}'
                )
                # End point marker
                ax.scatter(
                    volatilities[p, -1], returns[p, -1], times[-1],
                    s=100, c=[colors[-1]], marker='o', edgecolors='white'
                )
        else:
            # Single trajectory
            ax.scatter(volatilities, returns, times, c=times, cmap='viridis', s=50)
            ax.plot(volatilities, returns, times, color=self.accent_colors[0], alpha=0.5)

        # Efficient frontier surfaces
        if show_efficient_frontier:
            # Sample times for frontier surfaces
            frontier_times = [times[0], times[len(times)//2], times[-1]]

            for t in frontier_times:
                t_idx = np.argmin(np.abs(times - t))
                # Generate efficient frontier curve
                vols = np.linspace(0.05, 0.5, 20)
                rets = 0.02 + 0.3 * vols - 0.2 * vols**2  # Parabolic frontier

                ax.plot(
                    vols, rets, np.full_like(vols, t),
                    color='white', alpha=0.3, linewidth=1, linestyle='--'
                )

        # Add labels
        ax.set_xlabel('Risk (Volatility σ)', fontsize=11, color=self.text_color)
        ax.set_ylabel('Expected Return (μ)', fontsize=11, color=self.text_color)
        ax.set_zlabel('Time', fontsize=11, color=self.text_color)
        ax.set_title(title, fontsize=14, fontweight='bold', color=self.text_color, pad=20)

        # Add interpretation
        interp_text = (
            "Interpretation:\n"
            "• Movement UP: Increasing returns\n"
            "• Movement RIGHT: Increasing risk\n"
            "• Trajectory slope: Risk-adjusted trend\n"
            "• Curvature: Regime transitions"
        )
        ax.text2D(0.02, 0.15, interp_text, transform=ax.transAxes,
                 fontsize=8, color=self.text_color, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        if labels:
            ax.legend(loc='upper right', fontsize=9)

        ax.set_facecolor(self.bg_color)
        fig.patch.set_facecolor(self.bg_color)
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')

        return fig

    def plot_fokker_planck_evolution(
        self,
        x: np.ndarray,
        times: np.ndarray,
        pdfs: np.ndarray,
        title: str = "Fokker-Planck Probability Evolution\nPrice Distribution Dynamics",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot evolution of probability density over time.

        Shows how the PDF evolves, potentially splitting into bimodal
        distributions indicating regime uncertainty.

        Args:
            x: State space grid
            times: Time points
            pdfs: PDF values (n_times, n_x)
            title: Plot title
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')

        X, T = np.meshgrid(x, times)

        # Plot surface
        surf = ax.plot_surface(
            X, T, pdfs,
            cmap=cm.magma,
            alpha=0.9,
            linewidth=0.1,
            antialiased=True
        )

        # Add colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Probability Density', color=self.text_color)

        # Highlight specific time slices
        for i in [0, len(times)//2, -1]:
            ax.plot(
                x, np.full_like(x, times[i]), pdfs[i],
                color=self.accent_colors[0], linewidth=2, alpha=0.8
            )

        ax.set_xlabel('Price / Return (x)', fontsize=11, color=self.text_color)
        ax.set_ylabel('Time', fontsize=11, color=self.text_color)
        ax.set_zlabel('Probability Density P(x,t)', fontsize=11, color=self.text_color)
        ax.set_title(title, fontsize=14, fontweight='bold', color=self.text_color, pad=20)

        interp_text = (
            "Key Observations:\n"
            "• Spreading: Diffusion (σ)\n"
            "• Drift: Mean movement (μ)\n"
            "• Bimodal: Regime bifurcation\n"
            "• Fat tails: Jump risk"
        )
        ax.text2D(0.02, 0.15, interp_text, transform=ax.transAxes,
                 fontsize=8, color=self.text_color, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        ax.set_facecolor(self.bg_color)
        fig.patch.set_facecolor(self.bg_color)
        ax.view_init(elev=25, azim=-60)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')

        return fig

    def plot_regime_transitions(
        self,
        regime_probs: np.ndarray,
        times: np.ndarray,
        regime_names: List[str] = None,
        title: str = "Hidden Markov Model Regime Probabilities\nMarket State Detection",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot regime probability evolution over time.

        Args:
            regime_probs: Regime probabilities (n_times, n_regimes)
            times: Time points
            regime_names: Optional regime names
            title: Plot title
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        n_regimes = regime_probs.shape[1]

        if regime_names is None:
            regime_names = [f'Regime {i+1}' for i in range(n_regimes)]

        # Stacked area plot
        ax.stackplot(
            times, regime_probs.T,
            labels=regime_names,
            colors=self.accent_colors[:n_regimes],
            alpha=0.8
        )

        ax.set_xlabel('Time', fontsize=11, color=self.text_color)
        ax.set_ylabel('Regime Probability', fontsize=11, color=self.text_color)
        ax.set_title(title, fontsize=14, fontweight='bold', color=self.text_color)
        ax.set_ylim(0, 1)

        ax.legend(loc='upper right', fontsize=9)

        ax.set_facecolor(self.bg_color)
        fig.patch.set_facecolor(self.bg_color)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')

        return fig

    def plot_phase_portrait(
        self,
        trajectory: np.ndarray,
        regimes: np.ndarray = None,
        title: str = "Phase Portrait with Regime Coloring\nMarket Dynamics Visualization",
        save_path: str = None
    ) -> plt.Figure:
        """
        2D phase portrait colored by regime.

        Args:
            trajectory: State trajectory (n_points, 2 or 3)
            regimes: Optional regime labels for coloring
            title: Plot title
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.dpi)

        x, y = trajectory[:, 0], trajectory[:, 1]

        if regimes is not None:
            # Color by regime
            scatter = ax.scatter(
                x, y,
                c=regimes,
                cmap='Set1',
                s=2,
                alpha=0.6
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Regime', color=self.text_color)
        else:
            # Color by time
            colors = np.linspace(0, 1, len(x))
            scatter = ax.scatter(
                x, y,
                c=colors,
                cmap='viridis',
                s=2,
                alpha=0.6
            )

        ax.set_xlabel('x (Momentum)', fontsize=11, color=self.text_color)
        ax.set_ylabel('y (Price Deviation)', fontsize=11, color=self.text_color)
        ax.set_title(title, fontsize=14, fontweight='bold', color=self.text_color)

        ax.set_facecolor(self.bg_color)
        fig.patch.set_facecolor(self.bg_color)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor=self.bg_color, edgecolor='none')

        return fig
