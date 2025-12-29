"""
Phase Space Visualization for Financial Time Series.

Specialized visualizations for:
    - Takens embedding reconstruction
    - Attractor analysis
    - Lyapunov exponent visualization
    - Regime dynamics in phase space
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List


class PhaseSpaceVisualization:
    """
    Phase Space Visualization Suite.

    Creates publication-quality visualizations of reconstructed
    attractors and dynamical system analyses.

    Example:
        >>> viz = PhaseSpaceVisualization()
        >>> viz.plot_attractor_reconstruction(prices, tau=5, m=3)
        >>> viz.plot_recurrence_plot(trajectory)
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 10),
        style: str = 'dark'
    ):
        """
        Initialize visualization.

        Args:
            figsize: Figure size
            style: 'dark' or 'light' theme
        """
        self.figsize = figsize
        self.style = style

        if style == 'dark':
            plt.style.use('dark_background')
            self.colors = ['#00d4ff', '#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3']
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    def plot_attractor_reconstruction(
        self,
        series: np.ndarray,
        tau: int = 1,
        m: int = 3,
        title: str = "Attractor Reconstruction from Price Series",
        color_by: str = 'time',
        save_path: str = None
    ) -> plt.Figure:
        """
        Reconstruct and plot attractor from 1D time series.

        Uses Takens embedding theorem:
            v(t) = [x(t), x(t-τ), x(t-2τ), ...]

        Args:
            series: 1D time series (prices or returns)
            tau: Time delay
            m: Embedding dimension (2 or 3 for visualization)
            title: Plot title
            color_by: 'time', 'value', or 'velocity'
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        # Create embedding
        n = len(series) - (m - 1) * tau
        embedded = np.zeros((n, m))
        for i in range(m):
            embedded[:, i] = series[i * tau:i * tau + n]

        if m == 2:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=150)

            if color_by == 'time':
                colors = np.linspace(0, 1, n)
            elif color_by == 'value':
                colors = embedded[:, 0]
            else:  # velocity
                colors = np.diff(embedded[:, 0], prepend=embedded[0, 0])

            scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                               c=colors, cmap='plasma', s=2, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label=color_by.capitalize())

            ax.set_xlabel(f'x(t)', fontsize=11)
            ax.set_ylabel(f'x(t - {tau})', fontsize=11)

        else:  # m == 3
            fig = plt.figure(figsize=self.figsize, dpi=150)
            ax = fig.add_subplot(111, projection='3d')

            if color_by == 'time':
                colors = np.linspace(0, 1, n)
            elif color_by == 'value':
                colors = embedded[:, 0]
            else:
                colors = np.diff(embedded[:, 0], prepend=embedded[0, 0])

            scatter = ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                               c=colors, cmap='plasma', s=2, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label=color_by.capitalize(), shrink=0.6)

            ax.set_xlabel(f'x(t)', fontsize=11)
            ax.set_ylabel(f'x(t - {tau})', fontsize=11)
            ax.set_zlabel(f'x(t - {2*tau})', fontsize=11)

        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add embedding parameters
        param_text = f'Embedding: m={m}, τ={tau}'
        if m == 2:
            ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top')
        else:
            ax.text2D(0.02, 0.98, param_text, transform=ax.transAxes,
                     fontsize=10, verticalalignment='top')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_recurrence_plot(
        self,
        series: np.ndarray,
        epsilon: float = None,
        title: str = "Recurrence Plot\nDetecting Dynamical Patterns",
        save_path: str = None
    ) -> plt.Figure:
        """
        Create recurrence plot for detecting dynamical patterns.

        R(i,j) = 1 if ||x_i - x_j|| < ε, else 0

        Patterns:
            - Diagonal lines: Deterministic dynamics
            - Vertical/horizontal: Laminar states
            - Checkered: Periodic behavior

        Args:
            series: Time series
            epsilon: Distance threshold (auto if None)
            title: Plot title
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        n = len(series)

        # Compute distance matrix
        X = series.reshape(-1, 1)
        dist_matrix = np.abs(X - X.T)

        if epsilon is None:
            epsilon = np.std(series) * 0.1

        recurrence = (dist_matrix < epsilon).astype(int)

        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

        ax.imshow(recurrence, cmap='binary', origin='lower')
        ax.set_xlabel('Time i', fontsize=11)
        ax.set_ylabel('Time j', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add interpretation
        interp_text = (
            f'ε = {epsilon:.4f}\n'
            'Diagonal lines → Deterministic\n'
            'Dots → Stochastic\n'
            'Blocks → Regimes'
        )
        ax.text(0.02, 0.98, interp_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_lyapunov_spectrum(
        self,
        exponents: np.ndarray,
        dimension: int = None,
        title: str = "Lyapunov Spectrum Analysis",
        save_path: str = None
    ) -> plt.Figure:
        """
        Visualize Lyapunov exponent spectrum.

        Args:
            exponents: Lyapunov exponents (sorted descending)
            dimension: Kaplan-Yorke dimension estimate
            title: Plot title
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=150)

        # Spectrum bar plot
        ax1 = axes[0]
        colors = ['green' if e > 0 else 'red' for e in exponents]
        bars = ax1.bar(range(len(exponents)), exponents, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='white', linestyle='--', linewidth=1)
        ax1.set_xlabel('Exponent Index', fontsize=11)
        ax1.set_ylabel('Lyapunov Exponent λ', fontsize=11)
        ax1.set_title('Lyapunov Spectrum', fontsize=12, fontweight='bold')

        # Cumulative sum for Kaplan-Yorke dimension
        ax2 = axes[1]
        cumsum = np.cumsum(exponents)
        ax2.plot(range(len(exponents)), cumsum, 'o-', color=self.colors[0])
        ax2.axhline(y=0, color='white', linestyle='--', linewidth=1)
        ax2.fill_between(range(len(exponents)), 0, cumsum,
                        where=(cumsum > 0), alpha=0.3, color='green')
        ax2.fill_between(range(len(exponents)), 0, cumsum,
                        where=(cumsum <= 0), alpha=0.3, color='red')
        ax2.set_xlabel('Exponent Index', fontsize=11)
        ax2.set_ylabel('Cumulative Sum', fontsize=11)
        ax2.set_title('Kaplan-Yorke Dimension Estimate', fontsize=12, fontweight='bold')

        if dimension:
            ax2.text(0.95, 0.95, f'D_KY ≈ {dimension:.2f}',
                    transform=ax2.transAxes, fontsize=12,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_poincare_section(
        self,
        trajectory: np.ndarray,
        plane_coord: int = 2,
        plane_value: float = None,
        title: str = "Poincaré Section\nDimensional Reduction of Dynamics",
        save_path: str = None
    ) -> plt.Figure:
        """
        Create Poincaré section by intersecting trajectory with plane.

        Args:
            trajectory: 3D trajectory (n_points, 3)
            plane_coord: Coordinate defining plane (0, 1, or 2)
            plane_value: Value at which to slice
            title: Plot title
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        if plane_value is None:
            plane_value = np.mean(trajectory[:, plane_coord])

        # Find crossings
        crossings = []
        for i in range(len(trajectory) - 1):
            if ((trajectory[i, plane_coord] < plane_value and
                 trajectory[i+1, plane_coord] > plane_value) or
                (trajectory[i, plane_coord] > plane_value and
                 trajectory[i+1, plane_coord] < plane_value)):
                # Linear interpolation
                t = (plane_value - trajectory[i, plane_coord]) / \
                    (trajectory[i+1, plane_coord] - trajectory[i, plane_coord])
                point = trajectory[i] + t * (trajectory[i+1] - trajectory[i])
                crossings.append(point)

        crossings = np.array(crossings)

        if len(crossings) == 0:
            print("No crossings found!")
            return None

        # Get the two remaining coordinates
        coords = [i for i in range(3) if i != plane_coord]

        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

        ax.scatter(crossings[:, coords[0]], crossings[:, coords[1]],
                  c=np.arange(len(crossings)), cmap='viridis', s=10, alpha=0.6)

        coord_labels = ['x', 'y', 'z']
        ax.set_xlabel(coord_labels[coords[0]], fontsize=11)
        ax.set_ylabel(coord_labels[coords[1]], fontsize=11)
        ax.set_title(f'{title}\n{coord_labels[plane_coord]} = {plane_value:.2f} plane',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_bifurcation_diagram(
        self,
        parameter_range: np.ndarray,
        attractors: List[np.ndarray],
        parameter_name: str = 'ρ',
        title: str = "Bifurcation Diagram\nRoute to Chaos",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot bifurcation diagram showing route to chaos.

        Args:
            parameter_range: Array of parameter values
            attractors: List of attractor samples for each parameter
            parameter_name: Name of bifurcation parameter
            title: Plot title
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=150)

        for p, attractor in zip(parameter_range, attractors):
            # Plot last N points of attractor (after transient)
            n_plot = min(200, len(attractor))
            ax.scatter([p] * n_plot, attractor[-n_plot:],
                      s=0.5, c='white', alpha=0.5)

        ax.set_xlabel(f'Parameter {parameter_name}', fontsize=11)
        ax.set_ylabel('Attractor Values', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_shadow_manifold(
        self,
        trajectory: np.ndarray,
        title: str = "Shadow Manifold Projections",
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot 2D projections of 3D trajectory (shadow manifolds).

        Useful for understanding the structure of the attractor
        from different viewing angles.

        Args:
            trajectory: 3D trajectory (n_points, 3)
            title: Plot title
            save_path: Optional save path

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=150)

        projections = [
            (0, 1, 'x', 'y'),
            (0, 2, 'x', 'z'),
            (1, 2, 'y', 'z')
        ]

        for ax, (i, j, xlabel, ylabel) in zip(axes.flat[:3], projections):
            colors = np.linspace(0, 1, len(trajectory))
            ax.scatter(trajectory[:, i], trajectory[:, j],
                      c=colors, cmap='plasma', s=1, alpha=0.5)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f'{xlabel}-{ylabel} projection', fontsize=11)

        # 3D view in last subplot
        ax3d = fig.add_subplot(2, 2, 4, projection='3d')
        colors = np.linspace(0, 1, len(trajectory))
        ax3d.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                    c=colors, cmap='plasma', s=1, alpha=0.5)
        ax3d.set_xlabel('x', fontsize=10)
        ax3d.set_ylabel('y', fontsize=10)
        ax3d.set_zlabel('z', fontsize=10)
        ax3d.set_title('3D View', fontsize=11)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
