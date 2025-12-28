#!/usr/bin/env python3
"""
Quantitative Alpha: Complete Visualization Generation Script

This script generates all publication-quality visualizations for the
physics-based quantitative finance framework, including:

1. Lorenz Strange Attractor (Market Dynamics)
2. Implied Volatility Surface (Market Fear Topology)
3. Greeks Surfaces (Risk Exposure Mapping)
4. Risk-Return-Time 3D Coordinate System
5. Fokker-Planck Probability Evolution
6. HMM Regime Transitions
7. Kelly Growth Surface
8. Phase Space Reconstructions

Author: Quantitative Alpha Research Team
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.lorenz import LorenzSystem, LorenzAttractor
from src.core.phase_space import PhaseSpaceReconstructor
from src.models.stochastic import FokkerPlanck, LangevinDynamics
from src.models.options import BlackScholes, ImpliedVolatilitySurface, GreeksSurface
from src.models.hmm import HiddenMarkovRegime, ViterbiDecoder
from src.visualization.quant_viz import QuantVisualization
from src.visualization.surfaces import SurfacePlotter
from src.visualization.phase_space_viz import PhaseSpaceVisualization


def ensure_output_dir():
    """Create output directory for figures."""
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    return output_dir


def generate_lorenz_attractor(viz, output_dir):
    """Generate Lorenz attractor visualization."""
    print("Generating Lorenz Attractor...")

    # Create Lorenz system with financial calibration (sigma=13)
    lorenz = LorenzSystem(sigma=13.0, rho=28.0, beta=8/3)

    # Generate trajectory
    initial_state = np.array([1.0, 1.0, 1.0])
    result = lorenz.integrate(initial_state, t_span=(0, 100), n_points=50000)

    # Plot
    fig = viz.plot_lorenz_attractor(
        result['trajectory'],
        title="Lorenz Strange Attractor\nMarket Phase Space Dynamics (σ=13)",
        sigma=13.0,
        save_path=str(output_dir / 'lorenz_attractor.png')
    )
    plt.close(fig)

    print("  ✓ Saved: lorenz_attractor.png")


def generate_iv_surface(viz, output_dir):
    """Generate implied volatility surface visualization."""
    print("Generating IV Surface...")

    # Create IV surface using SABR model
    iv_surface = ImpliedVolatilitySurface()
    strikes, maturities, iv_grid = iv_surface.generate_sabr_surface(
        S=100,
        alpha=0.25,
        beta=0.5,
        rho=-0.4,  # Negative skew
        nu=0.5,
        n_strikes=31,
        n_maturities=15
    )

    # Plot
    fig = viz.plot_iv_surface(
        strikes, maturities, iv_grid,
        spot=100,
        title="Implied Volatility Surface\nSABR Model: Topology of Market Fear",
        save_path=str(output_dir / 'iv_surface.png')
    )
    plt.close(fig)

    print("  ✓ Saved: iv_surface.png")


def generate_greeks_surfaces(viz, output_dir):
    """Generate Greeks surfaces visualization."""
    print("Generating Greeks Surfaces...")

    # Create Greeks surface generator
    gs = GreeksSurface(S=100, r=0.05, sigma=0.25, n_strikes=40, n_maturities=40)

    # Generate all surfaces
    K_gamma, T_gamma, gamma_surface = gs.gamma_surface()
    K_vega, T_vega, vega_surface = gs.vega_surface()
    K_vanna, T_vanna, vanna_surface = gs.vanna_surface()
    K_volga, T_volga, volga_surface = gs.volga_surface()

    greeks = {
        'Gamma': gamma_surface,
        'Vega': vega_surface,
        'Vanna': vanna_surface,
        'Volga': volga_surface
    }

    # Plot
    fig = viz.plot_greeks_surfaces(
        K_gamma, T_gamma, greeks,
        spot=100,
        title="Black-Scholes Greek Surfaces\n3D Risk Exposure Mapping",
        save_path=str(output_dir / 'greeks_surfaces.png')
    )
    plt.close(fig)

    print("  ✓ Saved: greeks_surfaces.png")


def generate_risk_return_time(viz, output_dir):
    """Generate Risk-Return-Time 3D visualization."""
    print("Generating Risk-Return-Time Surface...")

    # Simulate portfolio evolution
    np.random.seed(42)
    n_periods = 100
    n_portfolios = 4

    times = np.linspace(0, 2, n_periods)  # 2 years

    # Different portfolio strategies
    returns = np.zeros((n_portfolios, n_periods))
    volatilities = np.zeros((n_portfolios, n_periods))

    # Portfolio 1: Conservative (low risk, low return)
    returns[0] = 0.04 + 0.01 * np.sin(2 * np.pi * times) + 0.005 * np.cumsum(np.random.randn(n_periods))
    volatilities[0] = 0.08 + 0.01 * np.sin(2 * np.pi * times)

    # Portfolio 2: Balanced
    returns[1] = 0.08 + 0.02 * np.sin(2 * np.pi * times) + 0.01 * np.cumsum(np.random.randn(n_periods))
    volatilities[1] = 0.15 + 0.02 * np.sin(2 * np.pi * times)

    # Portfolio 3: Aggressive
    returns[2] = 0.12 + 0.03 * np.sin(2 * np.pi * times) + 0.015 * np.cumsum(np.random.randn(n_periods))
    volatilities[2] = 0.25 + 0.03 * np.sin(2 * np.pi * times)

    # Portfolio 4: Momentum (regime-dependent)
    regime_shift = times > 1
    returns[3] = np.where(regime_shift, 0.15, 0.05) + 0.02 * np.random.randn(n_periods).cumsum()
    volatilities[3] = np.where(regime_shift, 0.30, 0.10) + 0.02 * np.abs(np.random.randn(n_periods))

    labels = ['Conservative', 'Balanced', 'Aggressive', 'Momentum']

    # Plot
    fig = viz.plot_risk_return_time(
        returns, volatilities, times,
        labels=labels,
        title="Risk-Return-Time 3D Coordinate System\nPortfolio Evolution Through Space-Time",
        save_path=str(output_dir / 'risk_return_time.png')
    )
    plt.close(fig)

    print("  ✓ Saved: risk_return_time.png")


def generate_fokker_planck(viz, output_dir):
    """Generate Fokker-Planck probability evolution visualization."""
    print("Generating Fokker-Planck Evolution...")

    # Create Fokker-Planck solver
    fp = FokkerPlanck(
        drift=lambda x: -0.1 * x,  # Mean-reverting drift
        diffusion=lambda x: 0.2 * np.ones_like(x),
        x_range=(-3, 3),
        n_grid=200
    )

    # Initial condition: Gaussian
    P0 = fp.initial_gaussian(mean=0.5, std=0.2)

    # Evolve
    times, pdfs = fp.evolve_trajectory(P0, t_final=2.0, n_snapshots=20)

    # Plot
    fig = viz.plot_fokker_planck_evolution(
        fp.x, times, pdfs,
        title="Fokker-Planck Probability Evolution\nPrice Distribution Dynamics",
        save_path=str(output_dir / 'fokker_planck.png')
    )
    plt.close(fig)

    print("  ✓ Saved: fokker_planck.png")


def generate_hmm_regimes(viz, output_dir):
    """Generate HMM regime transition visualization."""
    print("Generating HMM Regime Transitions...")

    np.random.seed(42)

    # Simulate regime probabilities
    n_times = 500
    n_regimes = 3

    times = np.arange(n_times)
    regime_probs = np.zeros((n_times, n_regimes))

    # Generate regime switches
    true_regime = 0
    for t in range(n_times):
        # Add some noise and regime transitions
        if np.random.rand() < 0.02:  # 2% chance of switch
            true_regime = (true_regime + 1) % n_regimes

        # Probabilities with noise
        probs = np.ones(n_regimes) * 0.1
        probs[true_regime] = 0.8
        probs += np.random.randn(n_regimes) * 0.05
        probs = np.clip(probs, 0.01, 0.99)
        regime_probs[t] = probs / probs.sum()

    regime_names = ['Bull (Low Vol)', 'Bear (High Vol)', 'Sideways']

    # Plot
    fig = viz.plot_regime_transitions(
        regime_probs, times,
        regime_names=regime_names,
        title="Hidden Markov Model Regime Probabilities\nMarket State Detection (Viterbi Decoding)",
        save_path=str(output_dir / 'hmm_regimes.png')
    )
    plt.close(fig)

    print("  ✓ Saved: hmm_regimes.png")


def generate_kelly_surface(output_dir):
    """Generate Kelly growth surface visualization."""
    print("Generating Kelly Growth Surface...")

    plotter = SurfacePlotter()

    mu_range = np.linspace(0.02, 0.25, 50)
    sigma_range = np.linspace(0.05, 0.40, 50)

    fig = plotter.plot_kelly_growth_surface(
        mu_range, sigma_range,
        save_path=str(output_dir / 'kelly_surface.png')
    )
    plt.close(fig)

    print("  ✓ Saved: kelly_surface.png")


def generate_var_surface(output_dir):
    """Generate VaR surface visualization."""
    print("Generating VaR Surface...")

    plotter = SurfacePlotter()

    confidences = np.linspace(0.90, 0.99, 30)
    horizons = np.arange(1, 31)

    fig = plotter.plot_var_surface(
        confidences, horizons,
        returns=np.random.randn(1000) * 0.01,
        sigma=0.20,
        save_path=str(output_dir / 'var_surface.png')
    )
    plt.close(fig)

    print("  ✓ Saved: var_surface.png")


def generate_phase_space(output_dir):
    """Generate phase space visualizations."""
    print("Generating Phase Space Visualizations...")

    phase_viz = PhaseSpaceVisualization()

    # Generate synthetic price data with chaos
    np.random.seed(42)
    n_points = 2000
    t = np.linspace(0, 100, n_points)

    # Lorenz-like price dynamics
    lorenz = LorenzSystem(sigma=13.0)
    result = lorenz.integrate([1, 1, 1], t_span=(0, 50), n_points=n_points)
    prices = 100 * np.exp(0.0001 * result['x'])

    # Attractor reconstruction
    fig = phase_viz.plot_attractor_reconstruction(
        prices,
        tau=10,
        m=3,
        title="Attractor Reconstruction from Price Series\nTakens Embedding (m=3, τ=10)",
        save_path=str(output_dir / 'attractor_reconstruction.png')
    )
    plt.close(fig)

    print("  ✓ Saved: attractor_reconstruction.png")

    # Shadow manifold
    fig = phase_viz.plot_shadow_manifold(
        result['trajectory'][1000:],  # Discard transient
        title="Shadow Manifold Projections\nLorenz Attractor Views",
        save_path=str(output_dir / 'shadow_manifold.png')
    )
    plt.close(fig)

    print("  ✓ Saved: shadow_manifold.png")


def generate_sharpe_surface(output_dir):
    """Generate Sharpe ratio surface."""
    print("Generating Sharpe Surface...")

    plotter = SurfacePlotter()

    returns_range = np.linspace(-0.05, 0.30, 50)
    volatility_range = np.linspace(0.05, 0.40, 50)

    fig = plotter.plot_sharpe_surface(
        returns_range, volatility_range,
        risk_free=0.02,
        save_path=str(output_dir / 'sharpe_surface.png')
    )
    plt.close(fig)

    print("  ✓ Saved: sharpe_surface.png")


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Quantitative Alpha: Generating All Visualizations")
    print("=" * 60)

    # Setup
    output_dir = ensure_output_dir()
    viz = QuantVisualization(figsize=(14, 10), style='dark')

    print(f"\nOutput directory: {output_dir}\n")

    # Generate all visualizations
    generate_lorenz_attractor(viz, output_dir)
    generate_iv_surface(viz, output_dir)
    generate_greeks_surfaces(viz, output_dir)
    generate_risk_return_time(viz, output_dir)
    generate_fokker_planck(viz, output_dir)
    generate_hmm_regimes(viz, output_dir)
    generate_kelly_surface(output_dir)
    generate_var_surface(output_dir)
    generate_phase_space(output_dir)
    generate_sharpe_surface(output_dir)

    print("\n" + "=" * 60)
    print("✓ All visualizations generated successfully!")
    print(f"  Location: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
