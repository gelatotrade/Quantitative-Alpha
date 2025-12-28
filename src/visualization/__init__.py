"""
Visualization Module for Quantitative Alpha.

Physics-inspired 3D visualizations for financial modeling:
- Lorenz attractors and phase space plots
- Implied volatility surfaces
- Greeks surfaces (Delta, Gamma, Vega, Vanna, Volga)
- Risk-Return-Time 3D coordinate systems
- Probability density evolution
- Regime transition diagrams
"""

from .quant_viz import QuantVisualization
from .surfaces import SurfacePlotter
from .phase_space_viz import PhaseSpaceVisualization
