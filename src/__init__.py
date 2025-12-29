"""
Quantitative Alpha: A Unified Framework for Deterministic Chaos,
Stochastic Physics, and Advanced Signal Processing

This package provides a comprehensive quantitative trading framework
that bridges theoretical physics and practical algorithmic trading.
"""

__version__ = "1.0.0"
__author__ = "Quantitative Alpha Research Team"

from .core import LorenzSystem, PhaseSpaceReconstructor
from .models import LangevinDynamics, FokkerPlanck, HiddenMarkovRegime
from .visualization import QuantVisualization
from .portfolio import HierarchicalRiskParity, KellyCriterion
