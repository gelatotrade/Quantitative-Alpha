"""
Stochastic and Statistical Models for Quantitative Trading.

This module provides:
- Stochastic Differential Equations (Langevin dynamics)
- Fokker-Planck probability evolution
- Hidden Markov Models for regime detection
- Black-Scholes Greeks and IV surfaces
"""

from .stochastic import LangevinDynamics, FokkerPlanck, SDESimulator
from .hmm import HiddenMarkovRegime, BaumWelchTrainer, ViterbiDecoder
from .options import BlackScholes, GreeksSurface, ImpliedVolatilitySurface
from .path_integral import FeynmanPathIntegral, QuantumFinance
