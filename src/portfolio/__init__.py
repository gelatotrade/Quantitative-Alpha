"""
Portfolio Optimization and Risk Management.

Modern portfolio optimization beyond Markowitz:
- Hierarchical Risk Parity (HRP)
- Kelly Criterion for position sizing
- Risk budgeting and tail risk management
"""

from .hrp import HierarchicalRiskParity
from .kelly import KellyCriterion, FractionalKelly
from .risk import RiskManager, TailRiskAnalyzer
