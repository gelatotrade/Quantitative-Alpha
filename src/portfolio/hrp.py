"""
Hierarchical Risk Parity (HRP) Portfolio Optimization

Developed by Marcos Lopez de Prado, HRP uses machine learning (graph theory)
to solve the diversification problem in a more robust way than traditional
Mean-Variance Optimization (MVO).

Key Advantages over MVO:
    1. No covariance matrix inversion (avoids instability)
    2. Robust to estimation error in correlations
    3. Naturally handles multicollinearity
    4. More stable out-of-sample performance

The Algorithm:
    1. Hierarchical Clustering: Build tree structure from correlations
    2. Quasi-Diagonalization: Reorder assets so similar ones are adjacent
    3. Recursive Bisection: Allocate capital top-down based on variance

Reference:
    Lopez de Prado, M. (2016). "Building Diversified Portfolios that
    Outperform Out-of-Sample." Journal of Portfolio Management.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class HRPResult:
    """
    Results from HRP optimization.

    Attributes:
        weights: Optimal portfolio weights
        asset_order: Assets ordered by clustering
        dendrogram: Clustering linkage matrix
        cluster_variances: Variance of each cluster
    """
    weights: np.ndarray
    asset_order: np.ndarray
    dendrogram: np.ndarray
    cluster_variances: Dict[str, float]


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity Portfolio Optimizer.

    Implements the full HRP algorithm with extensions:
        - Multiple linkage methods
        - Variance-based and equal-weight options
        - Risk budgeting constraints

    Example:
        >>> returns = pd.DataFrame(...)  # Historical returns
        >>> hrp = HierarchicalRiskParity()
        >>> result = hrp.optimize(returns)
        >>> print(f"Weights: {result.weights}")
    """

    def __init__(
        self,
        linkage_method: str = 'single',
        distance_metric: str = 'correlation'
    ):
        """
        Initialize HRP optimizer.

        Args:
            linkage_method: 'single', 'complete', 'average', 'ward'
            distance_metric: 'correlation' or 'euclidean'
        """
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric

    def _correlation_distance(self, corr_matrix: np.ndarray) -> np.ndarray:
        """
        Convert correlation matrix to distance matrix.

        Distance = sqrt(2 * (1 - correlation))

        This ensures:
            - Perfect correlation (ρ=1) → distance = 0
            - Zero correlation (ρ=0) → distance = √2
            - Perfect anti-correlation (ρ=-1) → distance = 2
        """
        return np.sqrt(2 * (1 - corr_matrix))

    def _get_quasi_diagonal(
        self,
        link: np.ndarray,
        n_assets: int
    ) -> np.ndarray:
        """
        Get quasi-diagonalization order from linkage matrix.

        Reorders assets so that correlated assets are adjacent.

        Args:
            link: Linkage matrix from hierarchical clustering
            n_assets: Number of assets

        Returns:
            Reordered indices
        """
        return leaves_list(link)

    def _get_cluster_variance(
        self,
        cov_matrix: np.ndarray,
        cluster_indices: np.ndarray
    ) -> float:
        """
        Calculate variance of a cluster using inverse-variance weighting.

        Args:
            cov_matrix: Full covariance matrix
            cluster_indices: Indices of assets in cluster

        Returns:
            Cluster variance
        """
        # Sub-covariance matrix for cluster
        cluster_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]

        # Inverse variance weights within cluster
        inv_var = 1 / np.diag(cluster_cov)
        weights = inv_var / np.sum(inv_var)

        # Cluster variance
        cluster_var = np.dot(weights, np.dot(cluster_cov, weights))

        return cluster_var

    def _recursive_bisection(
        self,
        cov_matrix: np.ndarray,
        sorted_indices: np.ndarray
    ) -> np.ndarray:
        """
        Recursive bisection to allocate weights.

        At each split, allocate inversely proportional to cluster variance.

        Args:
            cov_matrix: Covariance matrix
            sorted_indices: Quasi-diagonalized order

        Returns:
            Portfolio weights
        """
        weights = np.ones(len(sorted_indices))
        cluster_items = [sorted_indices.tolist()]

        while len(cluster_items) > 0:
            # Split each cluster
            new_clusters = []

            for cluster in cluster_items:
                if len(cluster) > 1:
                    # Split in half
                    mid = len(cluster) // 2
                    left = cluster[:mid]
                    right = cluster[mid:]

                    # Calculate cluster variances
                    left_var = self._get_cluster_variance(cov_matrix, np.array(left))
                    right_var = self._get_cluster_variance(cov_matrix, np.array(right))

                    # Allocate inversely proportional to variance
                    alpha = 1 - left_var / (left_var + right_var)

                    # Update weights
                    weights[left] *= alpha
                    weights[right] *= (1 - alpha)

                    # Add subclusters for next iteration
                    new_clusters.append(left)
                    new_clusters.append(right)

            cluster_items = [c for c in new_clusters if len(c) > 1]

        return weights

    def optimize(
        self,
        returns: np.ndarray,
        cov_matrix: np.ndarray = None,
        corr_matrix: np.ndarray = None
    ) -> HRPResult:
        """
        Perform HRP optimization.

        Args:
            returns: Returns matrix (n_samples, n_assets)
            cov_matrix: Optional precomputed covariance matrix
            corr_matrix: Optional precomputed correlation matrix

        Returns:
            HRPResult with optimal weights and diagnostics
        """
        n_assets = returns.shape[1]

        # Compute covariance and correlation if not provided
        if cov_matrix is None:
            cov_matrix = np.cov(returns.T)
        if corr_matrix is None:
            std = np.sqrt(np.diag(cov_matrix))
            corr_matrix = cov_matrix / np.outer(std, std)

        # Step 1: Hierarchical Clustering
        dist_matrix = self._correlation_distance(corr_matrix)
        np.fill_diagonal(dist_matrix, 0)  # Ensure diagonal is zero

        # Convert to condensed form for scipy
        dist_condensed = squareform(dist_matrix)

        # Hierarchical clustering
        link = linkage(dist_condensed, method=self.linkage_method)

        # Step 2: Quasi-diagonalization
        sorted_indices = self._get_quasi_diagonal(link, n_assets)

        # Step 3: Recursive bisection
        weights = self._recursive_bisection(cov_matrix, sorted_indices)

        # Compute cluster variances for diagnostics
        cluster_vars = {}
        half = len(sorted_indices) // 2
        cluster_vars['left'] = self._get_cluster_variance(
            cov_matrix, sorted_indices[:half]
        )
        cluster_vars['right'] = self._get_cluster_variance(
            cov_matrix, sorted_indices[half:]
        )

        return HRPResult(
            weights=weights,
            asset_order=sorted_indices,
            dendrogram=link,
            cluster_variances=cluster_vars
        )

    def optimize_with_constraints(
        self,
        returns: np.ndarray,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        sector_constraints: Dict[str, Tuple[List[int], float]] = None
    ) -> np.ndarray:
        """
        HRP with weight constraints.

        Args:
            returns: Returns matrix
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
            sector_constraints: Dict mapping sector name to (indices, max_weight)

        Returns:
            Constrained portfolio weights
        """
        # Get unconstrained HRP weights
        result = self.optimize(returns)
        weights = result.weights.copy()

        # Apply individual constraints
        weights = np.clip(weights, min_weight, max_weight)

        # Apply sector constraints
        if sector_constraints:
            for sector, (indices, max_sector) in sector_constraints.items():
                sector_weight = np.sum(weights[indices])
                if sector_weight > max_sector:
                    # Scale down sector proportionally
                    scale = max_sector / sector_weight
                    weights[indices] *= scale

        # Renormalize to sum to 1
        weights = weights / np.sum(weights)

        return weights

    def rolling_optimize(
        self,
        returns: np.ndarray,
        window: int = 252,
        rebalance_frequency: int = 21
    ) -> np.ndarray:
        """
        Rolling HRP optimization.

        Args:
            returns: Full returns matrix
            window: Lookback window for optimization
            rebalance_frequency: Days between rebalancing

        Returns:
            Time series of weights (n_periods, n_assets)
        """
        n_samples, n_assets = returns.shape
        n_periods = (n_samples - window) // rebalance_frequency + 1

        weights_history = np.zeros((n_periods, n_assets))

        for i in range(n_periods):
            start = i * rebalance_frequency
            end = start + window

            if end > n_samples:
                break

            window_returns = returns[start:end]
            result = self.optimize(window_returns)
            weights_history[i] = result.weights

        return weights_history

    def risk_contribution(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate marginal risk contribution of each asset.

        Risk contribution = w_i * (Σw)_i / σ_p

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix

        Returns:
            Risk contribution per asset (sums to 1)
        """
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_var)

        marginal_risk = np.dot(cov_matrix, weights)
        risk_contribution = weights * marginal_risk / portfolio_std

        # Normalize to percentages
        return risk_contribution / np.sum(risk_contribution)


class RiskBudgetOptimizer:
    """
    Risk Budget Optimization: Allocate risk, not capital.

    Instead of specifying target weights, specify target risk contributions.
    The optimizer finds weights that achieve the desired risk allocation.

    Equal Risk Contribution (ERC) = Risk Parity: Each asset contributes
    equally to portfolio risk.

    Example:
        >>> rbo = RiskBudgetOptimizer()
        >>> weights = rbo.equal_risk_contribution(cov_matrix)
    """

    def __init__(self):
        """Initialize risk budget optimizer."""
        pass

    def equal_risk_contribution(
        self,
        cov_matrix: np.ndarray,
        initial_weights: np.ndarray = None,
        tolerance: float = 1e-8,
        max_iterations: int = 1000
    ) -> np.ndarray:
        """
        Find Equal Risk Contribution (ERC) portfolio.

        Uses iterative reweighting to find weights where each asset
        has equal marginal contribution to risk.

        Args:
            cov_matrix: Covariance matrix
            initial_weights: Starting weights (default: equal)
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations

        Returns:
            ERC portfolio weights
        """
        n = cov_matrix.shape[0]

        if initial_weights is None:
            weights = np.ones(n) / n
        else:
            weights = initial_weights.copy()

        for iteration in range(max_iterations):
            # Calculate marginal risk contribution
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
            marginal_risk = np.dot(cov_matrix, weights)

            # Target: equal contribution
            target_contrib = portfolio_var / n

            # Update weights
            new_weights = np.sqrt(target_contrib / (marginal_risk + 1e-10))
            new_weights = new_weights / np.sum(new_weights)

            # Check convergence
            if np.max(np.abs(new_weights - weights)) < tolerance:
                break

            weights = new_weights

        return weights

    def target_risk_budget(
        self,
        cov_matrix: np.ndarray,
        risk_budget: np.ndarray,
        tolerance: float = 1e-8,
        max_iterations: int = 1000
    ) -> np.ndarray:
        """
        Find portfolio with specified risk budget.

        Args:
            cov_matrix: Covariance matrix
            risk_budget: Target risk contribution per asset (sums to 1)
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations

        Returns:
            Portfolio weights achieving target risk budget
        """
        n = cov_matrix.shape[0]
        risk_budget = np.array(risk_budget)
        risk_budget = risk_budget / np.sum(risk_budget)  # Normalize

        weights = risk_budget.copy()  # Initial guess

        for iteration in range(max_iterations):
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
            marginal_risk = np.dot(cov_matrix, weights)

            # Scale weights to match risk budget
            new_weights = np.sqrt(risk_budget * portfolio_var / (marginal_risk + 1e-10))
            new_weights = new_weights / np.sum(new_weights)

            if np.max(np.abs(new_weights - weights)) < tolerance:
                break

            weights = new_weights

        return weights
