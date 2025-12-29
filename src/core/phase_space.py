"""
Phase Space Reconstruction using Takens' Embedding Theorem

Takens' theorem allows reconstruction of a system's phase space from a
single observable time series. This is essential for analyzing financial
markets where we only observe price but the underlying dynamics involve
multiple hidden state variables.

The embedding: v_t = [P(t), P(t-τ), P(t-2τ), ..., P(t-(m-1)τ)]
creates an m-dimensional representation that is topologically equivalent
to the true (unknown) attractor.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class EmbeddingParameters:
    """
    Parameters for time-delay embedding.

    Attributes:
        dimension: Embedding dimension (m)
        delay: Time delay (τ)
        method: Method used for parameter estimation
    """
    dimension: int
    delay: int
    method: str = "auto"


class TakensEmbedding:
    """
    Time-Delay Embedding for Phase Space Reconstruction.

    Implements Takens' theorem to reconstruct the phase space of a
    dynamical system from a single time series observation.

    The embedding creates vectors:
        v_t = [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]

    where τ is the time delay and m is the embedding dimension.

    Example:
        >>> embedder = TakensEmbedding()
        >>> params = embedder.estimate_parameters(price_series)
        >>> embedded = embedder.embed(price_series, params)
        >>> attractor = embedder.reconstruct_attractor(embedded)
    """

    def __init__(self, max_dimension: int = 10, max_delay: int = 100):
        """
        Initialize the embedding processor.

        Args:
            max_dimension: Maximum dimension to consider
            max_delay: Maximum delay to consider
        """
        self.max_dimension = max_dimension
        self.max_delay = max_delay

    def estimate_delay_mutual_information(
        self,
        series: np.ndarray,
        max_lag: int = None,
        n_bins: int = 20
    ) -> Tuple[int, np.ndarray]:
        """
        Estimate optimal time delay using mutual information.

        The first minimum of mutual information gives a good delay
        where coordinates are sufficiently independent but still
        capture the dynamics.

        Args:
            series: Input time series
            max_lag: Maximum lag to consider
            n_bins: Number of bins for histogram

        Returns:
            Optimal delay and mutual information curve
        """
        if max_lag is None:
            max_lag = min(self.max_delay, len(series) // 10)

        mi = np.zeros(max_lag)

        # Discretize series for MI calculation
        bins = np.linspace(series.min(), series.max(), n_bins + 1)
        digitized = np.digitize(series, bins) - 1
        digitized = np.clip(digitized, 0, n_bins - 1)

        for lag in range(max_lag):
            if lag == 0:
                # Self mutual information
                p = np.histogram(digitized, bins=n_bins, density=True)[0]
                mi[0] = -entropy(p + 1e-10)
            else:
                x = digitized[:-lag]
                y = digitized[lag:]

                # Joint distribution
                joint_hist, _, _ = np.histogram2d(x, y, bins=n_bins, density=True)

                # Marginal distributions
                px = np.sum(joint_hist, axis=1)
                py = np.sum(joint_hist, axis=0)

                # Mutual information
                joint_nonzero = joint_hist > 0
                mi[lag] = np.sum(
                    joint_hist[joint_nonzero] * np.log(
                        joint_hist[joint_nonzero] /
                        (px[:, None] * py[None, :])[joint_nonzero] + 1e-10
                    )
                )

        # Find first local minimum
        for i in range(1, len(mi) - 1):
            if mi[i] < mi[i-1] and mi[i] < mi[i+1]:
                return i, mi

        # Fallback: use first significant drop
        return np.argmin(mi[1:]) + 1, mi

    def estimate_delay_autocorrelation(
        self,
        series: np.ndarray,
        threshold: float = 1/np.e
    ) -> Tuple[int, np.ndarray]:
        """
        Estimate delay using autocorrelation decay.

        Returns the lag where autocorrelation first drops below 1/e
        (decorrelation time).

        Args:
            series: Input time series
            threshold: Correlation threshold (default 1/e)

        Returns:
            Optimal delay and autocorrelation function
        """
        n = len(series)
        max_lag = min(self.max_delay, n // 4)

        # Normalized autocorrelation
        series_centered = series - np.mean(series)
        autocorr = np.correlate(series_centered, series_centered, mode='full')
        autocorr = autocorr[n-1:n-1+max_lag] / autocorr[n-1]

        # Find first crossing of threshold
        crossings = np.where(autocorr < threshold)[0]
        delay = crossings[0] if len(crossings) > 0 else max_lag // 2

        return delay, autocorr

    def estimate_dimension_fnn(
        self,
        series: np.ndarray,
        delay: int,
        max_dim: int = None,
        threshold_ratio: float = 15.0,
        threshold_atol: float = 2.0
    ) -> Tuple[int, np.ndarray]:
        """
        Estimate embedding dimension using False Nearest Neighbors (FNN).

        A "false" neighbor is one that appears close only because the
        embedding dimension is too low. The optimal dimension is where
        the fraction of false neighbors approaches zero.

        Args:
            series: Input time series
            delay: Time delay τ
            max_dim: Maximum dimension to test
            threshold_ratio: Distance ratio threshold
            threshold_atol: Absolute distance threshold

        Returns:
            Optimal dimension and FNN fraction curve
        """
        if max_dim is None:
            max_dim = self.max_dimension

        n = len(series)
        fnn_fractions = np.zeros(max_dim)

        for dim in range(1, max_dim + 1):
            # Create embedding
            embedded = self.embed(series, EmbeddingParameters(dim, delay))
            n_vectors = len(embedded)

            if n_vectors < 10:
                break

            # Find nearest neighbors
            false_nn_count = 0
            total_count = 0

            # Sample for efficiency
            sample_size = min(500, n_vectors - delay - 1)
            indices = np.random.choice(n_vectors - delay - 1, sample_size, replace=False)

            for i in indices:
                # Distance to all other points
                distances = np.linalg.norm(embedded - embedded[i], axis=1)
                distances[i] = np.inf  # Exclude self

                # Nearest neighbor
                nn_idx = np.argmin(distances)
                nn_dist = distances[nn_idx]

                if nn_dist > 0 and dim < max_dim:
                    # Check if it's a false neighbor by looking at next dimension
                    idx_future = i + delay
                    nn_future = nn_idx + delay

                    if idx_future < n and nn_future < n:
                        future_dist = abs(series[idx_future] - series[nn_future])

                        # False neighbor criteria
                        ratio = future_dist / nn_dist
                        if ratio > threshold_ratio:
                            false_nn_count += 1

                total_count += 1

            fnn_fractions[dim-1] = false_nn_count / total_count if total_count > 0 else 0

            # Stop if FNN drops to near zero
            if fnn_fractions[dim-1] < 0.01:
                break

        # Find dimension where FNN drops below 10%
        optimal_dim = 1
        for dim in range(1, max_dim + 1):
            if fnn_fractions[dim-1] < 0.1:
                optimal_dim = dim
                break
        else:
            optimal_dim = max_dim

        return optimal_dim, fnn_fractions

    def estimate_parameters(
        self,
        series: np.ndarray,
        delay_method: str = 'mutual_info'
    ) -> EmbeddingParameters:
        """
        Automatically estimate optimal embedding parameters.

        Args:
            series: Input time series
            delay_method: 'mutual_info' or 'autocorr'

        Returns:
            EmbeddingParameters with optimal values
        """
        # Estimate delay
        if delay_method == 'mutual_info':
            delay, _ = self.estimate_delay_mutual_information(series)
        else:
            delay, _ = self.estimate_delay_autocorrelation(series)

        # Estimate dimension
        dimension, _ = self.estimate_dimension_fnn(series, delay)

        return EmbeddingParameters(
            dimension=dimension,
            delay=delay,
            method=f"{delay_method}_fnn"
        )

    def embed(
        self,
        series: np.ndarray,
        params: EmbeddingParameters
    ) -> np.ndarray:
        """
        Create time-delay embedding of the series.

        Args:
            series: Input time series
            params: Embedding parameters

        Returns:
            Embedded vectors (n_vectors, dimension)
        """
        n = len(series)
        m = params.dimension
        tau = params.delay

        n_vectors = n - (m - 1) * tau

        if n_vectors <= 0:
            raise ValueError(
                f"Series too short for embedding: n={n}, m={m}, tau={tau}"
            )

        embedded = np.zeros((n_vectors, m))
        for i in range(m):
            embedded[:, i] = series[i * tau:i * tau + n_vectors]

        return embedded

    def reconstruct_attractor(
        self,
        embedded: np.ndarray,
        n_points: int = None
    ) -> np.ndarray:
        """
        Extract attractor from embedded phase space.

        Removes transients and downsamples if necessary.

        Args:
            embedded: Embedded vectors
            n_points: Number of points to keep

        Returns:
            Attractor points
        """
        # Remove initial transient (first 10%)
        transient = len(embedded) // 10
        attractor = embedded[transient:]

        if n_points is not None and n_points < len(attractor):
            indices = np.linspace(0, len(attractor) - 1, n_points, dtype=int)
            attractor = attractor[indices]

        return attractor


class PhaseSpaceReconstructor:
    """
    Complete phase space reconstruction pipeline for financial data.

    Combines Takens embedding with noise reduction and attractor analysis
    to extract the deterministic structure from noisy market data.

    Example:
        >>> reconstructor = PhaseSpaceReconstructor()
        >>> result = reconstructor.reconstruct(price_series)
        >>> print(f"Attractor dimension: {result['dimension']:.2f}")
        >>> print(f"Lyapunov exponent: {result['lyapunov']:.4f}")
    """

    def __init__(self):
        """Initialize the reconstructor."""
        self.embedder = TakensEmbedding()
        self.params = None
        self.attractor = None

    def preprocess(
        self,
        series: np.ndarray,
        detrend: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess time series for embedding.

        Args:
            series: Raw price series
            detrend: Remove linear trend
            normalize: Normalize to zero mean, unit variance

        Returns:
            Preprocessed series
        """
        processed = series.copy().astype(float)

        # Log transform for prices
        if np.all(processed > 0):
            processed = np.log(processed)

        # Detrend
        if detrend:
            t = np.arange(len(processed))
            slope, intercept = np.polyfit(t, processed, 1)
            processed = processed - (slope * t + intercept)

        # Normalize
        if normalize:
            processed = (processed - np.mean(processed)) / (np.std(processed) + 1e-10)

        return processed

    def reconstruct(
        self,
        series: np.ndarray,
        preprocess: bool = True,
        return_embedding: bool = True
    ) -> dict:
        """
        Perform complete phase space reconstruction.

        Args:
            series: Input time series
            preprocess: Apply preprocessing
            return_embedding: Include embedded vectors in output

        Returns:
            Dictionary with reconstruction results:
                - params: Embedding parameters
                - attractor: Reconstructed attractor points
                - dimension: Estimated attractor dimension
                - lyapunov: Largest Lyapunov exponent estimate
        """
        if preprocess:
            processed = self.preprocess(series)
        else:
            processed = series

        # Estimate embedding parameters
        self.params = self.embedder.estimate_parameters(processed)

        # Create embedding
        embedded = self.embedder.embed(processed, self.params)

        # Extract attractor
        self.attractor = self.embedder.reconstruct_attractor(embedded)

        # Estimate attractor dimension
        dimension = self._estimate_dimension(self.attractor)

        # Estimate Lyapunov exponent
        lyapunov = self._estimate_lyapunov(self.attractor)

        result = {
            'params': self.params,
            'dimension': dimension,
            'lyapunov': lyapunov,
            'n_points': len(self.attractor)
        }

        if return_embedding:
            result['attractor'] = self.attractor
            result['embedded'] = embedded

        return result

    def _estimate_dimension(self, attractor: np.ndarray) -> float:
        """Estimate correlation dimension of the attractor."""
        n = len(attractor)
        if n < 100:
            return np.nan

        # Sample distances
        sample_size = min(500, n)
        indices = np.random.choice(n, sample_size, replace=False)
        sample = attractor[indices]

        distances = pdist(sample)
        distances = distances[distances > 0]

        if len(distances) < 10:
            return np.nan

        # Correlation sum at multiple scales
        r_values = np.percentile(distances, [5, 10, 20, 30, 40, 50])
        C = np.array([np.mean(distances < r) for r in r_values])

        # Linear fit in log-log space
        valid = (C > 0) & (C < 1)
        if np.sum(valid) < 2:
            return np.nan

        log_r = np.log(r_values[valid])
        log_C = np.log(C[valid])
        dimension = np.polyfit(log_r, log_C, 1)[0]

        return dimension

    def _estimate_lyapunov(
        self,
        attractor: np.ndarray,
        dt: float = 1.0
    ) -> float:
        """Estimate largest Lyapunov exponent."""
        n = len(attractor)
        if n < 100:
            return np.nan

        epsilon = np.std(attractor) * 0.01
        lyapunov_estimates = []

        # Sample trajectories
        for _ in range(min(100, n // 10)):
            i = np.random.randint(0, n - 50)

            # Find close neighbor
            distances = np.linalg.norm(attractor - attractor[i], axis=1)
            distances[max(0, i-5):min(n, i+5)] = np.inf  # Exclude temporal neighbors

            close_mask = (distances > epsilon/2) & (distances < epsilon)
            if not np.any(close_mask):
                continue

            j = np.random.choice(np.where(close_mask)[0])
            d0 = distances[j]

            # Track divergence
            for k in [10, 20, 30]:
                if i + k >= n or j + k >= n:
                    break
                dk = np.linalg.norm(attractor[i+k] - attractor[j+k])
                if d0 > 0 and dk > 0:
                    lyapunov_estimates.append(np.log(dk/d0) / (k * dt))

        return np.mean(lyapunov_estimates) if lyapunov_estimates else np.nan

    def project_3d(self, attractor: np.ndarray = None) -> np.ndarray:
        """
        Project attractor to 3D for visualization.

        Uses PCA if dimension > 3.

        Args:
            attractor: Attractor points (uses stored if None)

        Returns:
            3D projection of attractor
        """
        if attractor is None:
            attractor = self.attractor

        if attractor is None:
            raise ValueError("No attractor available")

        if attractor.shape[1] == 3:
            return attractor

        if attractor.shape[1] > 3:
            # PCA projection
            centered = attractor - np.mean(attractor, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            top_3 = eigenvectors[:, -3:]
            return centered @ top_3

        # Pad if less than 3D
        result = np.zeros((len(attractor), 3))
        result[:, :attractor.shape[1]] = attractor
        return result
