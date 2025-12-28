"""
Hidden Markov Models for Market Regime Detection

Hidden Markov Models (HMMs) are a core component of quantitative trading
strategies, notably used by Renaissance Technologies. The key insight is
that markets are driven by hidden (unobservable) states/regimes that
manifest through observable signals (prices, volume, spreads).

Components:
    - Hidden States: Market regimes (Bull, Bear, Sideways, High/Low Vol)
    - Observations: Returns, volume, spreads, indicators
    - Transition Matrix: Probabilities of regime changes
    - Emission Distributions: How each regime generates observations

Key Algorithms:
    - Forward-Backward: Calculate state probabilities
    - Baum-Welch: Train model parameters (EM algorithm)
    - Viterbi: Decode most likely regime sequence in real-time
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_LOW_VOL = 0
    BULL_HIGH_VOL = 1
    BEAR_LOW_VOL = 2
    BEAR_HIGH_VOL = 3
    SIDEWAYS = 4


@dataclass
class HMMParameters:
    """
    Hidden Markov Model parameters.

    Attributes:
        n_states: Number of hidden states
        transition_matrix: State transition probabilities A[i,j] = P(S_t=j|S_{t-1}=i)
        initial_probs: Initial state distribution π
        emission_means: Mean of Gaussian emissions per state
        emission_covs: Covariance of emissions per state
    """
    n_states: int
    transition_matrix: np.ndarray
    initial_probs: np.ndarray
    emission_means: np.ndarray
    emission_covs: np.ndarray


class HiddenMarkovRegime:
    """
    Hidden Markov Model for Market Regime Detection.

    Identifies latent market regimes from observable data (returns, volume)
    using Gaussian emission distributions.

    Model Structure:
        P(O_1, ..., O_T, S_1, ..., S_T) = π_{S_1} · ∏ A_{S_t, S_{t+1}} · ∏ B_{S_t}(O_t)

    where:
        π: Initial state distribution
        A: Transition matrix
        B: Emission probabilities

    Example:
        >>> hmm = HiddenMarkovRegime(n_states=3)
        >>> hmm.fit(training_returns)
        >>> current_regime = hmm.decode_current(recent_returns)
        >>> if current_regime == 0:
        ...     print("Bull regime - go long")
    """

    def __init__(
        self,
        n_states: int = 3,
        n_features: int = 1,
        covariance_type: str = 'full'
    ):
        """
        Initialize the HMM.

        Args:
            n_states: Number of hidden states (regimes)
            n_features: Dimension of observation vectors
            covariance_type: 'full', 'diag', or 'spherical'
        """
        self.n_states = n_states
        self.n_features = n_features
        self.covariance_type = covariance_type

        # Initialize parameters
        self._init_params()

    def _init_params(self):
        """Initialize model parameters with reasonable defaults."""
        n = self.n_states
        d = self.n_features

        # Transition matrix: slight persistence (diagonal dominant)
        self.A = np.ones((n, n)) * 0.1 / (n - 1)
        np.fill_diagonal(self.A, 0.9)

        # Initial probabilities: uniform
        self.pi = np.ones(n) / n

        # Emission means: spread across typical return range
        self.means = np.linspace(-0.02, 0.02, n).reshape(n, d)

        # Emission covariances
        if self.covariance_type == 'full':
            self.covs = np.array([np.eye(d) * 0.01 for _ in range(n)])
        elif self.covariance_type == 'diag':
            self.covs = np.ones((n, d)) * 0.01
        else:  # spherical
            self.covs = np.ones(n) * 0.01

    def emission_prob(
        self,
        obs: np.ndarray,
        state: int
    ) -> float:
        """
        Calculate emission probability P(O|S=state).

        Args:
            obs: Observation vector
            state: Hidden state index

        Returns:
            Emission probability (density)
        """
        if self.covariance_type == 'full':
            return multivariate_normal.pdf(obs, self.means[state], self.covs[state])
        else:
            return norm.pdf(obs, self.means[state], np.sqrt(self.covs[state])).prod()

    def emission_log_prob(
        self,
        obs: np.ndarray,
        state: int
    ) -> float:
        """Calculate log emission probability."""
        if self.covariance_type == 'full':
            return multivariate_normal.logpdf(obs, self.means[state], self.covs[state])
        else:
            return norm.logpdf(obs, self.means[state], np.sqrt(self.covs[state])).sum()

    def forward(
        self,
        observations: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm: compute α_t(i) = P(O_1,...,O_t, S_t=i).

        Uses log-space for numerical stability.

        Args:
            observations: Sequence of observations (T, n_features)

        Returns:
            (log_alpha, log_likelihood) tuple
        """
        T = len(observations)
        n = self.n_states

        log_alpha = np.zeros((T, n))

        # Initialization
        for j in range(n):
            log_alpha[0, j] = (
                np.log(self.pi[j] + 1e-300) +
                self.emission_log_prob(observations[0], j)
            )

        # Recursion
        for t in range(1, T):
            for j in range(n):
                log_alpha[t, j] = (
                    logsumexp(log_alpha[t-1] + np.log(self.A[:, j] + 1e-300)) +
                    self.emission_log_prob(observations[t], j)
                )

        log_likelihood = logsumexp(log_alpha[-1])
        return log_alpha, log_likelihood

    def backward(
        self,
        observations: np.ndarray
    ) -> np.ndarray:
        """
        Backward algorithm: compute β_t(i) = P(O_{t+1},...,O_T | S_t=i).

        Args:
            observations: Sequence of observations

        Returns:
            log_beta array
        """
        T = len(observations)
        n = self.n_states

        log_beta = np.zeros((T, n))

        # Initialization: log(1) = 0
        log_beta[-1] = 0

        # Recursion (backwards)
        for t in range(T - 2, -1, -1):
            for i in range(n):
                terms = np.zeros(n)
                for j in range(n):
                    terms[j] = (
                        np.log(self.A[i, j] + 1e-300) +
                        self.emission_log_prob(observations[t+1], j) +
                        log_beta[t+1, j]
                    )
                log_beta[t, i] = logsumexp(terms)

        return log_beta

    def compute_posteriors(
        self,
        observations: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior probabilities γ_t(i) and ξ_t(i,j).

        γ_t(i) = P(S_t=i | O)
        ξ_t(i,j) = P(S_t=i, S_{t+1}=j | O)

        Args:
            observations: Observation sequence

        Returns:
            (gamma, xi) tuple
        """
        T = len(observations)
        n = self.n_states

        log_alpha, log_likelihood = self.forward(observations)
        log_beta = self.backward(observations)

        # Gamma: P(S_t = i | O)
        log_gamma = log_alpha + log_beta - log_likelihood
        gamma = np.exp(log_gamma)

        # Xi: P(S_t = i, S_{t+1} = j | O)
        xi = np.zeros((T - 1, n, n))

        for t in range(T - 1):
            for i in range(n):
                for j in range(n):
                    log_xi = (
                        log_alpha[t, i] +
                        np.log(self.A[i, j] + 1e-300) +
                        self.emission_log_prob(observations[t+1], j) +
                        log_beta[t+1, j] -
                        log_likelihood
                    )
                    xi[t, i, j] = np.exp(log_xi)

        return gamma, xi

    def fit(
        self,
        observations: np.ndarray,
        n_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = False
    ) -> Dict[str, List[float]]:
        """
        Fit HMM using Baum-Welch algorithm (Expectation-Maximization).

        Args:
            observations: Training observations (T, n_features)
            n_iterations: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress

        Returns:
            Dictionary with training history
        """
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        history = {'log_likelihood': []}
        prev_ll = float('-inf')

        for iteration in range(n_iterations):
            # E-step: compute posteriors
            gamma, xi = self.compute_posteriors(observations)
            _, log_likelihood = self.forward(observations)

            history['log_likelihood'].append(log_likelihood)

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: LL = {log_likelihood:.4f}")

            # Check convergence
            if abs(log_likelihood - prev_ll) < tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            prev_ll = log_likelihood

            # M-step: update parameters
            self._m_step(observations, gamma, xi)

        return history

    def _m_step(
        self,
        observations: np.ndarray,
        gamma: np.ndarray,
        xi: np.ndarray
    ):
        """M-step: update parameters given posteriors."""
        T = len(observations)
        n = self.n_states

        # Update initial probabilities
        self.pi = gamma[0] / np.sum(gamma[0])

        # Update transition matrix
        for i in range(n):
            denom = np.sum(gamma[:-1, i])
            for j in range(n):
                self.A[i, j] = np.sum(xi[:, i, j]) / (denom + 1e-300)

        # Normalize rows
        self.A = self.A / (self.A.sum(axis=1, keepdims=True) + 1e-300)

        # Update emission parameters
        for j in range(n):
            # Weighted mean
            weights = gamma[:, j]
            weight_sum = np.sum(weights)

            self.means[j] = np.sum(weights[:, None] * observations, axis=0) / (weight_sum + 1e-300)

            # Weighted covariance
            diff = observations - self.means[j]
            if self.covariance_type == 'full':
                self.covs[j] = (
                    np.sum(weights[:, None, None] * (diff[:, :, None] @ diff[:, None, :]), axis=0)
                    / (weight_sum + 1e-300)
                )
                # Regularization
                self.covs[j] += np.eye(self.n_features) * 1e-6
            else:
                self.covs[j] = np.sum(weights[:, None] * diff**2, axis=0) / (weight_sum + 1e-300) + 1e-6


class ViterbiDecoder:
    """
    Viterbi Algorithm for Real-Time Regime Decoding.

    Finds the most likely sequence of hidden states:
        argmax_{S} P(S | O) = argmax_{S} P(O | S) P(S)

    This is critical for live trading:
        - Detects regime changes BEFORE price moves significantly
        - Enables proactive position adjustment

    Example:
        >>> decoder = ViterbiDecoder(trained_hmm)
        >>> regime_sequence = decoder.decode(recent_observations)
        >>> current_regime = regime_sequence[-1]
    """

    def __init__(self, hmm: HiddenMarkovRegime):
        """
        Initialize decoder with trained HMM.

        Args:
            hmm: Trained HiddenMarkovRegime model
        """
        self.hmm = hmm

    def decode(
        self,
        observations: np.ndarray
    ) -> np.ndarray:
        """
        Decode the most likely state sequence.

        Args:
            observations: Observation sequence

        Returns:
            Most likely state sequence
        """
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        T = len(observations)
        n = self.hmm.n_states

        # Viterbi variables
        log_delta = np.zeros((T, n))
        psi = np.zeros((T, n), dtype=int)

        # Initialization
        for j in range(n):
            log_delta[0, j] = (
                np.log(self.hmm.pi[j] + 1e-300) +
                self.hmm.emission_log_prob(observations[0], j)
            )

        # Recursion
        for t in range(1, T):
            for j in range(n):
                candidates = log_delta[t-1] + np.log(self.hmm.A[:, j] + 1e-300)
                psi[t, j] = np.argmax(candidates)
                log_delta[t, j] = (
                    candidates[psi[t, j]] +
                    self.hmm.emission_log_prob(observations[t], j)
                )

        # Backtracking
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(log_delta[-1])

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def decode_online(
        self,
        observations: np.ndarray,
        prior_log_delta: np.ndarray = None
    ) -> Tuple[int, np.ndarray, float]:
        """
        Online Viterbi decoding for streaming data.

        Args:
            observations: New observations to process
            prior_log_delta: Log-delta from previous step

        Returns:
            (current_state, new_log_delta, confidence) tuple
        """
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        n = self.hmm.n_states

        if prior_log_delta is None:
            # Initialize
            log_delta = np.zeros(n)
            for j in range(n):
                log_delta[j] = (
                    np.log(self.hmm.pi[j] + 1e-300) +
                    self.hmm.emission_log_prob(observations[0], j)
                )
            obs_start = 1
        else:
            log_delta = prior_log_delta.copy()
            obs_start = 0

        # Process new observations
        for t in range(obs_start, len(observations)):
            new_log_delta = np.zeros(n)
            for j in range(n):
                candidates = log_delta + np.log(self.hmm.A[:, j] + 1e-300)
                new_log_delta[j] = (
                    np.max(candidates) +
                    self.hmm.emission_log_prob(observations[t], j)
                )
            log_delta = new_log_delta

        # Current best state
        current_state = np.argmax(log_delta)

        # Confidence: probability gap between best and second-best
        sorted_delta = np.sort(log_delta)[::-1]
        confidence = 1 - np.exp(sorted_delta[1] - sorted_delta[0])

        return current_state, log_delta, confidence


class BaumWelchTrainer:
    """
    Advanced Baum-Welch Trainer with Financial Enhancements.

    Provides additional training features for financial applications:
        - Multiple sequence training
        - Regime persistence constraints
        - Information criteria for model selection

    Example:
        >>> trainer = BaumWelchTrainer(n_states=4)
        >>> best_model = trainer.train_with_selection(
        ...     data, n_states_range=[2, 3, 4, 5]
        ... )
    """

    def __init__(
        self,
        n_states: int = 3,
        n_features: int = 1,
        min_persistence: float = 0.8
    ):
        """
        Initialize trainer.

        Args:
            n_states: Number of hidden states
            n_features: Observation dimension
            min_persistence: Minimum self-transition probability
        """
        self.n_states = n_states
        self.n_features = n_features
        self.min_persistence = min_persistence

    def train_multiple_sequences(
        self,
        sequences: List[np.ndarray],
        n_iterations: int = 100
    ) -> HiddenMarkovRegime:
        """
        Train HMM on multiple observation sequences.

        Args:
            sequences: List of observation sequences
            n_iterations: Maximum iterations

        Returns:
            Trained HMM
        """
        hmm = HiddenMarkovRegime(self.n_states, self.n_features)

        prev_ll = float('-inf')

        for iteration in range(n_iterations):
            total_gamma = []
            total_xi = []
            total_ll = 0

            # E-step across all sequences
            for obs in sequences:
                if obs.ndim == 1:
                    obs = obs.reshape(-1, 1)
                gamma, xi = hmm.compute_posteriors(obs)
                _, ll = hmm.forward(obs)

                total_gamma.append(gamma)
                total_xi.append(xi)
                total_ll += ll

            # Check convergence
            if abs(total_ll - prev_ll) < 1e-6:
                break
            prev_ll = total_ll

            # M-step: aggregate statistics
            self._m_step_aggregated(hmm, sequences, total_gamma, total_xi)

            # Enforce minimum persistence
            self._enforce_persistence(hmm)

        return hmm

    def _m_step_aggregated(
        self,
        hmm: HiddenMarkovRegime,
        sequences: List[np.ndarray],
        gammas: List[np.ndarray],
        xis: List[np.ndarray]
    ):
        """Aggregated M-step across multiple sequences."""
        n = hmm.n_states

        # Initial probabilities: average of first timestep gammas
        hmm.pi = np.mean([g[0] for g in gammas], axis=0)
        hmm.pi = hmm.pi / np.sum(hmm.pi)

        # Transition matrix
        for i in range(n):
            for j in range(n):
                num = sum(np.sum(xi[:, i, j]) for xi in xis)
                denom = sum(np.sum(gamma[:-1, i]) for gamma in gammas)
                hmm.A[i, j] = num / (denom + 1e-300)

        hmm.A = hmm.A / (hmm.A.sum(axis=1, keepdims=True) + 1e-300)

        # Emission parameters
        for j in range(n):
            all_weighted_obs = []
            all_weights = []

            for obs, gamma in zip(sequences, gammas):
                if obs.ndim == 1:
                    obs = obs.reshape(-1, 1)
                all_weighted_obs.append(gamma[:, j:j+1] * obs)
                all_weights.append(gamma[:, j])

            total_weights = np.concatenate(all_weights)
            total_weighted_obs = np.vstack(all_weighted_obs)

            hmm.means[j] = np.sum(total_weighted_obs, axis=0) / (np.sum(total_weights) + 1e-300)

    def _enforce_persistence(self, hmm: HiddenMarkovRegime):
        """Enforce minimum regime persistence."""
        for i in range(hmm.n_states):
            if hmm.A[i, i] < self.min_persistence:
                deficit = self.min_persistence - hmm.A[i, i]
                hmm.A[i, i] = self.min_persistence

                # Redistribute deficit from off-diagonal
                off_diag_sum = np.sum(hmm.A[i, :]) - hmm.A[i, i]
                if off_diag_sum > 0:
                    for j in range(hmm.n_states):
                        if j != i:
                            hmm.A[i, j] *= (1 - self.min_persistence) / off_diag_sum

        # Renormalize
        hmm.A = hmm.A / (hmm.A.sum(axis=1, keepdims=True) + 1e-300)

    def compute_bic(
        self,
        hmm: HiddenMarkovRegime,
        observations: np.ndarray
    ) -> float:
        """
        Compute Bayesian Information Criterion for model selection.

        BIC = -2 * log(L) + k * log(n)

        Lower is better.
        """
        _, log_likelihood = hmm.forward(observations)
        n_samples = len(observations)

        # Number of parameters
        n_states = hmm.n_states
        n_features = hmm.n_features

        # Transition: n_states * (n_states - 1) free parameters
        # Initial: n_states - 1 free parameters
        # Means: n_states * n_features
        # Covariances: depends on type
        if hmm.covariance_type == 'full':
            cov_params = n_states * n_features * (n_features + 1) / 2
        else:
            cov_params = n_states * n_features

        n_params = (
            n_states * (n_states - 1) +
            (n_states - 1) +
            n_states * n_features +
            cov_params
        )

        bic = -2 * log_likelihood + n_params * np.log(n_samples)
        return bic

    def select_best_n_states(
        self,
        observations: np.ndarray,
        n_states_range: List[int] = [2, 3, 4, 5],
        n_iterations: int = 100
    ) -> Tuple[HiddenMarkovRegime, Dict]:
        """
        Select optimal number of states using BIC.

        Args:
            observations: Training data
            n_states_range: List of n_states to try
            n_iterations: Training iterations per model

        Returns:
            (best_model, comparison_results) tuple
        """
        results = {}
        best_bic = float('inf')
        best_model = None

        for n_states in n_states_range:
            hmm = HiddenMarkovRegime(n_states, self.n_features)
            hmm.fit(observations, n_iterations=n_iterations)

            bic = self.compute_bic(hmm, observations)
            _, ll = hmm.forward(observations)

            results[n_states] = {
                'bic': bic,
                'log_likelihood': ll,
                'model': hmm
            }

            if bic < best_bic:
                best_bic = bic
                best_model = hmm

        return best_model, results
