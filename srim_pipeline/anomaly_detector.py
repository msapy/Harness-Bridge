"""
anomaly_detector.py
===================
AnomalyDetector class: unsupervised Gaussian distribution-based anomaly
detection following Tran & Ozer (2020), S.2.3 (Eqs. 28-31).

Mathematical background
-----------------------
Given a baseline (reference) dataset of m identified modal features
{x^(1), x^(2), ..., x^(n)} from the earliest (healthy) time segments:

(i)  Fit a Gaussian to each feature j:

         p(x_j; mu_j, sigma^2_j) = 1/sqrt(2pisigma^2_j) * exp(-(x_j - mu_j)^2 / 2sigma^2_j)

     with:   mu_j = (1/n) Σ x_j^(i)          (Eq. 29)
             sigma^2_j = (1/n) Σ (x_j^(i) - mu_j)^2 (Eq. 30)

(ii) For a new observation x̂, compute the multivariate probability:

         p(x̂) = ∏_j p(x̂_j; mu_j, sigma^2_j)      (Eq. 31)

(iii) Flag as anomaly if p(x̂) < ε.

The threshold ε is expressed as a k-sigma level:
    ε_k = p(mu_j - k*sigma_j ; mu_j, sigma^2_j)  for integer k = 1, 2, 3, ...
which gives ε_k = 1/sqrt(2pisigma^2_j) * exp(-k^2/2).

Boolean operators combine univariate anomaly flags:
    AND  - flag global anomaly only if ALL selected features are anomalous.
    OR   - flag if ANY feature is anomalous.

Structural context (Harness Bridge)
------------------------------------
Days are labelled by structural state relative to concrete pours:

    State 0 (PRE_LOWER_POUR)  - Day 1:  baseline, no concrete
    State 1 (POST_LOWER_POUR) - Day 2+: lower deck slab poured (mass increase)
    State 2 (UPPER_ACTIVITY)  - Day 8:  upper deck pour (complex transition)
    State 3 (POST_UPPER_POUR) - Day 9+: both slabs in place (max mass)

We expect concrete pours to DECREASE natural frequencies (added mass) and
potentially change damping ratios -- these are the anomaly signatures.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Literal


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FeatureTimeSeries = Dict[str, List[Optional[float]]]

# Operator choices
Operator = Literal["AND", "OR", "FREQ_ONLY", "DAMP_ONLY"]


class AnomalyDetector:
    """
    Gaussian anomaly detector for modal parameters over time.

    Parameters
    ----------
    n_baseline_segments : int
        Number of segments from the FIRST day (Day 1) used to fit the
        baseline Gaussian.  Segments with missing modal parameters are skipped
        and do not count towards this total.
    k_sigma_values : tuple of int
        k-sigma threshold levels to evaluate simultaneously (default (1, 2, 3)).
        Larger k -> looser threshold -> fewer false positives, more false negatives.
    mode_indices : tuple of int
        Which identified cluster indices to track (0-based; 0 = lowest freq mode).
        Default (0, 1) tracks first and second modes.
    features : tuple of str
        Which features to track per mode. Choices: 'freq', 'damping'.
        Default ('freq', 'damping') -- both frequency and damping.
    min_baseline_count : int
        Minimum baseline observations required to fit a reliable Gaussian.
        If fewer are available, detection is skipped for that feature.
    """

    def __init__(
        self,
        n_baseline_segments: int = 30,
        k_sigma_values: Tuple[int, ...] = (1, 2, 3),
        mode_indices: Tuple[int, ...] = (0, 1),
        features: Tuple[str, ...] = ("freq", "damping"),
        min_baseline_count: int = 5,
    ):
        self.n_baseline_segments = n_baseline_segments
        self.k_sigma_values = k_sigma_values
        self.mode_indices = mode_indices
        self.features = features
        self.min_baseline_count = min_baseline_count

        # Baseline statistics: feature_name -> (mu, sigma)
        self._baseline_stats: Dict[str, Tuple[float, float]] = {}

        # Pre-computed thresholds: feature_name -> {k: epsilon_k}
        self._thresholds: Dict[str, Dict[int, float]] = {}

        # History of all probability values (for diagnostics)
        self._probs_history: Dict[str, List[Optional[float]]] = {}

    # ================================================================== #
    #  Feature key helpers                                                  #
    # ================================================================== #

    def _feature_key(self, mode_idx: int, feat: str) -> str:
        """Return canonical string key, e.g. 'mode0_freq'."""
        return f"mode{mode_idx}_{feat}"

    def _all_feature_keys(self) -> List[str]:
        return [
            self._feature_key(m, f)
            for m in self.mode_indices
            for f in self.features
        ]

    # ================================================================== #
    #  Baseline fitting (Eqs 29-30)                                        #
    # ================================================================== #

    def fit_baseline(
        self, modal_time_series: List[Dict]
    ) -> "AnomalyDetector":
        """
        Fit a Gaussian distribution to each tracked feature using the first
        n_baseline_segments observations.

        The baseline should be drawn exclusively from Day 1 (PRE_LOWER_POUR),
        the structurally undisturbed state.

        Parameters
        ----------
        modal_time_series : List[Dict]
            Ordered list of per-segment cluster results.
            Each element is the output of ModalClusterer.process(), i.e.:
                {mode_index (int): {'freq': float, 'damping': float, ...}}

        Returns
        -------
        self  (for chaining)
        """
        # Accumulate baseline observations per feature
        baseline_values: Dict[str, List[float]] = {
            k: [] for k in self._all_feature_keys()
        }

        valid_count = 0
        for seg_result in modal_time_series:
            if valid_count >= self.n_baseline_segments:
                break

            for m_idx in self.mode_indices:
                cluster = seg_result.get(m_idx, None)
                if cluster is None:
                    continue
                for feat in self.features:
                    val = cluster.get(feat, None)
                    if val is not None and np.isfinite(val):
                        key = self._feature_key(m_idx, feat)
                        baseline_values[key].append(float(val))

            valid_count += 1

        # Fit Gaussian for each feature (Eqs. 29-30)
        self._baseline_stats = {}
        self._thresholds = {}
        self._probs_history = {k: [] for k in self._all_feature_keys()}

        for key, vals in baseline_values.items():
            if len(vals) < self.min_baseline_count:
                print(
                    f"[AnomalyDetector] WARNING: Only {len(vals)} baseline "
                    f"samples for '{key}' -- skipping (need {self.min_baseline_count})."
                )
                continue

            arr = np.array(vals)
            mu = float(np.mean(arr))             # Eq. 29
            sigma = float(np.std(arr, ddof=0))   # Eq. 30 (population std)

            if sigma < 1e-12:
                # Zero variance -- use a tiny nominal sigma to avoid division by 0
                sigma = 1e-6 * abs(mu) if abs(mu) > 0 else 1e-6

            self._baseline_stats[key] = (mu, sigma)

            # Pre-compute thresholds ε_k for each k (S.2.3 step iii)
            # ε_k = p(mu - k*sigma) = 1/sqrt(2pisigma^2) * exp(-k^2/2)
            self._thresholds[key] = {
                k: self._gaussian_pdf(mu - k * sigma, mu, sigma)
                for k in self.k_sigma_values
            }

        fitted = list(self._baseline_stats.keys())
        print(
            f"[AnomalyDetector] Baseline fitted on {valid_count} segments for "
            f"{len(fitted)} features: {fitted}"
        )
        return self

    # ================================================================== #
    #  Gaussian PDF (Eq. 28)                                               #
    # ================================================================== #

    @staticmethod
    def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
        """
        Evaluate the univariate Gaussian probability density at x.

        p(x; mu, sigma^2) = 1/sqrt(2pisigma^2) * exp(-(x-mu)^2/(2sigma^2))   (Eq. 28)
        """
        z = (x - mu) / sigma
        return float(np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi)))

    # ================================================================== #
    #  Single-segment detection                                            #
    # ================================================================== #

    def detect_segment(
        self, seg_result: Dict, k: int
    ) -> Dict[str, bool]:
        """
        Compute univariate anomaly flag for every tracked feature in one segment.

        Parameters
        ----------
        seg_result : Dict
            Output of ModalClusterer.process() for ONE segment.
        k : int
            k-sigma threshold level.

        Returns
        -------
        Dict  feature_key -> True (anomaly) / False (normal) / None (missing)
        """
        flags: Dict[str, Optional[bool]] = {}

        for m_idx in self.mode_indices:
            cluster = seg_result.get(m_idx, None)
            for feat in self.features:
                key = self._feature_key(m_idx, feat)

                if key not in self._baseline_stats:
                    flags[key] = None
                    continue

                val = cluster.get(feat, None) if cluster else None

                if val is None or not np.isfinite(val):
                    flags[key] = None
                    continue

                mu, sigma = self._baseline_stats[key]
                prob = self._gaussian_pdf(float(val), mu, sigma)

                # Store probability history for diagnostics
                self._probs_history[key].append(prob)

                threshold = self._thresholds[key][k]
                flags[key] = bool(prob < threshold)

        return flags

    # ================================================================== #
    #  Boolean operators (S.2.3, last paragraph)                            #
    # ================================================================== #

    @staticmethod
    def apply_operator(flags: Dict[str, Optional[bool]], operator: Operator) -> Optional[bool]:
        """
        Combine univariate anomaly flags into a single global indicator.

        Operators:
            AND       - anomaly only if ALL valid features are anomalous.
            OR        - anomaly if ANY valid feature is anomalous.
            FREQ_ONLY - use only frequency-based features.
            DAMP_ONLY - use only damping-based features.

        If no valid (non-None) flags exist, returns None (missing/uncertain).
        """
        if operator == "FREQ_ONLY":
            relevant = {k: v for k, v in flags.items() if "freq" in k}
        elif operator == "DAMP_ONLY":
            relevant = {k: v for k, v in flags.items() if "damping" in k}
        else:
            relevant = flags

        valid = [v for v in relevant.values() if v is not None]
        if not valid:
            return None

        if operator == "AND":
            return all(valid)
        else:  # OR, FREQ_ONLY, DAMP_ONLY
            return any(valid)

    # ================================================================== #
    #  Full time-series detection                                          #
    # ================================================================== #

    def run(
        self,
        all_modal_results: List[Dict],
        operators: Optional[List[Operator]] = None,
    ) -> Dict:
        """
        Run anomaly detection across all segments in the time series.

        Parameters
        ----------
        all_modal_results : List[Dict]
            Ordered list of per-segment cluster results (all days concatenated).
        operators : list of str or None
            Which Boolean operators to evaluate.  Defaults to
            ['AND', 'OR', 'FREQ_ONLY', 'DAMP_ONLY'].

        Returns
        -------
        Dict with structure:
            {
              'feature_values': {feature_key: [val_or_None, ...]},
              'feature_probs':  {feature_key: {k: [prob_or_None, ...]}},
              'anomaly_flags':  {
                  k: {
                      operator: [True/False/None, ...]
                  }
              },
              'baseline_stats': {feature_key: (mu, sigma)},
            }
        """
        if not self._baseline_stats:
            raise RuntimeError("Call fit_baseline() before run().")

        if operators is None:
            operators = ["AND", "OR", "FREQ_ONLY", "DAMP_ONLY"]

        n_segs = len(all_modal_results)
        all_feature_keys = self._all_feature_keys()

        # Storage
        feature_values: Dict[str, List[Optional[float]]] = {
            k: [] for k in all_feature_keys
        }
        feature_probs: Dict[str, Dict[int, List]] = {
            k: {kk: [] for kk in self.k_sigma_values}
            for k in all_feature_keys
        }
        anomaly_flags: Dict[int, Dict[str, List]] = {
            kk: {op: [] for op in operators}
            for kk in self.k_sigma_values
        }

        for seg_result in all_modal_results:
            # Collect feature values
            for m_idx in self.mode_indices:
                cluster = seg_result.get(m_idx, None)
                for feat in self.features:
                    key = self._feature_key(m_idx, feat)
                    val = cluster.get(feat, None) if cluster else None
                    feature_values[key].append(val)

            # Compute flags for each k
            for kk in self.k_sigma_values:
                flags = self.detect_segment(seg_result, kk)

                # Per-feature probabilities
                for key in all_feature_keys:
                    if key in self._baseline_stats:
                        val = feature_values[key][-1]
                        if val is not None and np.isfinite(val):
                            mu, sigma = self._baseline_stats[key]
                            prob = self._gaussian_pdf(float(val), mu, sigma)
                        else:
                            prob = None
                    else:
                        prob = None
                    feature_probs[key][kk].append(prob)

                # Boolean operators
                for op in operators:
                    global_flag = self.apply_operator(flags, op)
                    anomaly_flags[kk][op].append(global_flag)

        return {
            "feature_values": feature_values,
            "feature_probs": feature_probs,
            "anomaly_flags": anomaly_flags,
            "baseline_stats": self._baseline_stats,
            "thresholds": self._thresholds,
            "n_segments": n_segs,
        }

    # ================================================================== #
    #  Summary statistics                                                  #
    # ================================================================== #

    def baseline_summary(self) -> str:
        """Return a formatted table of baseline statistics."""
        if not self._baseline_stats:
            return "No baseline fitted yet."

        lines = ["Feature              |     mu     |     sigma  |  sigma/mu (%)"]
        lines.append("-" * 60)
        for key, (mu, sigma) in self._baseline_stats.items():
            cv = abs(sigma / mu * 100) if abs(mu) > 1e-10 else float("nan")
            lines.append(f"{key:<20} | {mu:10.4f} | {sigma:10.6f} | {cv:8.2f}")

        return "\n".join(lines)
