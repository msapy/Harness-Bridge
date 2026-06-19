"""
metrics_evaluator.py
====================
MetricsEvaluator class: computes all eight information retrieval metrics
from Table 1 of Tran & Ozer (2020) to quantify anomaly detection performance
against ground-truth structural state labels.

Structural state context (Harness Bridge)
------------------------------------------
Rather than "damage" vs "healthy", the labels here reflect structural state
changes caused by concrete pours:

    Label 0  (NORMAL / no anomaly expected):
        -> Day 1: Pre-lower-deck-pour baseline.

    Label 1  (ANOMALOUS / structural state change):
        -> Day 2:  Lower deck concrete poured (significant mass addition).
        -> Day 3-7: Post lower deck pour (increased mass, lower frequencies).
        -> Day 8:  Upper deck pour activity (mass + dynamic loading).
        -> Day 9+: Post upper deck pour (maximum mass state).

Metrics (Table 1)
-----------------

+------------------------------------+----------------------------------+
|  Metric                            |  Formula                         |
|------------------------------------+----------------------------------|
|  TPR  - Recall / True Positive Rate|  TP / (TP + FN)                  |
|  TNR  - Selectivity / True Neg Rate|  TN / (TN + FP)                  |
|  PPV  - Precision / Pos Pred Value |  TP / (TP + FP)                  |
|  NPV  - Negative Predictive Value  |  TN / (TN + FN)                  |
|  FPR  - Fall-out / False Pos Rate  |  FP / (FP + TN)                  |
|  FDR  - False Discovery Rate       |  FP / (FP + TP)                  |
|  FNR  - Miss Rate / False Neg Rate |  FN / (FN + TP)                  |
|  FOR  - False Omission Rate        |  FN / (FN + TN)                  |
+------------------------------------+----------------------------------+

Favoured metrics (higher is better): TPR, TNR, PPV, NPV.
Unfavoured metrics (lower is better): FPR, FDR, FNR, FOR.

Relationship to threshold k (paper):
    Decreasing k  ->  stricter detection  ->  TPR(up), FPR(up)
    Increasing k  ->  looser detection   ->  TNR(up), FNR(up)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Structural state label definitions
# ---------------------------------------------------------------------------

# Day -> (state_id, state_name)
DAY_STATES = {
    1:  (0, "PRE_LOWER_POUR"),     # Baseline - label 0
    2:  (1, "POST_LOWER_POUR"),    # After lower deck pour - label 1
    3:  (1, "POST_LOWER_POUR"),
    4:  (1, "POST_LOWER_POUR"),
    5:  (1, "POST_LOWER_POUR"),
    6:  (1, "POST_LOWER_POUR"),
    7:  (1, "POST_LOWER_POUR"),
    8:  (2, "UPPER_POUR_ACTIVITY"),# During upper deck pour - label 1
    9:  (3, "POST_UPPER_POUR"),    # After upper deck pour - label 1
    10: (3, "POST_UPPER_POUR"),
}

# Any state > 0 is "anomalous" (structural state has changed)
ANOMALY_LABEL = 1
NORMAL_LABEL = 0


class MetricsEvaluator:
    """
    Compute information retrieval metrics from Table 1.

    Parameters
    ----------
    true_labels : List[int]
        Ground-truth binary labels for each segment (0 = normal, 1 = anomaly).
        Must be the same length as the predicted labels passed to compute_metrics().
    """

    def __init__(self, true_labels: List[int]):
        self.true_labels = np.asarray(true_labels, dtype=int)

    # ================================================================== #
    #  Label construction helpers                                          #
    # ================================================================== #

    @staticmethod
    def build_labels_from_day_counts(
        segments_per_day: Dict[int, int],
        anomaly_days: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Build a flat list of true labels given how many segments each day
        contributed.

        Parameters
        ----------
        segments_per_day : dict
            {day_number: n_segments}.  Day numbers should be 1-based.
        anomaly_days : list of int or None
            Which days are considered structurally "anomalous" (label=1).
            If None, uses the default Harness Bridge scheme (days 2-10 = 1).

        Returns
        -------
        List[int] - one label per segment.
        """
        if anomaly_days is None:
            # Default: Day 1 is baseline (normal), all others are state-changed
            anomaly_days = [d for d in segments_per_day if d != 1]

        labels = []
        for day, n_segs in sorted(segments_per_day.items()):
            lbl = ANOMALY_LABEL if day in anomaly_days else NORMAL_LABEL
            labels.extend([lbl] * n_segs)

        return labels

    # ================================================================== #
    #  Confusion matrix                                                    #
    # ================================================================== #

    def confusion_matrix(
        self, predicted: List[Optional[bool]]
    ) -> Tuple[int, int, int, int]:
        """
        Compute the four cells of the confusion matrix.

        Segments where predicted = None (missing modal identification) are
        treated conservatively as False Negatives (missed detections) when
        true_label=1, and as True Negatives when true_label=0.

        Returns
        -------
        (TP, TN, FP, FN)
        """
        pred_arr = np.array(
            [1 if p is True else 0 for p in predicted], dtype=int
        )
        true = self.true_labels[: len(pred_arr)]

        TP = int(np.sum((pred_arr == 1) & (true == 1)))
        TN = int(np.sum((pred_arr == 0) & (true == 0)))
        FP = int(np.sum((pred_arr == 1) & (true == 0)))
        FN = int(np.sum((pred_arr == 0) & (true == 1)))

        return TP, TN, FP, FN

    # ================================================================== #
    #  Eight metrics (Table 1)                                             #
    # ================================================================== #

    def compute_metrics(
        self, predicted: List[Optional[bool]]
    ) -> Dict[str, Optional[float]]:
        """
        Compute all eight information retrieval metrics.

        Parameters
        ----------
        predicted : List[bool or None]
            Binary anomaly predictions from AnomalyDetector.run()
            for a specific k and operator.

        Returns
        -------
        Dict with keys: 'TP', 'TN', 'FP', 'FN',
                        'TPR', 'TNR', 'PPV', 'NPV',
                        'FPR', 'FDR', 'FNR', 'FOR'
            Values are floats in [0, 1] or None if denominator is zero.
        """
        TP, TN, FP, FN = self.confusion_matrix(predicted)

        def safe_div(num: int, den: int) -> Optional[float]:
            return float(num) / den if den > 0 else None

        metrics = {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            # Favoured metrics ((up) better)
            "TPR": safe_div(TP, TP + FN),   # Recall / Sensitivity
            "TNR": safe_div(TN, TN + FP),   # Selectivity / Specificity
            "PPV": safe_div(TP, TP + FP),   # Precision / Pos Pred Value
            "NPV": safe_div(TN, TN + FN),   # Negative Predictive Value
            # Unfavoured metrics ((dn) better)
            "FPR": safe_div(FP, FP + TN),   # Fall-out
            "FDR": safe_div(FP, FP + TP),   # False Discovery Rate
            "FNR": safe_div(FN, FN + TP),   # Miss Rate
            "FOR": safe_div(FN, FN + TN),   # False Omission Rate
        }

        return metrics

    # ================================================================== #
    #  Multi-threshold sweep                                               #
    # ================================================================== #

    def evaluate_all(
        self,
        anomaly_flags: Dict[int, Dict[str, List[Optional[bool]]]],
    ) -> Dict[int, Dict[str, Dict]]:
        """
        Evaluate metrics for all k-sigma levels and all operators.

        Parameters
        ----------
        anomaly_flags : dict
            Output of AnomalyDetector.run()['anomaly_flags']:
            {k: {operator: [bool/None, ...]}}

        Returns
        -------
        Nested dict: {k: {operator: metrics_dict}}
        """
        results = {}
        for k, op_dict in anomaly_flags.items():
            results[k] = {}
            for op, pred_list in op_dict.items():
                results[k][op] = self.compute_metrics(pred_list)

        return results

    # ================================================================== #
    #  Pretty-print / reporting                                            #
    # ================================================================== #

    @staticmethod
    def format_table(
        all_results: Dict[int, Dict[str, Dict]],
        operators: Optional[List[str]] = None,
    ) -> str:
        """
        Format evaluation results as a readable ASCII table.

        Parameters
        ----------
        all_results : dict
            Output of evaluate_all().
        operators : list of str or None
            Subset of operators to include (default: all).

        Returns
        -------
        str - formatted table.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ANOMALY DETECTION PERFORMANCE METRICS  (Tran & Ozer 2020, Table 1)")
        lines.append("=" * 80)

        favour_metrics = ["TPR", "TNR", "PPV", "NPV"]
        unfavour_metrics = ["FPR", "FDR", "FNR", "FOR"]
        count_metrics = ["TP", "TN", "FP", "FN"]

        for k, op_dict in sorted(all_results.items()):
            lines.append(f"\n-- k = {k}sigma threshold --")
            header = f"{'Operator':<15}" + "".join(
                f"{m:>8}" for m in count_metrics + favour_metrics + unfavour_metrics
            )
            lines.append(header)
            lines.append("-" * len(header))

            for op, metrics in op_dict.items():
                if operators and op not in operators:
                    continue

                row = f"{op:<15}"
                for m in count_metrics:
                    val = metrics.get(m)
                    row += f"{val:>8d}" if val is not None else f"{'N/A':>8}"
                for m in favour_metrics + unfavour_metrics:
                    val = metrics.get(m)
                    if val is None:
                        row += f"{'N/A':>8}"
                    else:
                        row += f"{val:>8.3f}"
                lines.append(row)

        lines.append("\nFavoured (up): TPR, TNR, PPV, NPV   |   Unfavoured (dn): FPR, FDR, FNR, FOR")
        return "\n".join(lines)

    @staticmethod
    def best_operator(
        results_for_k: Dict[str, Dict], metric: str = "TPR"
    ) -> Tuple[str, float]:
        """
        Find the operator with the highest value of a given metric for one k.

        Parameters
        ----------
        results_for_k : dict   {operator -> metrics_dict}
        metric : str           Which metric to maximise.

        Returns
        -------
        (best_operator_name, best_metric_value)
        """
        best_op = None
        best_val = -np.inf
        for op, mets in results_for_k.items():
            val = mets.get(metric)
            if val is not None and val > best_val:
                best_val = val
                best_op = op
        return best_op, best_val
