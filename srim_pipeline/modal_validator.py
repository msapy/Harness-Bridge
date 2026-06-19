"""
modal_validator.py
==================
Validation and quality metrics for the Harness Bridge modal survey.

Computes all metrics needed to confirm that the automated SRIM + ModeTracker
pipeline reliably identifies the correct first 3 bending modes:

  1. MAC matrices
     - Within-day:  3 × 3 matrix of MAC between day-mean mode shapes
       (should be near-identity: diagonal ≈ 1, off-diagonal < 0.1)
     - Cross-day:  10 × 10 matrix of MAC(day i mode k, day j mode k)
       for each mode k   (high diagonal → same physical mode tracked)

  2. Modal Phase Collinearity (MPC)
     - Mean and std per mode per day.

  3. Coefficient of Variation (CV)
     - CV_f  = σ_f / μ_f   (< 5 % expected for a stable mode)
     - CV_ξ  = σ_ξ / μ_ξ  (< 30 % for damping)

  4. Identification Rate (IR)
     - Fraction of segments in a day where a mode was successfully identified.

  5. Frequency shift relative to Day 1 (Δf / f_ref × 100 %)

  6. Anomaly detection metrics (Table 1 of Tran & Ozer 2020)
     - Same Gaussian k-sigma framework as the existing pipeline.
     - Ground truth: Day 1 = normal (0), Days 2-10 = anomalous (1).
     - TP, TN, FP, FN → TPR, TNR, PPV, NPV, FPR, FDR.

Methodology follows Bridge 1 analysis in Tran & Ozer (2020), Sensors 20:4752.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
TrackResult    = Dict          # {mode_idx: ClusterResult or None}
DayResults     = List[TrackResult]   # one TrackResult per segment of a day


# ===========================================================================
#  Helper: MAC between two real or complex arrays
# ===========================================================================

def _mac(phi1: np.ndarray, phi2: np.ndarray) -> float:
    """MAC between two vectors (real or complex)."""
    phi1 = np.asarray(phi1, dtype=complex)
    phi2 = np.asarray(phi2, dtype=complex)
    num = abs(phi1.conj() @ phi2) ** 2
    den = np.real(phi1.conj() @ phi1) * np.real(phi2.conj() @ phi2)
    if den < 1e-20:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return float(np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi)))


# ===========================================================================
#  ModalValidator
# ===========================================================================

class ModalValidator:
    """
    Compute all validation and quality metrics for the modal survey.

    Parameters
    ----------
    n_modes : int
        Number of tracked modes (default 3).
    n_days : int
        Number of monitoring days (default 10).
    """

    def __init__(self, n_modes: int = 3, n_days: int = 10):
        self.n_modes = n_modes
        self.n_days  = n_days

    # ================================================================== #
    #  Per-day per-mode aggregate statistics                              #
    # ================================================================== #

    def day_statistics(
        self,
        day_results: DayResults,
    ) -> Dict[int, Dict]:
        """
        Compute per-mode summary statistics for one day's segments.

        Returns
        -------
        dict: {mode_idx: {
            'freqs', 'dampings', 'shapes', 'mpc_vals', 'mac_to_ref_vals',
            'freq_mean', 'freq_std', 'freq_median',
            'damp_mean', 'damp_std', 'damp_median',
            'mpc_mean',  'mpc_std',
            'cv_freq',   'cv_damp',
            'id_rate',   'n_segs', 'n_found',
            'mean_shape'
        }}
        """
        n_segs = len(day_results)
        stats: Dict[int, Dict] = {}

        for m in range(self.n_modes):
            freqs      = []
            dampings   = []
            shapes     = []
            mpc_vals   = []
            mac_to_ref = []

            for seg_res in day_results:
                cr = seg_res.get(m, None)
                if cr is None:
                    continue
                freqs.append(cr["freq"])
                dampings.append(cr["damping"])
                if "shape" in cr and cr["shape"] is not None:
                    shapes.append(np.asarray(cr["shape"], dtype=float))
                if "mpc" in cr and cr["mpc"] is not None:
                    mpc_vals.append(cr["mpc"])
                if "mac_to_ref" in cr and cr["mac_to_ref"] is not None:
                    mac_to_ref.append(cr["mac_to_ref"])

            n_found  = len(freqs)
            id_rate  = n_found / n_segs if n_segs > 0 else 0.0

            if n_found == 0:
                stats[m] = {
                    "freqs": [], "dampings": [], "shapes": [],
                    "mpc_vals": [], "mac_to_ref_vals": [],
                    "freq_mean": None, "freq_std": None, "freq_median": None,
                    "damp_mean": None, "damp_std": None, "damp_median": None,
                    "mpc_mean":  None, "mpc_std":  None,
                    "cv_freq":   None, "cv_damp":  None,
                    "id_rate":   id_rate, "n_segs": n_segs, "n_found": 0,
                    "mean_shape": None,
                }
                continue

            f_arr = np.array(freqs)
            d_arr = np.array(dampings)

            freq_mean   = float(np.mean(f_arr))
            freq_std    = float(np.std(f_arr, ddof=1)) if n_found > 1 else 0.0
            freq_median = float(np.median(f_arr))
            damp_mean   = float(np.mean(d_arr))
            damp_std    = float(np.std(d_arr, ddof=1)) if n_found > 1 else 0.0
            damp_median = float(np.median(d_arr))

            cv_freq = freq_std / freq_mean if freq_mean > 0 else None
            cv_damp = damp_std / damp_mean if damp_mean > 0 else None

            mpc_mean = float(np.mean(mpc_vals)) if mpc_vals else None
            mpc_std  = float(np.std(mpc_vals))  if mpc_vals else None

            # Mean shape for cross-day MAC
            mean_shape = None
            if shapes:
                ref_s = shapes[0]
                aligned_shapes = []
                for s in shapes:
                    # Align sign to reference
                    if np.real(s @ ref_s) < 0:
                        aligned_shapes.append(-s)
                    else:
                        aligned_shapes.append(s)
                stacked = np.vstack(aligned_shapes)
                ms = np.mean(stacked, axis=0)
                norm = np.max(np.abs(ms))
                mean_shape = ms / norm if norm > 0 else ms

            stats[m] = {
                "freqs":           freqs,
                "dampings":        dampings,
                "shapes":          shapes,
                "mpc_vals":        mpc_vals,
                "mac_to_ref_vals": mac_to_ref,
                "freq_mean":       freq_mean,
                "freq_std":        freq_std,
                "freq_median":     freq_median,
                "damp_mean":       damp_mean,
                "damp_std":        damp_std,
                "damp_median":     damp_median,
                "mpc_mean":        mpc_mean,
                "mpc_std":         mpc_std,
                "cv_freq":         cv_freq,
                "cv_damp":         cv_damp,
                "id_rate":         id_rate,
                "n_segs":          n_segs,
                "n_found":         n_found,
                "mean_shape":      mean_shape,
            }

        return stats

    def all_days_statistics(
        self,
        per_day_results: Dict[int, DayResults],
    ) -> Dict[int, Dict[int, Dict]]:
        """
        Compute day_statistics for every day.

        Returns
        -------
        dict: {day_number: {mode_idx: stats_dict}}
        """
        return {
            day: self.day_statistics(day_results)
            for day, day_results in per_day_results.items()
        }

    # ================================================================== #
    #  Within-day MAC matrix (3 × 3 mode orthogonality)                  #
    # ================================================================== #

    def within_day_mac(
        self,
        day_stats: Dict[int, Dict],
    ) -> np.ndarray:
        """
        Compute the n_modes × n_modes MAC matrix of day-mean shapes.

        Entry (i, j) = MAC(mean_shape_i, mean_shape_j).
        The matrix should be near-identity for orthogonal physical modes.

        Returns
        -------
        np.ndarray of shape (n_modes, n_modes)
        """
        mat = np.zeros((self.n_modes, self.n_modes))
        shapes = [
            day_stats.get(m, {}).get("mean_shape", None)
            for m in range(self.n_modes)
        ]
        for i in range(self.n_modes):
            for j in range(self.n_modes):
                s1, s2 = shapes[i], shapes[j]
                if s1 is None or s2 is None:
                    mat[i, j] = np.nan
                else:
                    mat[i, j] = _mac(s1.astype(complex), s2.astype(complex))
        return mat

    # ================================================================== #
    #  Cross-day MAC matrix (10 × 10 per mode)                           #
    # ================================================================== #

    def cross_day_mac(
        self,
        all_days_stats: Dict[int, Dict[int, Dict]],
        mode_idx: int,
        days: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Compute n_days × n_days MAC matrix for one mode.

        Entry (i, j) = MAC(mean_shape_day_i, mean_shape_day_j).
        High diagonal values confirm the same physical mode is tracked
        across all days despite structural state changes.

        Returns
        -------
        mac_matrix : np.ndarray, shape (n_days, n_days)
        days_list  : list of day numbers used (for axis labels)
        """
        if days is None:
            days = sorted(all_days_stats.keys())

        n = len(days)
        mat = np.zeros((n, n))

        shapes = []
        for d in days:
            s = all_days_stats.get(d, {}).get(mode_idx, {}).get("mean_shape", None)
            shapes.append(s)

        for i in range(n):
            for j in range(n):
                s1, s2 = shapes[i], shapes[j]
                if s1 is None or s2 is None:
                    mat[i, j] = np.nan
                else:
                    mat[i, j] = _mac(s1.astype(complex), s2.astype(complex))

        return mat, days

    # ================================================================== #
    #  Frequency shift relative to Day 1                                  #
    # ================================================================== #

    def frequency_shifts(
        self,
        all_days_stats: Dict[int, Dict[int, Dict]],
        baseline_day: int = 1,
    ) -> Dict[int, Dict[int, Optional[float]]]:
        """
        Compute Δf_i (%) = (f_day - f_day1) / f_day1 × 100 %.

        Returns
        -------
        dict: {mode_idx: {day: delta_f_pct or None}}
        """
        shifts: Dict[int, Dict[int, Optional[float]]] = {
            m: {} for m in range(self.n_modes)
        }
        days = sorted(all_days_stats.keys())

        for m in range(self.n_modes):
            f_ref = all_days_stats.get(baseline_day, {}).get(m, {}).get("freq_median", None)
            for d in days:
                f_d = all_days_stats.get(d, {}).get(m, {}).get("freq_median", None)
                if f_ref is None or f_d is None or f_ref == 0:
                    shifts[m][d] = None
                else:
                    shifts[m][d] = (f_d - f_ref) / f_ref * 100.0

        return shifts

    # ================================================================== #
    #  Anomaly detection (Gaussian k-sigma, Bridge 1 methodology)        #
    # ================================================================== #

    def fit_gaussian_baseline(
        self,
        baseline_day_results: DayResults,
        features: Tuple[str, ...] = ("freq", "damping"),
    ) -> Dict[str, Tuple[float, float]]:
        """
        Fit Gaussian baselines (μ, σ) to modal features from Day 1.

        Parameters
        ----------
        baseline_day_results : list of TrackResult (Day 1 only)
        features : which features to fit ('freq', 'damping')

        Returns
        -------
        dict: {'mode{i}_{feat}': (mu, sigma)}
        """
        stats: Dict[str, Tuple[float, float]] = {}

        for m in range(self.n_modes):
            for feat in features:
                vals = []
                for seg_res in baseline_day_results:
                    cr = seg_res.get(m, None)
                    if cr is None:
                        continue
                    v = cr.get(feat, None)
                    if v is not None and np.isfinite(v):
                        vals.append(float(v))

                if len(vals) < 5:
                    continue

                arr   = np.array(vals)
                mu    = float(np.mean(arr))
                sigma = float(np.std(arr, ddof=0))
                if sigma < 1e-12:
                    sigma = 1e-6 * abs(mu) if abs(mu) > 0 else 1e-6

                key = f"mode{m}_{feat}"
                stats[key] = (mu, sigma)

        return stats

    def detect_anomalies(
        self,
        all_track_results: List[TrackResult],
        baseline_stats: Dict[str, Tuple[float, float]],
        k_sigma_values: Tuple[int, ...] = (1, 2, 3),
        operators: Tuple[str, ...] = ("AND", "OR", "FREQ_ONLY", "DAMP_ONLY"),
    ) -> Dict:
        """
        Apply Gaussian k-sigma anomaly detection (Eqs. 28-31, Tran & Ozer 2020).

        Returns
        -------
        dict with 'anomaly_flags': {k: {operator: [True/False/None, ...]}}
        """
        # Pre-compute thresholds
        thresholds: Dict[str, Dict[int, float]] = {}
        for key, (mu, sigma) in baseline_stats.items():
            thresholds[key] = {
                k: _gaussian_pdf(mu - k * sigma, mu, sigma)
                for k in k_sigma_values
            }

        flags_out: Dict[int, Dict[str, List]] = {
            k: {op: [] for op in operators}
            for k in k_sigma_values
        }

        for seg_res in all_track_results:
            # Compute per-feature anomaly flags for this segment
            seg_flags: Dict[str, Optional[bool]] = {}

            for key, (mu, sigma) in baseline_stats.items():
                parts = key.split("_")   # 'mode{m}_{feat}'
                m    = int(parts[0].replace("mode", ""))
                feat = "_".join(parts[1:])

                cr  = seg_res.get(m, None)
                val = cr.get(feat, None) if cr else None

                if val is None or not np.isfinite(val):
                    seg_flags[key] = None
                    continue

                prob = _gaussian_pdf(float(val), mu, sigma)

                for k in k_sigma_values:
                    thresh = thresholds[key][k]
                    seg_flags[key] = bool(prob < thresh)
                    # (overwritten each k — we handle per-k below)

            # Per-k flags with correct threshold
            for k in k_sigma_values:
                seg_flags_k: Dict[str, Optional[bool]] = {}

                for key, (mu, sigma) in baseline_stats.items():
                    parts = key.split("_")
                    m    = int(parts[0].replace("mode", ""))
                    feat = "_".join(parts[1:])

                    cr  = seg_res.get(m, None)
                    val = cr.get(feat, None) if cr else None

                    if val is None or not np.isfinite(val):
                        seg_flags_k[key] = None
                    else:
                        prob   = _gaussian_pdf(float(val), mu, sigma)
                        thresh = thresholds[key][k]
                        seg_flags_k[key] = bool(prob < thresh)

                # Boolean operators
                for op in operators:
                    if op == "FREQ_ONLY":
                        relevant = {k2: v for k2, v in seg_flags_k.items() if "freq" in k2}
                    elif op == "DAMP_ONLY":
                        relevant = {k2: v for k2, v in seg_flags_k.items() if "damping" in k2}
                    else:
                        relevant = seg_flags_k

                    valid = [v for v in relevant.values() if v is not None]
                    if not valid:
                        global_flag = None
                    elif op == "AND":
                        global_flag = all(valid)
                    else:
                        global_flag = any(valid)

                    flags_out[k][op].append(global_flag)

        return {"anomaly_flags": flags_out}

    def compute_detection_metrics(
        self,
        anomaly_flags: Dict[int, Dict[str, List]],
        true_labels: List[int],
    ) -> Dict[int, Dict[str, Dict]]:
        """
        Compute TP/TN/FP/FN and all Table 1 metrics for all k and operators.

        Returns
        -------
        dict: {k: {operator: metrics_dict}}
        """
        results: Dict[int, Dict[str, Dict]] = {}

        for k, op_dict in anomaly_flags.items():
            results[k] = {}
            for op, preds in op_dict.items():
                pred_arr = np.array(
                    [1 if p is True else 0 for p in preds], dtype=int
                )
                true_arr = np.array(true_labels[:len(pred_arr)], dtype=int)

                TP = int(np.sum((pred_arr == 1) & (true_arr == 1)))
                TN = int(np.sum((pred_arr == 0) & (true_arr == 0)))
                FP = int(np.sum((pred_arr == 1) & (true_arr == 0)))
                FN = int(np.sum((pred_arr == 0) & (true_arr == 1)))

                def safe_div(n: int, d: int) -> Optional[float]:
                    return float(n) / d if d > 0 else None

                results[k][op] = {
                    "TP": TP, "TN": TN, "FP": FP, "FN": FN,
                    "TPR": safe_div(TP, TP + FN),
                    "TNR": safe_div(TN, TN + FP),
                    "PPV": safe_div(TP, TP + FP),
                    "NPV": safe_div(TN, TN + FN),
                    "FPR": safe_div(FP, FP + TN),
                    "FDR": safe_div(FP, FP + TP),
                    "FNR": safe_div(FN, FN + TP),
                    "FOR": safe_div(FN, FN + TN),
                }

        return results

    # ================================================================== #
    #  LaTeX table generators                                             #
    # ================================================================== #

    def latex_modal_summary(
        self,
        all_days_stats: Dict[int, Dict[int, Dict]],
        caption: str = "Per-day per-mode modal parameter summary.",
        label: str = "tab:modal_summary",
    ) -> str:
        """
        Generate LaTeX table: days × modes, columns = μ_f ± σ_f, μ_ξ ± σ_ξ, MPC, IR.
        """
        n_modes = self.n_modes
        days    = sorted(all_days_stats.keys())

        # Build column spec: Day | (f_med ± std, xi_med ± std, MPC, IR) × n_modes
        col_spec = "l" + ("cccc" * n_modes)
        mode_headers = " & ".join(
            [f"\\multicolumn{{4}}{{c}}{{Mode {m+1}}}" for m in range(n_modes)]
        )
        sub_headers = " & ".join(
            [r"$\hat{f}$ (Hz) & $\hat{\xi}$ (\%) & MPC & IR (\%)"] * n_modes
        )

        rows = []
        rows.append(r"\begin{table}[htbp]")
        rows.append(r"\centering")
        rows.append(r"\small")
        rows.append(f"\\caption{{{caption}}}")
        rows.append(f"\\label{{{label}}}")
        rows.append(f"\\begin{{tabular}}{{{col_spec}}}")
        rows.append(r"\toprule")
        rows.append(f"Day & {mode_headers} \\\\")

        # Sub-header midrules
        col_start = 2
        for m in range(n_modes):
            col_end = col_start + 3
            rows.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
            col_start = col_end + 1

        rows.append(f"& {sub_headers} \\\\")
        rows.append(r"\midrule")

        def _fmt_f(s: Dict) -> str:
            if s.get("freq_median") is None:
                return r"\multicolumn{4}{c}{--}"
            f   = s["freq_median"]
            fs  = s["freq_std"] or 0.0
            xi  = (s["damp_median"] or 0.0) * 100
            xis = (s["damp_std"]   or 0.0) * 100
            mpc = s["mpc_mean"]
            ir  = s["id_rate"] * 100
            mpc_str = f"{mpc:.2f}" if mpc is not None else "--"
            return (
                f"${f:.2f}\\pm{fs:.2f}$ & "
                f"${xi:.2f}\\pm{xis:.2f}$ & "
                f"{mpc_str} & "
                f"{ir:.0f}"
            )

        for d in days:
            d_stats  = all_days_stats.get(d, {})
            row_vals = " & ".join([_fmt_f(d_stats.get(m, {})) for m in range(n_modes)])
            rows.append(f"{d} & {row_vals} \\\\")

        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}")
        rows.append(r"\end{table}")
        return "\n".join(rows)

    def latex_freq_shift(
        self,
        shifts: Dict[int, Dict[int, Optional[float]]],
        days: List[int],
        caption: str = "Natural frequency shift relative to Day~1 baseline (\\%).",
        label: str = "tab:freq_shift",
    ) -> str:
        """
        Generate LaTeX table: days × modes showing Δf (%).
        """
        col_spec = "l" + "c" * self.n_modes
        mode_header = " & ".join([f"Mode {m+1}" for m in range(self.n_modes)])

        rows = []
        rows.append(r"\begin{table}[htbp]")
        rows.append(r"\centering")
        rows.append(f"\\caption{{{caption}}}")
        rows.append(f"\\label{{{label}}}")
        rows.append(f"\\begin{{tabular}}{{{col_spec}}}")
        rows.append(r"\toprule")
        rows.append(f"Day & {mode_header} \\\\")
        rows.append(r"\midrule")

        for d in days:
            vals = []
            for m in range(self.n_modes):
                v = shifts.get(m, {}).get(d, None)
                vals.append(f"{v:+.2f}" if v is not None else "--")
            rows.append(f"{d} & " + " & ".join(vals) + r" \\")

        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}")
        rows.append(r"\end{table}")
        return "\n".join(rows)

    def latex_mac_within(
        self,
        all_days_mac: Dict[int, np.ndarray],
        days: List[int],
        caption: str = "Within-day MAC matrices between the three tracked modes.",
        label: str = "tab:mac_within",
    ) -> str:
        """
        Generate LaTeX table of within-day 3×3 MAC matrices, one subtable per day.
        """
        rows = []
        rows.append(r"\begin{table}[htbp]")
        rows.append(r"\centering")
        rows.append(f"\\caption{{{caption}}}")
        rows.append(f"\\label{{{label}}}")

        n = self.n_modes
        col_spec = "c" + "c" * n
        mode_labels = " & ".join([f"M{m+1}" for m in range(n)])

        for d in days:
            mac_mat = all_days_mac.get(d, None)
            rows.append(r"\begin{subtable}[t]{0.18\linewidth}")
            rows.append(r"\centering")
            rows.append(f"\\caption*{{Day {d}}}")
            rows.append(f"\\begin{{tabular}}{{{col_spec}}}")
            rows.append(r"\toprule")
            rows.append(f"& {mode_labels} \\\\")
            rows.append(r"\midrule")

            if mac_mat is None:
                for i in range(n):
                    row_vals = " & ".join(["--"] * n)
                    rows.append(f"M{i+1} & {row_vals} \\\\")
            else:
                for i in range(n):
                    vals = []
                    for j in range(n):
                        v = mac_mat[i, j]
                        if np.isnan(v):
                            vals.append("--")
                        elif i == j:
                            vals.append(f"\\textbf{{{v:.3f}}}")
                        else:
                            vals.append(f"{v:.3f}")
                    rows.append(f"M{i+1} & " + " & ".join(vals) + r" \\")

            rows.append(r"\bottomrule")
            rows.append(r"\end{tabular}")
            rows.append(r"\end{subtable}")
            rows.append(r"\hfill")

        rows.append(r"\end{table}")
        return "\n".join(rows)

    def latex_validation_metrics(
        self,
        all_days_stats: Dict[int, Dict[int, Dict]],
        caption: str = "Modal identification quality metrics per day and mode.",
        label: str = "tab:validation",
    ) -> str:
        """
        Generate LaTeX table: CV_f (%), CV_ξ (%), MPC mean, IR (%) per mode per day.
        """
        days    = sorted(all_days_stats.keys())
        n_modes = self.n_modes
        col_spec = "l" + ("cccc" * n_modes)
        mode_header = " & ".join([
            f"\\multicolumn{{4}}{{c}}{{Mode {m+1}}}" for m in range(n_modes)
        ])
        sub_header = " & ".join(
            [r"CV$_f$ & CV$_\xi$ & MPC & IR (\%)"] * n_modes
        )

        rows = []
        rows.append(r"\begin{table}[htbp]")
        rows.append(r"\centering")
        rows.append(r"\small")
        rows.append(f"\\caption{{{caption}}}")
        rows.append(f"\\label{{{label}}}")
        rows.append(f"\\begin{{tabular}}{{{col_spec}}}")
        rows.append(r"\toprule")
        rows.append(f"Day & {mode_header} \\\\")

        col_start = 2
        for m in range(n_modes):
            col_end = col_start + 3
            rows.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
            col_start = col_end + 1

        rows.append(f"& {sub_header} \\\\")
        rows.append(r"\midrule")

        def _fmt(s: Dict) -> str:
            if not s or s.get("freq_median") is None:
                return r"\multicolumn{4}{c}{--}"
            cvf = (s.get("cv_freq") or 0.0) * 100
            cvd = (s.get("cv_damp") or 0.0) * 100
            mpc = s.get("mpc_mean")
            ir  = s.get("id_rate", 0.0) * 100
            mpc_str = f"{mpc:.2f}" if mpc is not None else "--"
            return f"{cvf:.1f} & {cvd:.1f} & {mpc_str} & {ir:.0f}"

        for d in days:
            d_stats  = all_days_stats.get(d, {})
            row_vals = " & ".join([_fmt(d_stats.get(m, {})) for m in range(n_modes)])
            rows.append(f"{d} & {row_vals} \\\\")

        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}")
        rows.append(r"\end{table}")
        return "\n".join(rows)

    def latex_detection_metrics(
        self,
        detection_metrics: Dict[int, Dict[str, Dict]],
        caption: str = "Anomaly detection performance metrics (Tran \\& Ozer 2020, Table~1).",
        label: str = "tab:detection",
    ) -> str:
        """
        Generate LaTeX table: TPR, TNR, PPV, NPV, FPR, FDR per k and operator.
        """
        favour   = ["TPR", "TNR", "PPV", "NPV"]
        unfavour = ["FPR", "FDR"]
        all_cols = favour + unfavour
        col_spec = "ll" + "c" * len(all_cols)
        header   = " & ".join(all_cols)

        rows = []
        rows.append(r"\begin{table}[htbp]")
        rows.append(r"\centering")
        rows.append(r"\small")
        rows.append(f"\\caption{{{caption}}}")
        rows.append(f"\\label{{{label}}}")
        rows.append(f"\\begin{{tabular}}{{{col_spec}}}")
        rows.append(r"\toprule")
        rows.append(f"$k$ & Operator & {header} \\\\")
        rows.append(r"\midrule")

        for k in sorted(detection_metrics.keys()):
            op_dict = detection_metrics[k]
            first = True
            for op, mets in op_dict.items():
                vals = []
                for col in all_cols:
                    v = mets.get(col, None)
                    vals.append(f"{v:.3f}" if v is not None else "--")
                k_str = str(k) if first else ""
                rows.append(f"{k_str} & {op} & " + " & ".join(vals) + r" \\")
                first = False
            rows.append(r"\midrule")

        rows.append(r"\bottomrule")
        rows.append(r"\end{tabular}")
        rows.append(r"\end{table}")
        return "\n".join(rows)
