"""
srim_shm_pipeline.py
====================
Main runner for the simultaneous modal parameter anomaly detection pipeline.

Orchestrates the full workflow described in Tran & Ozer (2020), Sensors 20(17):4752,
applied to the Harness Bridge 10-day accelerometer dataset:

    Day 1  - 20 May 2026  -> BASELINE (pre-lower-deck-pour)
    Day 2  - 21 May 2026  -> POST lower deck pour  <- structural state change
    Day 3  - 22 May 2026  -> Post lower deck pour
    Day 4  - 25 May 2026  -> Post lower deck pour
    Day 5  - 27 May 2026  -> Post lower deck pour
    Day 6  - 29 May 2026  -> Post lower deck pour
    Day 7  -  2 Jun 2026  -> Post lower deck pour (J7 interpolated)
    Day 8  -  3 Jun 2026  -> UPPER DECK POUR activity (transition)
    Day 9  -  8 Jun 2026  -> Post upper deck pour  <- heavier state
    Day 10 - 10 Jun 2026  -> Post upper deck pour

Pipeline steps
--------------
1.  DataSegmenter   - overlapping windows (20 s, 50 % overlap)
2.  SRIMIdentifier  - output-only SRIM per window -> raw poles per order
3.  ModalClusterer  - stabilization -> clearance (MPC) -> clustering -> MAD
4.  AnomalyDetector - Gaussian baseline (Day 1) -> k-sigma thresholds -> flags
5.  MetricsEvaluator- TPR / TNR / PPV / FPR / FDR etc.
6.  Visualisation   - matplotlib plots (stabilization diagram, modal time-series,
                      anomaly flags, metrics table)

Usage
-----
    # Quick test on Day 1 only:
    python srim_shm_pipeline.py --days 1 --n-segs 20

    # Full run, all 10 days:
    python srim_shm_pipeline.py

    # Custom parameters:
    python srim_shm_pipeline.py --seg-length 30 --overlap 0.5 --max-order 30

Run from the parent directory (Harness Bridge/).
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/batch use
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------- Local modules ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from data_segmenter   import DataSegmenter
from srim_identifier  import SRIMIdentifier
from modal_clusterer  import ModalClusterer
from anomaly_detector import AnomalyDetector
from metrics_evaluator import MetricsEvaluator

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "Tests" / "Tests"
OUTPUT_DIR = Path(__file__).parent / "output"

# All 10 days in chronological order
ALL_DAYS = list(range(1, 11))

# Structural state labels for metrics evaluation
# Day 1 = normal (0); all others = anomalous (structural state change, 1)
STATE_LABELS = {
    1:  0,   # PRE_LOWER_POUR -- baseline
    2:  1,   # POST_LOWER_POUR
    3:  1,
    4:  1,
    5:  1,
    6:  1,
    7:  1,
    8:  1,   # UPPER_POUR_ACTIVITY
    9:  1,   # POST_UPPER_POUR
    10: 1,
}

# Plot colour scheme for structural states
STATE_COLORS = {
    0: "#22c55e",   # green - normal
    1: "#f97316",   # orange - post lower pour
    2: "#ef4444",   # red - upper pour activity
    3: "#a855f7",   # purple - post upper pour
}
STATE_NAMES = {
    0: "Baseline (Pre-pour)",
    1: "Post Lower Deck Pour",
    2: "Upper Deck Pour Activity",
    3: "Post Upper Deck Pour",
}
DAY_STATES = {
    1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 3, 10: 3
}


# ---------------------------------------------------------------------------
# Helper: find CSV for a given day
# ---------------------------------------------------------------------------

def find_csv(day: int, data_dir: Path) -> Optional[Path]:
    """Locate the cleaned CSV for a given day number."""
    candidate = data_dir / f"day{day} Clean.csv"
    if candidate.exists():
        return candidate
    # fallback - look for any match
    for p in data_dir.glob(f"day{day}*.csv"):
        if "Clean" in p.name:
            return p
    return None


# ---------------------------------------------------------------------------
# Core processing: one day
# ---------------------------------------------------------------------------

def process_day(
    day: int,
    data_dir: Path,
    segmenter_kwargs: dict,
    srim_kwargs: dict,
    clusterer_kwargs: dict,
    max_segments: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[Dict], List[str], int]:
    """
    Load, segment, identify, and cluster one day's data.

    Returns
    -------
    modal_results : List[Dict]
        One dict per segment (ModalClusterer.process() output).
    start_times : List[str]
        Timestamp string at the start of each segment.
    n_segs_processed : int
        Number of segments actually processed.
    """
    csv_path = find_csv(day, data_dir)
    if csv_path is None:
        print(f"[Day {day}] WARNING: CSV not found in {data_dir}")
        return [], [], 0

    # -- Segmentation -----------------------------------------------------
    seg = DataSegmenter(**segmenter_kwargs)
    seg.load_csv(csv_path)
    segments = seg.get_segments()

    if max_segments is not None:
        segments = segments[:max_segments]

    # -- SRIM + Clustering per segment -------------------------------------
    srim = SRIMIdentifier(**srim_kwargs)
    clusterer = ModalClusterer(**clusterer_kwargs)

    modal_results: List[Dict] = []
    start_times: List[str] = []
    n_ok = 0

    t0 = time.time()
    for seg_info in segments:
        data = seg_info["data"]   # (N_seg, m)
        seg_idx = seg_info["seg_index"]

        try:
            # Output-only SRIM: returns {order: [Pole, ...]}
            # Suppress the expected complex-log RuntimeWarning (non-physical
            # poles with |mu| > 1 produce log of complex numbers; they are
            # filtered out by the positive-damping check downstream).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                poles_by_order = srim.identify(data)

            # Stabilization -> clearance -> clustering -> MAD
            result = clusterer.process(poles_by_order)

        except Exception as exc:
            warnings.warn(
                f"[Day {day} seg {seg_idx}] SRIM/cluster failed: {exc}"
            )
            result = {}

        modal_results.append(result)
        start_times.append(seg_info["start_time"])
        if result:
            n_ok += 1

    elapsed = time.time() - t0
    if verbose:
        print(
            f"[Day {day}]  {len(segments)} segs processed in {elapsed:.1f}s  "
            f"({n_ok} with identified modes, "
            f"{len(segments)-n_ok} empty)"
        )

    return modal_results, start_times, len(segments)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_stabilization_diagram(
    poles_by_order: Dict,
    day: int,
    seg_idx: int,
    clusterer: ModalClusterer,
    out_dir: Path,
    fs: float = 128.0,
):
    """
    Plot frequency vs. system order (stabilization diagram) for one segment.
    Colour-codes: all poles (grey), type-3 stable (blue), cleared/physical (orange).
    """
    diag = clusterer.stabilization_diagram_data(poles_by_order)
    orders = sorted(poles_by_order.keys())

    fig, ax = plt.subplots(figsize=(14, 6),
                           facecolor="#0f172a")
    ax.set_facecolor("#0f172a")

    # All poles - grey dots
    for order, plist in poles_by_order.items():
        for p in plist:
            ax.scatter(
                p["freq"], order,
                c="#374151", s=8, alpha=0.4, linewidths=0, zorder=1
            )

    # Stable (type-3) - blue crosses
    for p in diag["stable"]:
        ax.scatter(
            p["freq"], p["order"],
            c="#60a5fa", s=20, marker="+", alpha=0.7,
            linewidths=1.2, zorder=2
        )

    # Cleared (physical) - orange circles
    for p in diag["cleared"]:
        ax.scatter(
            p["freq"], p["order"],
            c="#fb923c", s=24, marker="o", alpha=0.9,
            linewidths=0, zorder=3
        )

    ax.set_xlim(0, fs / 2)
    ax.set_ylim(min(orders) - 1, max(orders) + 1)
    ax.set_xlabel("Frequency (Hz)", color="#94a3b8", fontsize=11)
    ax.set_ylabel("System Order", color="#94a3b8", fontsize=11)
    ax.tick_params(colors="#94a3b8")
    ax.set_title(
        f"Stabilization Diagram -- Day {day}, Segment {seg_idx}",
        color="#e2e8f0", fontsize=13, fontweight="bold", pad=12
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.grid(True, color="#1e293b", linewidth=0.5)

    legend_elements = [
        mpatches.Patch(color="#374151", label="All poles"),
        mpatches.Patch(color="#60a5fa", label="Stable (freq + MAC)"),
        mpatches.Patch(color="#fb923c", label="Cleared (MPC > threshold)"),
    ]
    ax.legend(handles=legend_elements, facecolor="#1e293b",
              edgecolor="#334155", labelcolor="#e2e8f0", fontsize=9)

    fig.tight_layout()
    out_path = out_dir / f"stabilization_day{day}_seg{seg_idx}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def plot_modal_timeseries(
    all_results: List[Dict],
    all_start_times: List[str],
    segments_per_day: Dict[int, int],
    out_dir: Path,
    mode_indices: Tuple[int, ...] = (0, 1),
):
    """
    Plot time-series of identified natural frequencies and damping ratios
    across all days, colour-coded by structural state.
    """
    n_modes = len(mode_indices)
    fig, axes = plt.subplots(
        n_modes * 2, 1,
        figsize=(18, 4 * n_modes * 2),
        facecolor="#0f172a",
        sharex=True
    )
    if n_modes * 2 == 1:
        axes = [axes]

    for ax in axes:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.grid(True, color="#1e293b", linewidth=0.5, axis="y")

    seg_x = np.arange(len(all_results))

    # Map segment -> day
    seg_days = []
    for day, n_segs in sorted(segments_per_day.items()):
        seg_days.extend([day] * n_segs)
    seg_days = seg_days[:len(all_results)]

    # Plot each mode
    for row, m_idx in enumerate(mode_indices):
        ax_freq = axes[row * 2]
        ax_damp = axes[row * 2 + 1]

        freqs, damps, colors_f, colors_d, xs = [], [], [], [], []

        for seg_i, result in enumerate(all_results):
            cluster = result.get(m_idx, None)
            if cluster is None:
                continue
            f = cluster.get("freq")
            d = cluster.get("damping")
            if f is None or d is None:
                continue
            if not (np.isfinite(f) and np.isfinite(d)):
                continue

            day = seg_days[seg_i] if seg_i < len(seg_days) else 1
            state = DAY_STATES.get(day, 0)
            c = STATE_COLORS.get(state, "#64748b")

            freqs.append(f)
            damps.append(d * 100)    # % for readability
            colors_f.append(c)
            colors_d.append(c)
            xs.append(seg_i)

        if freqs:
            ax_freq.scatter(xs, freqs, c=colors_f, s=10, alpha=0.75,
                            linewidths=0, zorder=2)
            ax_damp.scatter(xs, damps, c=colors_d, s=10, alpha=0.75,
                            linewidths=0, zorder=2)

        ax_freq.set_ylabel(
            f"Mode {m_idx+1} Freq (Hz)", color="#94a3b8", fontsize=10
        )
        ax_damp.set_ylabel(
            f"Mode {m_idx+1} Damp (%)", color="#94a3b8", fontsize=10
        )
        ax_freq.set_title(
            f"Mode {m_idx+1} -- Natural Frequency",
            color="#e2e8f0", fontsize=11, pad=6
        )

    # Day boundary lines
    cum = 0
    for day, n_segs in sorted(segments_per_day.items()):
        if cum > 0:
            for ax in axes:
                ax.axvline(cum, color="#475569", linewidth=1.2,
                           linestyle="--", alpha=0.8, zorder=1)
            axes[0].text(
                cum + 0.5, axes[0].get_ylim()[1],
                f"D{day}", color="#94a3b8", fontsize=8, va="top"
            )
        cum += n_segs

    # Legend
    handles = [
        mpatches.Patch(color=c, label=STATE_NAMES[s])
        for s, c in STATE_COLORS.items()
        if s in set(DAY_STATES.values())
    ]
    axes[0].legend(
        handles=handles, facecolor="#1e293b", edgecolor="#334155",
        labelcolor="#e2e8f0", fontsize=8, loc="upper right"
    )

    axes[-1].set_xlabel("Segment Index", color="#94a3b8", fontsize=10)
    fig.suptitle(
        "Harness Bridge -- Identified Modal Parameters Across All Days\n"
        "(SRIM output-only, stabilization + hierarchical clustering)",
        color="#e2e8f0", fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    out_path = out_dir / "modal_timeseries.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Modal time-series saved -> {out_path}")
    return out_path


def plot_anomaly_flags(
    detection_output: Dict,
    segments_per_day: Dict[int, int],
    true_labels: List[int],
    out_dir: Path,
    k_to_plot: int = 2,
    operators_to_plot: Optional[List[str]] = None,
):
    """
    Plot binary anomaly detection flags alongside ground-truth labels.
    """
    if operators_to_plot is None:
        operators_to_plot = ["AND", "OR", "FREQ_ONLY", "DAMP_ONLY"]

    flags_k = detection_output["anomaly_flags"].get(k_to_plot, {})
    n_ops = len(operators_to_plot)

    fig, axes = plt.subplots(
        n_ops + 1, 1,
        figsize=(18, 2.5 * (n_ops + 1)),
        facecolor="#0f172a",
        sharex=True
    )

    for ax in axes:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # Ground truth
    ax_gt = axes[0]
    seg_x = np.arange(len(true_labels))
    ax_gt.bar(seg_x, true_labels, color=["#22c55e" if l == 0 else "#ef4444"
                                         for l in true_labels],
              width=1.0, alpha=0.8)
    ax_gt.set_ylabel("True Label", color="#94a3b8", fontsize=9)
    ax_gt.set_yticks([0, 1])
    ax_gt.set_yticklabels(["Normal", "Changed"], color="#94a3b8", fontsize=8)
    ax_gt.set_title("Ground Truth (Concrete Pour State Changes)",
                    color="#e2e8f0", fontsize=10)

    # Predicted flags
    for i, op in enumerate(operators_to_plot):
        ax = axes[i + 1]
        preds = flags_k.get(op, [])
        n = len(preds)
        seg_x_p = np.arange(n)
        bar_colors = []
        heights = []
        for p in preds:
            if p is True:
                bar_colors.append("#f97316")
                heights.append(1)
            elif p is False:
                bar_colors.append("#1e40af")
                heights.append(0.5)
            else:
                bar_colors.append("#374151")
                heights.append(0.1)

        ax.bar(seg_x_p, heights, color=bar_colors, width=1.0, alpha=0.9)
        ax.set_ylabel(op, color="#94a3b8", fontsize=9)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(["Normal", "Anomaly"], color="#94a3b8", fontsize=8)

    # Day boundary lines
    cum = 0
    for day, n_segs in sorted(segments_per_day.items()):
        if cum > 0:
            for ax in axes:
                ax.axvline(cum, color="#475569", linewidth=1.2,
                           linestyle="--", alpha=0.8)
        cum += n_segs

    axes[-1].set_xlabel("Segment Index", color="#94a3b8", fontsize=10)
    fig.suptitle(
        f"Anomaly Flags -- k = {k_to_plot}sigma threshold\n"
        "Orange=Anomaly | Blue=Normal | Grey=Missing",
        color="#e2e8f0", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    out_path = out_dir / f"anomaly_flags_k{k_to_plot}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Anomaly flags saved -> {out_path}")
    return out_path


def plot_metrics_heatmap(
    all_metrics: Dict[int, Dict[str, Dict]],
    out_dir: Path,
):
    """
    Plot a heatmap of performance metrics across k-sigma and operators.
    """
    metric_cols = ["TPR", "TNR", "PPV", "NPV", "FPR", "FDR"]
    ks = sorted(all_metrics.keys())
    ops = list(list(all_metrics.values())[0].keys())

    # Collect data: rows = (k, op) combinations, cols = metrics
    row_labels = []
    data_matrix = []

    for k in ks:
        for op in ops:
            row_labels.append(f"k={k} {op}")
            row_vals = []
            for m in metric_cols:
                val = all_metrics[k][op].get(m)
                row_vals.append(val if val is not None else 0.0)
            data_matrix.append(row_vals)

    data_arr = np.array(data_matrix)

    fig, ax = plt.subplots(
        figsize=(12, max(4, len(row_labels) * 0.55)),
        facecolor="#0f172a"
    )
    ax.set_facecolor("#0f172a")

    im = ax.imshow(data_arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Metric value")

    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, color="#e2e8f0", fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, color="#e2e8f0", fontsize=9)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(metric_cols)):
            val = data_arr[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="black" if 0.35 < val < 0.75 else "white",
                fontsize=8
            )

    ax.set_title(
        "Performance Metrics Heatmap (Tran & Ozer 2020, Table 1)",
        color="#e2e8f0", fontsize=12, fontweight="bold", pad=10
    )
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    fig.tight_layout()
    out_path = out_dir / "metrics_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Metrics heatmap saved -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    """
    Full end-to-end pipeline execution.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  SRIM-Based Simultaneous Modal Parameter Anomaly Detection")
    print("  Harness Bridge Campaign -- Tran & Ozer (2020) Implementation")
    print("=" * 70 + "\n")

    # -- Configure pipeline components ------------------------------------

    segmenter_kwargs = dict(
        fs=args.fs,
        segment_length_s=args.seg_length,
        overlap=args.overlap,
        detrend=True,
        max_rows=None,
    )

    srim_kwargs = dict(
        fs=args.fs,
        i_factor=args.i_factor,
        max_order=args.max_order,
        min_freq=args.min_freq,
        max_freq=args.max_freq or args.fs / 2.0,
    )

    clusterer_kwargs = dict(
        lim_f=args.lim_f,
        lim_xi=args.lim_xi,
        lim_mac=args.lim_mac,
        mpc_threshold=args.mpc_threshold,
        cluster_threshold=args.cluster_threshold,
        max_damping=args.max_damping,
        min_cluster_size=args.min_cluster_size,
    )

    print(f"Segmenter : {args.seg_length}s windows, {args.overlap*100:.0f}% overlap")
    print(f"SRIM      : i_factor={args.i_factor}, orders 2..{args.max_order} (step 2)")
    print(f"Clusterer : lim_f={args.lim_f}, cluster_thresh={args.cluster_threshold}")
    print(f"Days      : {args.days}\n")

    # -- Process each day -------------------------------------------------

    all_modal_results: List[Dict] = []
    all_start_times: List[str] = []
    segments_per_day: Dict[int, int] = {}

    for day in args.days:
        modal_results, start_times, n_processed = process_day(
            day=day,
            data_dir=Path(args.data_dir),
            segmenter_kwargs=segmenter_kwargs,
            srim_kwargs=srim_kwargs,
            clusterer_kwargs=clusterer_kwargs,
            max_segments=args.n_segs,
            verbose=True,
        )

        all_modal_results.extend(modal_results)
        all_start_times.extend(start_times)
        segments_per_day[day] = n_processed

    total_segs = len(all_modal_results)
    identified = sum(1 for r in all_modal_results if r)
    print(f"\nTotal segments : {total_segs}")
    print(f"With >=1 mode  : {identified}  ({100*identified/max(total_segs,1):.1f}%)")

    # -- Stabilization diagram for first segment of Day 1 (diagnostic) ----
    if args.plot_stab:
        diag_day = args.days[0]
        print(f"\n[Diagnostic] Generating stabilization diagram for Day {diag_day}...")
        csv_path = find_csv(diag_day, Path(args.data_dir))
        if csv_path:
            seg_obj = DataSegmenter(**segmenter_kwargs)
            seg_obj.load_csv(csv_path)
            seg_list = seg_obj.get_segments()
            if seg_list:
                srim_diag = SRIMIdentifier(**srim_kwargs)
                clusterer_diag = ModalClusterer(**clusterer_kwargs)
                data_s0 = seg_list[0]["data"]
                poles_s0 = srim_diag.identify(data_s0)
                plot_stabilization_diagram(
                    poles_s0, diag_day, 0, clusterer_diag, OUTPUT_DIR, fs=args.fs
                )
                print(f"[Plot] Stabilization diagram saved to {OUTPUT_DIR}")

    # -- Modal parameter time-series plot ---------------------------------
    if args.plot_modal and total_segs > 0:
        plot_modal_timeseries(
            all_modal_results,
            all_start_times,
            segments_per_day,
            OUTPUT_DIR,
            mode_indices=tuple(range(args.n_modes_track)),
        )

    # -- Anomaly detection -------------------------------------------------
    print("\n[Anomaly Detection]")

    # Baseline: first n_baseline segments from Day 1
    day1_results = all_modal_results[:segments_per_day.get(1, 0)]
    n_baseline = min(args.n_baseline, len(day1_results))
    baseline_data = day1_results[:n_baseline]

    detector = AnomalyDetector(
        n_baseline_segments=n_baseline,
        k_sigma_values=(1, 2, 3),
        mode_indices=tuple(range(args.n_modes_track)),
        features=("freq", "damping"),
    )
    detector.fit_baseline(baseline_data)
    print("\n" + detector.baseline_summary())

    detection_output = detector.run(all_modal_results)

    # -- Build true labels -------------------------------------------------
    true_labels = MetricsEvaluator.build_labels_from_day_counts(
        segments_per_day,
        anomaly_days=[d for d in segments_per_day if d != 1],
    )

    # -- Anomaly flag plots ------------------------------------------------
    if args.plot_anomaly and total_segs > 0:
        for k_plot in [1, 2, 3]:
            plot_anomaly_flags(
                detection_output,
                segments_per_day,
                true_labels,
                OUTPUT_DIR,
                k_to_plot=k_plot,
            )

    # -- Metrics evaluation ------------------------------------------------
    print("\n[Metrics Evaluation]")
    evaluator = MetricsEvaluator(true_labels)
    all_metrics = evaluator.evaluate_all(detection_output["anomaly_flags"])

    print(MetricsEvaluator.format_table(all_metrics))

    if args.plot_metrics and total_segs > 0:
        plot_metrics_heatmap(all_metrics, OUTPUT_DIR)

    # -- Save results to JSON ----------------------------------------------
    if args.save_json:
        json_out = {
            "config": vars(args),
            "segments_per_day": segments_per_day,
            "n_total_segments": total_segs,
            "n_identified": identified,
            "baseline_stats": {
                k: list(v)
                for k, v in detector._baseline_stats.items()
            },
            "metrics": {
                str(k): {
                    op: {
                        m: (v if v is not None else "null")
                        for m, v in mets.items()
                    }
                    for op, mets in op_dict.items()
                }
                for k, op_dict in all_metrics.items()
            },
            # Compact modal time-series (freq and damping of tracked modes)
            "modal_timeseries": [
                {
                    "seg": i,
                    "time": all_start_times[i] if i < len(all_start_times) else "",
                    "modes": {
                        str(m): {
                            "freq": res[m]["freq"] if m in res else None,
                            "damping": res[m]["damping"] if m in res else None,
                        }
                        for m in range(args.n_modes_track)
                    }
                }
                for i, res in enumerate(all_modal_results)
            ],
        }
        json_path = OUTPUT_DIR / "pipeline_results.json"
        with open(json_path, "w") as fh:
            json.dump(json_out, fh, indent=2, default=str)
        print(f"\n[Output] Results saved -> {json_path}")

    print("\nOK Pipeline complete.  Outputs in:", OUTPUT_DIR)
    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SRIM-based SHM anomaly detection -- Harness Bridge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Directory containing 'dayN Clean.csv' files."
    )
    parser.add_argument(
        "--days",
        nargs="+",
        type=int,
        default=ALL_DAYS,
        help="Day numbers to process (e.g. --days 1 2 3)."
    )
    parser.add_argument(
        "--n-segs",
        type=int,
        default=None,
        help="Max segments per day (None = all). Set small for quick tests."
    )

    # Segmenter
    parser.add_argument("--fs",          type=float, default=128.0)
    parser.add_argument("--seg-length",  type=float, default=60.0,
                        help="Segment duration [s]. 60s gives good resolution for 17-18 Hz bridge modes.")
    parser.add_argument("--overlap",     type=float, default=0.5,
                        help="Fractional overlap [0, 1).")

    # SRIM
    parser.add_argument("--i-factor",   type=int,   default=10,
                        help="Block rows in Hankel matrix (i). Must satisfy i > n_max/m + 1.")
    parser.add_argument("--max-order",  type=int,   default=40,
                        help="Maximum system order n evaluated.")
    parser.add_argument("--min-freq",   type=float, default=0.2,
                        help="Lower frequency cutoff [Hz].")
    parser.add_argument("--max-freq",   type=float, default=25.0,
                        help="Upper frequency cutoff [Hz]. Default 25 Hz covers Harness Bridge structural modes (17-18 Hz).")

    # Clusterer
    parser.add_argument("--lim-f",            type=float, default=0.01)
    parser.add_argument("--lim-xi",           type=float, default=0.05)
    parser.add_argument("--lim-mac",          type=float, default=0.05)
    parser.add_argument("--mpc-threshold",    type=float, default=0.5)
    parser.add_argument("--cluster-threshold",type=float, default=0.5)
    parser.add_argument("--max-damping",      type=float, default=0.10)
    parser.add_argument("--min-cluster-size", type=int,   default=3)

    # Anomaly detection
    parser.add_argument("--n-baseline",    type=int, default=30,
                        help="Number of Day-1 segments used to fit the baseline Gaussian.")
    parser.add_argument("--n-modes-track", type=int, default=2,
                        help="Number of identified mode clusters to track.")

    # Outputs
    parser.add_argument("--plot-stab",    action="store_true", default=True)
    parser.add_argument("--plot-modal",   action="store_true", default=True)
    parser.add_argument("--plot-anomaly", action="store_true", default=True)
    parser.add_argument("--plot-metrics", action="store_true", default=True)
    parser.add_argument("--save-json",    action="store_true", default=True)
    parser.add_argument("--no-plots",     action="store_true", default=False,
                        help="Disable all plots (faster, for headless runs).")

    args = parser.parse_args()

    if args.no_plots:
        args.plot_stab = False
        args.plot_modal = False
        args.plot_anomaly = False
        args.plot_metrics = False

    return args


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
