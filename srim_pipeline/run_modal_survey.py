"""
run_modal_survey.py
===================
Main runner for the Harness Bridge Modal Survey.

Implements the Bridge 1 methodology from Tran & Ozer (2020) applied to the
Harness Bridge 10-day accelerometer campaign:

  1. SRIM system identification per 60-second segment
  2. Stabilization diagram → MPC clearance
  3. Day 1 consensus → 3 reference mode shapes (ModeTracker Phase 1)
  4. MAC-anchored mode assignment across all days (ModeTracker Phase 2)
  5. Validation metrics: MAC, MPC, CV, identification rate
  6. Anomaly detection: Gaussian k-sigma (k=1,2,3) × 4 operators
  7. All figures and LaTeX tables saved to output/modal_survey/

Usage (from the 'Harness Bridge' directory):
    python srim_pipeline/run_modal_survey.py
    python srim_pipeline/run_modal_survey.py --n-segs 30   # quick test
    python srim_pipeline/run_modal_survey.py --days 1 2 3
"""

import argparse
import io
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Force UTF-8 output regardless of terminal locale (fixes cp1254 / Turkish OS)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------- Local modules ---------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from data_segmenter   import DataSegmenter
from srim_identifier  import SRIMIdentifier
from modal_clusterer  import ModalClusterer
from mode_tracker     import ModeTracker
from modal_validator  import ModalValidator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR   = SCRIPT_DIR.parent / "Tests" / "Tests"
OUTPUT_DIR = SCRIPT_DIR / "output" / "modal_survey"

ALL_DAYS = list(range(1, 12))

# Structural state colour scheme (consistent with existing pipeline)
STATE_COLORS = {
    0: "#22c55e",   # green   — Day 1 baseline
    1: "#f97316",   # orange  — Post lower deck pour
    2: "#ef4444",   # red     — Upper deck pour activity
    3: "#a855f7",   # purple  — Post upper deck pour
}
STATE_NAMES = {
    0: "Baseline (Day 1)",
    1: "Post Lower Deck Pour",
    2: "Upper Deck Pour Activity",
    3: "Post Upper Deck Pour / Facade",
}
DAY_STATES = {
    1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 3, 10: 3, 11: 3
}

MODE_COLORS = [
    "#60a5fa", # blue
    "#f472b6", # pink
    "#34d399", # green
    "#fbbf24", # yellow/amber
    "#a78bfa", # purple
    "#f87171", # red
    "#2dd4bf", # teal
    "#a3e635", # lime
    "#38bdf8", # light blue
    "#fb923c", # orange
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dark_fig(figsize=(14, 5)):
    fig = plt.figure(figsize=figsize, facecolor="#0f172a")
    return fig


def _style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.grid(True, color="#1e293b", linewidth=0.5)
    if xlabel:
        ax.set_xlabel(xlabel, color="#94a3b8", fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color="#94a3b8", fontsize=10)
    if title:
        ax.set_title(title, color="#e2e8f0", fontsize=11, fontweight="bold", pad=8)


def find_csv(day: int, data_dir: Path) -> Optional[Path]:
    candidate = data_dir / f"day{day} Clean.csv"
    if candidate.exists():
        return candidate
    for p in data_dir.glob(f"day{day}*.csv"):
        if "Clean" in p.name:
            return p
    return None


# ---------------------------------------------------------------------------
# Step 1: SRIM identification for one day → raw poles per segment
# ---------------------------------------------------------------------------

def process_day_raw(
    day: int,
    data_dir: Path,
    segmenter_kwargs: dict,
    srim_kwargs: dict,
    max_segments: Optional[int] = None,
) -> Tuple[List[Dict], List[str], int]:
    """
    Load, segment, and identify poles (raw) for one day.

    Returns
    -------
    poles_list  : list of poles_by_order dicts (one per segment)
    start_times : list of timestamp strings
    n_segs      : number of segments processed
    """
    csv_path = find_csv(day, data_dir)
    if csv_path is None:
        print(f"[Day {day}] WARNING: CSV not found, skipping.")
        return [], [], 0

    seg = DataSegmenter(**segmenter_kwargs)
    seg.load_csv(csv_path)
    segments = seg.get_segments()
    if max_segments:
        segments = segments[:max_segments]

    srim = SRIMIdentifier(**srim_kwargs)
    poles_list:  List[Dict] = []
    start_times: List[str]  = []

    t0 = time.time()
    for seg_info in segments:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pbo = srim.identify(seg_info["data"])
        except Exception as exc:
            warnings.warn(f"[Day {day}] SRIM failed: {exc}")
            pbo = {}

        poles_list.append(pbo)
        start_times.append(seg_info["start_time"])

    elapsed = time.time() - t0
    print(
        f"[Day {day}]  {len(segments)} segs identified in {elapsed:.1f}s"
    )
    return poles_list, start_times, len(segments)


# ---------------------------------------------------------------------------
# Step 2 & 3: Reference building + tracking (run in main)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_stabilization_day1(
    poles_by_order: Dict,
    cleared_poles: List,
    tracker: ModeTracker,
    out_dir: Path,
    fs: float = 128.0,
):
    """Fig 1: Stabilization diagram for Day 1 segment 0, reference modes highlighted."""
    clusterer = ModalClusterer()
    stable    = clusterer.build_stabilization(poles_by_order)

    orders = sorted(poles_by_order.keys())

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0f172a")
    _style_ax(ax, "Frequency (Hz)", "System Order",
              "Stabilization Diagram — Day 1, Segment 0 (Reference Modes Highlighted)")

    # All poles
    for order, plist in poles_by_order.items():
        for p in plist:
            ax.scatter(p["freq"], order, c="#374151", s=6, alpha=0.3,
                       linewidths=0, zorder=1)

    # Stable poles
    for p in stable:
        ax.scatter(p["freq"], p["order"], c="#60a5fa", s=14, marker="+",
                   alpha=0.6, linewidths=1.0, zorder=2)

    # Cleared poles
    for p in cleared_poles:
        ax.scatter(p["freq"], p["order"], c="#fb923c", s=18, marker="o",
                   alpha=0.85, linewidths=0, zorder=3)

    # Reference mode vertical bands
    for i, (f_ref, col) in enumerate(zip(tracker.reference_frequencies, MODE_COLORS)):
        fwin = f_ref * tracker.freq_window
        ax.axvspan(f_ref - fwin, f_ref + fwin, alpha=0.10, color=col, zorder=0)
        ax.axvline(f_ref, color=col, lw=1.5, ls="--", zorder=4,
                   label=f"Mode {i+1} ref ({f_ref:.2f} Hz)")

    ax.set_xlim(0, min(fs / 2, 35))
    ax.set_ylim(min(orders) - 1, max(orders) + 1)

    legend_elems = [
        mpatches.Patch(color="#374151", label="All poles"),
        mpatches.Patch(color="#60a5fa", label="Stable (freq+MAC)"),
        mpatches.Patch(color="#fb923c", label="Cleared (MPC pass)"),
    ] + [
        mpatches.Patch(color=MODE_COLORS[i],
                       label=f"Mode {i+1} ref ({tracker.reference_frequencies[i]:.2f} Hz)")
        for i in range(tracker.n_modes)
    ]
    ax.legend(handles=legend_elems, facecolor="#1e293b", edgecolor="#334155",
              labelcolor="#e2e8f0", fontsize=8, loc="upper right")

    fig.tight_layout()
    out = out_dir / "Fig_02_Stabilization_with_PSD.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Fig 1] {out.name}")
    return out


def plot_freq_evolution(
    all_days_stats: Dict,
    days: List[int],
    n_modes: int,
    out_dir: Path,
):
    """Fig 2: Frequency evolution (median ± 1σ) per mode per day."""
    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5), facecolor="#0f172a")
    if n_modes == 1:
        axes = [axes]

    for m, ax in enumerate(axes):
        _style_ax(ax, "Day", "Frequency (Hz)", f"Mode {m+1} — Natural Frequency")

        xs, meds, stds, cols = [], [], [], []
        for d in days:
            s = all_days_stats.get(d, {}).get(m, {})
            fm = s.get("freq_median")
            fs = s.get("freq_std", 0.0) or 0.0
            if fm is None:
                continue
            xs.append(d)
            meds.append(fm)
            stds.append(fs)
            cols.append(STATE_COLORS[DAY_STATES.get(d, 0)])

        if xs:
            for xi, yi, si, ci in zip(xs, meds, stds, cols):
                ax.errorbar(xi, yi, yerr=si, fmt="o", color=ci, ms=8,
                            capsize=4, elinewidth=1.5, zorder=3)
            ax.plot(xs, meds, color="#94a3b8", lw=1, ls="--", alpha=0.4, zorder=2)

        ax.set_xticks(days)

    # Legend
    handles = [
        mpatches.Patch(color=c, label=STATE_NAMES[s])
        for s, c in STATE_COLORS.items()
        if s in set(DAY_STATES.values())
    ]
    axes[0].legend(handles=handles, facecolor="#1e293b", edgecolor="#334155",
                   labelcolor="#e2e8f0", fontsize=8)

    fig.suptitle("Harness Bridge — Natural Frequency Evolution Across 10 Days",
                 color="#e2e8f0", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "Fig_06_Freq_Evolution_10Days.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Fig 2] {out.name}")


def plot_damp_evolution(
    all_days_stats: Dict,
    days: List[int],
    n_modes: int,
    out_dir: Path,
):
    """Fig 3: Damping evolution per mode per day."""
    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5), facecolor="#0f172a")
    if n_modes == 1:
        axes = [axes]

    for m, ax in enumerate(axes):
        _style_ax(ax, "Day", "Damping Ratio (%)", f"Mode {m+1} — Damping Ratio")

        for d in days:
            s     = all_days_stats.get(d, {}).get(m, {})
            damps = s.get("dampings", [])
            if not damps:
                continue
            col = STATE_COLORS[DAY_STATES.get(d, 0)]
            vals_pct = [x * 100 for x in damps]
            ax.boxplot(
                vals_pct, positions=[d], widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor=col, alpha=0.7),
                medianprops=dict(color="white", lw=2),
                whiskerprops=dict(color="#94a3b8"),
                capprops=dict(color="#94a3b8"),
                flierprops=dict(marker=".", color=col, alpha=0.5),
            )

        ax.set_xticks(days)

    handles = [
        mpatches.Patch(color=c, label=STATE_NAMES[s])
        for s, c in STATE_COLORS.items()
        if s in set(DAY_STATES.values())
    ]
    axes[0].legend(handles=handles, facecolor="#1e293b", edgecolor="#334155",
                   labelcolor="#e2e8f0", fontsize=8)

    fig.suptitle("Harness Bridge — Damping Ratio Evolution Across 10 Days",
                 color="#e2e8f0", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "Fig_07_Damp_Evolution_10Days.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Fig 3] {out.name}")


def _plot_mac_heatmap(
    mat: np.ndarray,
    ax,
    xticklabels: List,
    yticklabels: List,
    title: str = "",
    cmap=None,
    annotate: bool = True,
):
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            "mac_cmap", ["#1e293b", "#3b82f6", "#22c55e"], N=256
        )
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, color="#e2e8f0", fontsize=8, rotation=45)
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels, color="#e2e8f0", fontsize=8)
    ax.tick_params(colors="#94a3b8")
    if title:
        ax.set_title(title, color="#e2e8f0", fontsize=10, fontweight="bold", pad=6)
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                txt = f"{v:.2f}" if not np.isnan(v) else "–"
                fc  = "white" if v < 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=fc, fontsize=7)
    return im


def plot_mac_within_day1(
    all_days_stats: Dict,
    validator: ModalValidator,
    out_dir: Path,
):
    """Fig 4: 3×3 within-day MAC for Day 1."""
    mac_mat = validator.within_day_mac(all_days_stats.get(1, {}))
    mode_labels = [f"M{i+1}" for i in range(validator.n_modes)]

    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    im = _plot_mac_heatmap(mac_mat, ax, mode_labels, mode_labels,
                           "Within-Day MAC — Day 1 Mode Orthogonality")
    plt.colorbar(im, ax=ax, label="MAC", fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="#94a3b8")

    fig.tight_layout()
    out = out_dir / "Fig_09_MAC_Within_Day1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Fig 4] {out.name}")


def plot_mac_cross_day(
    all_days_stats: Dict,
    validator: ModalValidator,
    days: List[int],
    mode_idx: int,
    fig_num: int,
    out_dir: Path,
):
    """Figs 5-7: Cross-day MAC for each mode."""
    mac_mat, day_list = validator.cross_day_mac(all_days_stats, mode_idx, days)
    day_labels = [f"D{d}" for d in day_list]

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    im = _plot_mac_heatmap(mac_mat, ax, day_labels, day_labels,
                           f"Cross-Day MAC — Mode {mode_idx+1}")
    plt.colorbar(im, ax=ax, label="MAC").ax.yaxis.set_tick_params(color="#94a3b8")

    fig.tight_layout()
    fname = f"Fig_{fig_num:02d}_MAC_CrossDay_M{mode_idx+1}.png"
    out   = out_dir / fname
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Fig {fig_num}] {out.name}")


def plot_mode_shapes(
    all_days_stats: Dict,
    tracker: ModeTracker,
    days: List[int],
    out_dir: Path,
    n_sensors: int = 7,
):
    """Fig 8: Mode shapes per mode, overlaid across day groups, interpolated with updated BCs (J5, J6 as rollers)."""
    from scipy.interpolate import CubicSpline
    
    n_modes = tracker.n_modes
    sensor_joints = [2, 3, 5, 6, 7, 9, 10]
    
    # Boundary conditions: J1, J11 fixed; J4, J8 roller
    fixed_joints = [1, 11]
    zero_joints = [4, 8]  # J5 and J6 free again
    all_supports = set(fixed_joints + zero_joints)
    
    spacings = [1.82, 1.82, 1.82, 2.0, 2.0, 2.0, 2.0, 1.25, 1.25, 1.25]
    joint_positions = [0.0]
    for s in spacings:
        joint_positions.append(joint_positions[-1] + s)
    joint_positions = np.array(joint_positions)
    
    # Group days by structural state
    state_groups = {
        "Baseline (Day 1)":       [d for d in days if DAY_STATES[d] == 0],
        "Post Lower Pour (D2–7)": [d for d in days if DAY_STATES[d] == 1],
        "Upper Pour (D8)":        [d for d in days if DAY_STATES[d] == 2],
        "Post Upper Pour (D9–11)": [d for d in days if DAY_STATES[d] == 3],
    }
    group_colors = list(STATE_COLORS.values())

    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5),
                             facecolor="#0f172a")
    if n_modes == 1:
        axes = [axes]

    x_fine = np.linspace(0, joint_positions[-1], 200)

    for m, ax in enumerate(axes):
        _style_ax(ax, "Bridge Span (m)", "Normalised Deflection",
                  f"Mode {m+1} Shape — Day Groups")

        # Helper to interpolate shape
        def get_interpolated_shape(shape_vals):
            # shape_vals has length 7 (corresponding to sensor_joints)
            known_x = []
            known_y = []
            for j in range(1, 12):
                pos = joint_positions[j - 1]
                if j in all_supports:
                    known_x.append(pos)
                    known_y.append(0.0)
                elif j in sensor_joints:
                    sensor_idx = sensor_joints.index(j)
                    known_x.append(pos)
                    known_y.append(shape_vals[sensor_idx])
            
            # Sort by position
            sorted_idx = np.argsort(known_x)
            kx = np.array(known_x)[sorted_idx]
            ky = np.array(known_y)[sorted_idx]
            
            cs = CubicSpline(kx, ky, bc_type='natural')
            y_fine = cs(x_fine)
            # Normalise
            norm = np.max(np.abs(y_fine))
            if norm > 0:
                y_fine /= norm
            return y_fine

        # Plot Each Day
        # Create a colormap for the days
        cmap = plt.get_cmap('tab10')
        for d_idx, d in enumerate(days):
            s = all_days_stats.get(d, {}).get(m, {})
            ms = s.get("mean_shape", None)
            if ms is not None and len(ms) == n_sensors:
                # Norm first to avoid large scaling mismatches
                norm_s = ms / (np.max(np.abs(ms)) + 1e-30)
                y_fine = get_interpolated_shape(norm_s)
                
                # Day 1 is dashed, others are solid
                ls = "--" if d == 1 else "-"
                lw = 2.5 if d == 1 else 1.5
                alpha = 1.0 if d == 1 else 0.8
                col = cmap(d_idx % 10)
                
                ax.plot(x_fine, y_fine, color=col, lw=lw, ls=ls,
                        label=f"Day {d}", alpha=alpha, zorder=3)

        # Draw supports on the plot baseline
        for j in all_supports:
            pos = joint_positions[j - 1]
            ax.plot(pos, 0, marker="^", color="#475569", markersize=8, zorder=6)
            ax.text(pos, -0.15, f"J{j}", ha='center', va='top', fontsize=7, color="#64748b")
            
        # Draw sensor locations
        for j in sensor_joints:
            if j not in all_supports:
                pos = joint_positions[j - 1]
                ax.axvline(pos, color="#1e293b", lw=0.8, ls=":", zorder=1)
                ax.text(pos, 1.05, f"S{sensor_joints.index(j)+1}", ha='center', va='bottom', fontsize=7, color="#64748b")

        ax.set_xlim(-0.5, joint_positions[-1] + 0.5)
        ax.set_ylim(-1.2, 1.2)
        ax.axhline(0, color="#334155", lw=1.0, zorder=1)
        ax.legend(facecolor="#1e293b", edgecolor="#334155",
                  labelcolor="#e2e8f0", fontsize=7, loc="lower right", ncol=2)

    fig.suptitle("Identified Mode Shapes Across All Days (BCs: J5, J6 Rollers)",
                 color="#e2e8f0", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "Fig_05_Mode_Shapes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Fig 8] {out.name}")
    
    # -------------------------------------------------------------------------
    # Generate Interactive Plotly HTML
    # -------------------------------------------------------------------------
    plotly_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    fig_html = make_subplots(rows=1, cols=n_modes, subplot_titles=[f"Mode {m+1} Shape" for m in range(n_modes)])
    
    for m in range(n_modes):
        col_idx = m + 1
        
        # Helper to interpolate shape for plotly
        def get_interpolated_shape(shape_vals):
            known_x, known_y = [], []
            for j in range(1, 12):
                pos = joint_positions[j - 1]
                if j in all_supports:
                    known_x.append(pos)
                    known_y.append(0.0)
                elif j in sensor_joints:
                    sensor_idx = sensor_joints.index(j)
                    known_x.append(pos)
                    known_y.append(shape_vals[sensor_idx])
            
            sorted_idx = np.argsort(known_x)
            kx = np.array(known_x)[sorted_idx]
            ky = np.array(known_y)[sorted_idx]
            
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(kx, ky, bc_type='natural')
            y_fine = cs(x_fine)
            norm = np.max(np.abs(y_fine))
            if norm > 0:
                y_fine /= norm
            return y_fine
            
        for d_idx, d in enumerate(days):
            s = all_days_stats.get(d, {}).get(m, {})
            ms = s.get("mean_shape", None)
            if ms is not None and len(ms) == n_sensors:
                norm_s = ms / (np.max(np.abs(ms)) + 1e-30)
                y_fine = get_interpolated_shape(norm_s)
                
                ls = "dash" if d == 1 else "solid"
                lw = 3 if d == 1 else 2
                
                fig_html.add_trace(go.Scatter(
                    x=x_fine, y=y_fine, mode='lines',
                    name=f"Day {d}",
                    line=dict(color=plotly_colors[d_idx % len(plotly_colors)], width=lw, dash=ls),
                    showlegend=True if m == 0 else False,
                    legendgroup=f"Day {d}"
                ), row=1, col=col_idx)

        # Draw zero line
        fig_html.add_hline(y=0, line_width=1, line_color="#334155", row=1, col=col_idx)
        
        # Draw supports
        for j in all_supports:
            pos = joint_positions[j - 1]
            fig_html.add_trace(go.Scatter(
                x=[pos], y=[0], mode='markers+text',
                marker=dict(symbol='triangle-up', size=10, color='#475569'),
                text=[f"J{j}"], textposition="bottom center",
                showlegend=False, hoverinfo='skip'
            ), row=1, col=col_idx)

        # Update axes
        fig_html.update_xaxes(title_text="Bridge Span (m)", range=[-0.5, joint_positions[-1] + 0.5], row=1, col=col_idx)
        fig_html.update_yaxes(title_text="Normalised Deflection" if m == 0 else "", range=[-1.2, 1.2], row=1, col=col_idx)

    fig_html.update_layout(
        title="Interactive Mode Shapes Across All Days (Click legend to toggle)",
        template="plotly_dark",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        height=500
    )
    
    out_html = out_dir / "Fig_05_Mode_Shapes.html"
    fig_html.write_html(str(out_html))
    print(f"  [Fig 8 HTML] {out_html.name}")


def plot_identification_rate(
    all_days_stats: Dict,
    days: List[int],
    n_modes: int,
    out_dir: Path,
):
    """Fig 9: Stacked bar of identification rate per mode per day."""
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0f172a")
    _style_ax(ax, "Day", "Identification Rate (%)",
              "Mode Identification Rate per Day")

    x     = np.arange(len(days))
    width = 0.25
    offsets = np.linspace(-(n_modes - 1) * width / 2,
                           (n_modes - 1) * width / 2, n_modes)

    for m in range(n_modes):
        rates = []
        for d in days:
            ir = all_days_stats.get(d, {}).get(m, {}).get("id_rate", 0.0) * 100
            rates.append(ir)
        ax.bar(x + offsets[m], rates, width, color=MODE_COLORS[m],
               alpha=0.85, label=f"Mode {m+1}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"D{d}" for d in days], color="#94a3b8")
    ax.set_ylim(0, 110)
    ax.axhline(70, color="#ef4444", ls="--", lw=1, alpha=0.6, label="70% threshold")
    ax.legend(facecolor="#1e293b", edgecolor="#334155",
              labelcolor="#e2e8f0", fontsize=9)

    fig.tight_layout()
    out = out_dir / "Fig_11_Identification_Rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Fig 9] {out.name}")


def plot_freq_scatter_all(
    per_day_results: Dict[int, List],
    days: List[int],
    segments_per_day: Dict[int, int],
    n_modes: int,
    out_dir: Path,
):
    """Fig 10: Scatter of all individual segment frequencies."""
    fig, axes = plt.subplots(n_modes, 1, figsize=(18, 4 * n_modes),
                             facecolor="#0f172a", sharex=True)
    if n_modes == 1:
        axes = [axes]

    # Build global segment index
    seg_offset = 0
    day_boundaries = []
    for d in days:
        day_boundaries.append(seg_offset)
        seg_offset += segments_per_day.get(d, 0)
    day_boundaries.append(seg_offset)

    for m, ax in enumerate(axes):
        _style_ax(ax, "", f"Mode {m+1} Freq (Hz)",
                  f"Mode {m+1} — Per-Segment Frequency")

        seg_global = 0
        for d in days:
            day_results = per_day_results.get(d, [])
            col = STATE_COLORS[DAY_STATES.get(d, 0)]
            xs, ys = [], []
            for seg_res in day_results:
                cr = seg_res.get(m, None)
                if cr is not None and cr.get("freq") is not None:
                    xs.append(seg_global)
                    ys.append(cr["freq"])
                seg_global += 1

            if xs:
                ax.scatter(xs, ys, c=col, s=8, alpha=0.75, linewidths=0, zorder=2)

        # Day boundary lines
        seg_offset2 = 0
        for i, d in enumerate(days):
            if i > 0:
                ax.axvline(seg_offset2, color="#475569", lw=1, ls="--", alpha=0.7)
            seg_offset2 += segments_per_day.get(d, 0)

    # Day labels on top axis
    seg_offset3 = 0
    for d in days:
        n_s = segments_per_day.get(d, 0)
        axes[0].text(seg_offset3 + n_s / 2, axes[0].get_ylim()[1],
                     f"D{d}", color="#94a3b8", fontsize=8,
                     ha="center", va="bottom")
        seg_offset3 += n_s

    axes[-1].set_xlabel("Segment Index", color="#94a3b8", fontsize=10)
    handles = [
        mpatches.Patch(color=c, label=STATE_NAMES[s])
        for s, c in STATE_COLORS.items() if s in set(DAY_STATES.values())
    ]
    axes[0].legend(handles=handles, facecolor="#1e293b", edgecolor="#334155",
                   labelcolor="#e2e8f0", fontsize=8)
    fig.suptitle("Harness Bridge — Per-Segment Modal Frequencies (All Days)",
                 color="#e2e8f0", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "Fig_08_Freq_Scatter_Segments.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Fig 10] {out.name}")


def plot_anomaly_detection(
    detection_metrics: Dict,
    out_dir: Path,
):
    """Fig 11: Anomaly detection metrics heatmap (same as Bridge 1 figure)."""
    metric_cols = ["TPR", "TNR", "PPV", "NPV", "FPR", "FDR"]
    ops         = ["AND", "OR", "FREQ_ONLY", "DAMP_ONLY"]
    ks          = sorted(detection_metrics.keys())

    row_labels  = []
    data_matrix = []

    for k in ks:
        for op in ops:
            row_labels.append(f"k={k} {op}")
            mets = detection_metrics.get(k, {}).get(op, {})
            row_vals = [mets.get(c) or 0.0 for c in metric_cols]
            data_matrix.append(row_vals)

    data_arr = np.array(data_matrix)

    fig, ax = plt.subplots(
        figsize=(12, max(4, len(row_labels) * 0.55)),
        facecolor="#0f172a"
    )
    ax.set_facecolor("#0f172a")
    im = ax.imshow(data_arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Metric value").ax.yaxis.set_tick_params(color="#94a3b8")

    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, color="#e2e8f0", fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, color="#e2e8f0", fontsize=9)
    ax.tick_params(colors="#94a3b8")

    for i in range(len(row_labels)):
        for j in range(len(metric_cols)):
            v  = data_arr[i, j]
            fc = "black" if 0.35 < v < 0.75 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=fc, fontsize=8)

    ax.set_title(
        "Anomaly Detection Performance (Tran & Ozer 2020, Table 1 — Bridge 1 Methodology)",
        color="#e2e8f0", fontsize=11, fontweight="bold", pad=10
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    fig.tight_layout()
    out = out_dir / "Fig_12_Anomaly_Detection.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Fig 11] {out.name}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_survey(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Harness Bridge — Modal Survey")
    print("  3-Mode Tracking + Anomaly Detection (Tran & Ozer 2020 Bridge 1)")
    print("=" * 70 + "\n")

    # -- Pipeline parameters -----------------------------------------------
    segmenter_kwargs = dict(
        fs=args.fs,
        segment_length_s=args.seg_length,
        overlap=args.overlap,
        detrend=True,
    )
    srim_kwargs = dict(
        fs=args.fs,
        i_factor=args.i_factor,
        max_order=args.max_order,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
    )
    clusterer = ModalClusterer(
        lim_f=args.lim_f,
        lim_mac=args.lim_mac,
        mpc_threshold=args.mpc_threshold,
        max_damping=args.max_damping,
        min_cluster_size=args.min_cluster_size,
        cluster_threshold=args.cluster_thresh,
    )
    tracker = ModeTracker(
        n_modes=3,
        mac_min=args.mac_min,
        freq_window=args.freq_window,
        global_cluster_threshold=args.global_cluster_thresh,
        mpc_threshold=args.mpc_threshold,
        max_damping=args.max_damping,
    )
    # Validator is initialized after tracker modes are established in Step 2

    print(f"Segment  : {args.seg_length}s, {args.overlap*100:.0f}% overlap")
    print(f"SRIM     : i={args.i_factor}, orders 2..{args.max_order}")
    print(f"Tracking : mac_min={args.mac_min}, freq_window={args.freq_window*100:.0f}%")
    print(f"Days     : {args.days}\n")

    # =================================================================== #
    #  Step 1: SRIM identification for all days                           #
    # =================================================================== #
    print("─" * 50)
    print("Step 1: SRIM Identification")
    print("─" * 50)

    all_poles:      Dict[int, List[Dict]] = {}  # day → list[poles_by_order]
    all_times:      Dict[int, List[str]]  = {}
    segments_per_day: Dict[int, int]      = {}

    for day in args.days:
        poles_list, start_times, n_segs = process_day_raw(
            day, Path(args.data_dir), segmenter_kwargs, srim_kwargs,
            max_segments=args.n_segs,
        )
        all_poles[day]   = poles_list
        all_times[day]   = start_times
        segments_per_day[day] = n_segs

    # =================================================================== #
    #  Step 2: Build reference mode shapes from Day 1                     #
    # =================================================================== #
    print("\n" + "─" * 50)
    print("Step 2: Day 1 Reference Mode Shapes")
    print("─" * 50)

    day1_poles = all_poles.get(1, [])
    if day1_poles:
        stable0  = clusterer.build_stabilization(day1_poles[0])
        cleared0 = clusterer.clear_diagram(stable0)
        raw_clusters = clusterer.cluster_poles(cleared0)
        clean_clusters = clusterer.remove_outliers(raw_clusters)
        cluster_modes = sorted(
            clean_clusters.items(),
            key=lambda kv: np.median([p["freq"] for p in kv[1]]),
        )
        # Use k-sigma statistical verification to merge indistinct modes
        final_modes, clean_clusters = clusterer.verify_and_merge_modes(
            cluster_modes,
            k=args.merge_k,
            gate=args.merge_gate,
            mac_threshold=args.merge_mac
        )
        
        # Inject explicit methodology modes
        tracker.set_explicit_references(final_modes)
        
        # Save for Fig 1
        meth_data_dict = {
            'poles_by_order': day1_poles[0],
            'all_poles': [p for plist in day1_poles[0].values() for p in plist],
            'stable_poles': stable0,
            'cleared_poles': cleared0,
            'clean_clusters': clean_clusters,
            'final_modes': final_modes
        }
    else:
        stable0 = []
        cleared0 = []
        meth_data_dict = {}

    validator = ModalValidator(n_modes=tracker.n_modes, n_days=len(args.days))

    # =================================================================== #
    #  Step 3: Track modes across all days                                #
    # =================================================================== #
    print("\n" + "─" * 50)
    print("Step 3: MAC-Anchored Mode Tracking")
    print("─" * 50)

    per_day_results: Dict[int, List] = {}   # day → list[TrackResult]
    all_track_results: List          = []   # flat list (all days concatenated)

    for day in args.days:
        day_track = tracker.track_all_days(all_poles[day])
        all_track_results.extend(day_track)

    # Cross-Day Selection: We keep all tracking modes since we injected explicit methodology modes.
    # We don't truncate, we just ensure they are sorted by frequency (which they already are).
    # all_track_results = tracker.select_best_modes(all_track_results, top_n=tracker.n_modes)

    # Re-slice all_track_results into per_day_results
    idx = 0
    for day in args.days:
        n_segs = segments_per_day[day]
        day_track = all_track_results[idx : idx + n_segs]
        per_day_results[day] = day_track
        idx += n_segs

        n_found = [
            sum(1 for tr in day_track if tr.get(m) is not None)
            for m in range(tracker.n_modes)
        ]
        
        # Build dynamic string for all modes
        found_str = " ".join([f"M{m+1}={n_found[m]:3d}" for m in range(tracker.n_modes)])
        print(
            f"  Day {day:2d}: {segments_per_day[day]:3d} segs | {found_str}"
        )

    # =================================================================== #
    #  Step 4: Validation metrics                                         #
    # =================================================================== #
    print("\n" + "─" * 50)
    print("Step 4: Validation Metrics")
    print("─" * 50)

    all_days_stats = validator.all_days_statistics(per_day_results)
    shifts         = validator.frequency_shifts(all_days_stats)

    # Print summary
    print(f"\n{'Day':>4} | {'Mode':>5} | {'f (Hz)':>10} | {'ξ (%)':>8} | {'MPC':>6} | {'IR (%)':>7}")
    print("-" * 55)
    for d in args.days:
        for m in range(tracker.n_modes):
            s   = all_days_stats.get(d, {}).get(m, {})
            fm  = s.get("freq_median")
            dm  = s.get("damp_median")
            mpc = s.get("mpc_mean")
            ir  = s.get("id_rate", 0.0) * 100
            if fm:
                print(
                    f"{d:>4} | M{m+1:>4} | {fm:>10.3f} | "
                    f"{(dm or 0)*100:>8.3f} | "
                    f"{(mpc or 0):>6.3f} | {ir:>7.1f}"
                )

    # =================================================================== #
    #  Step 5: Anomaly detection (Bridge 1 methodology)                   #
    # =================================================================== #
    print("\n" + "─" * 50)
    print("Step 5: Anomaly Detection")
    print("─" * 50)

    # Build true labels: Day 1 = 0, all others = 1
    true_labels = []
    for d in args.days:
        lbl = 0 if d == 1 else 1
        true_labels.extend([lbl] * segments_per_day[d])

    # Fit Gaussian on Day 1
    baseline_stats = validator.fit_gaussian_baseline(per_day_results.get(1, []))
    print(f"\n  Baseline features fitted: {list(baseline_stats.keys())}")

    # Detect anomalies across all segments
    detection_out    = validator.detect_anomalies(all_track_results, baseline_stats)
    detection_metrics = validator.compute_detection_metrics(
        detection_out["anomaly_flags"], true_labels
    )

    # Print summary table
    print(f"\n{'k':>3} | {'Operator':>10} | {'TPR':>6} | {'TNR':>6} | {'PPV':>6} | {'FPR':>6} | {'FDR':>6}")
    print("-" * 55)
    for k in sorted(detection_metrics.keys()):
        for op, mets in detection_metrics[k].items():
            print(
                f"{k:>3} | {op:>10} | "
                f"{(mets.get('TPR') or 0):.3f} | "
                f"{(mets.get('TNR') or 0):.3f} | "
                f"{(mets.get('PPV') or 0):.3f} | "
                f"{(mets.get('FPR') or 0):.3f} | "
                f"{(mets.get('FDR') or 0):.3f}"
            )

    # =================================================================== #
    #  Step 6: Figures                                                     #
    # =================================================================== #
    print("\n" + "─" * 50)
    print("Step 6: Generating Figures")
    print("─" * 50)

    days = args.days

    from paper_figures import plot_fig01_methodology_4panel, plot_fig02_to_04_multiday_clusters
    import pandas as pd
    
    # 1. Methodology Fig 01
    plot_fig01_methodology_4panel(meth_data_dict, OUTPUT_DIR)

    # 2. Multi-day Figs 02-04
    print("Preparing representative segments for multi-day methodology figures...")
    target_days = []
    for d in days:
        label = f"Day {d} ({STATE_NAMES.get(DAY_STATES.get(d, 0), 'Unknown')})"
        target_days.append((d, label))

    multiday_data_dict = {}
    for day_num, label in target_days:
        if day_num in all_poles and len(all_poles[day_num]) > 0:
            csv_path = None
            for p in Path(args.data_dir).glob(f"day{day_num}*.csv"):
                csv_path = p
                break
            if csv_path:
                from data_segmenter import DataSegmenter
                seg = DataSegmenter(fs=args.fs, segment_length_s=args.seg_length, overlap=args.overlap)
                seg.load_csv(csv_path)
                segments = seg.get_segments()
                if segments:
                    multiday_data_dict[label] = {
                        'clusterer': clusterer,
                        'poles_by_order': all_poles[day_num][0],
                        'data': segments[0]['data']
                    }
    plot_fig02_to_04_multiday_clusters(multiday_data_dict, args.fs, OUTPUT_DIR)

    # 3. Evolution and tracking Figs 05-12
    plot_mode_shapes(all_days_stats, tracker, days, OUTPUT_DIR)
    plot_freq_evolution(all_days_stats, days, tracker.n_modes, OUTPUT_DIR)
    plot_damp_evolution(all_days_stats, days, tracker.n_modes, OUTPUT_DIR)
    plot_freq_scatter_all(per_day_results, days, segments_per_day, tracker.n_modes, OUTPUT_DIR)
    plot_mac_within_day1(all_days_stats, validator, OUTPUT_DIR)

    for m in range(tracker.n_modes):
        plot_mac_cross_day(all_days_stats, validator, days, m, 10, OUTPUT_DIR)

    plot_identification_rate(all_days_stats, days, tracker.n_modes, OUTPUT_DIR)
    plot_anomaly_detection(detection_metrics, OUTPUT_DIR)

    # =================================================================== #
    #  Step 7: LaTeX tables                                               #
    # =================================================================== #
    print("\n" + "─" * 50)
    print("Step 7: LaTeX Tables")
    print("─" * 50)

    # Tab 1: modal summary
    tex1 = validator.latex_modal_summary(all_days_stats)
    (OUTPUT_DIR / "tab1_modal_summary.tex").write_text(tex1, encoding="utf-8")
    print("  [Tab 1] tab1_modal_summary.tex")

    # Tab 2: frequency shifts
    tex2 = validator.latex_freq_shift(shifts, days)
    (OUTPUT_DIR / "tab2_freq_shift.tex").write_text(tex2, encoding="utf-8")
    print("  [Tab 2] tab2_freq_shift.tex")

    # Tab 3: within-day MAC matrices
    all_days_mac = {
        d: validator.within_day_mac(all_days_stats.get(d, {}))
        for d in days
    }
    tex3 = validator.latex_mac_within(all_days_mac, days)
    (OUTPUT_DIR / "tab3_mac_within.tex").write_text(tex3, encoding="utf-8")
    print("  [Tab 3] tab3_mac_within.tex")

    # Tab 4: validation metrics
    tex4 = validator.latex_validation_metrics(all_days_stats)
    (OUTPUT_DIR / "tab4_validation_metrics.tex").write_text(tex4, encoding="utf-8")
    print("  [Tab 4] tab4_validation_metrics.tex")

    # Tab 5: anomaly detection
    tex5 = validator.latex_detection_metrics(detection_metrics)
    (OUTPUT_DIR / "tab5_detection_metrics.tex").write_text(tex5, encoding="utf-8")
    print("  [Tab 5] tab5_detection_metrics.tex")

    # =================================================================== #
    #  Step 8: Save JSON                                                   #
    # =================================================================== #

    def _ser(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.float32, np.float64)):
            return float(v)
        if isinstance(v, (np.int32, np.int64)):
            return int(v)
        return str(v)

    summary = {
        "config": vars(args),
        "segments_per_day": segments_per_day,
        "reference_frequencies": tracker.reference_frequencies,
        "per_day_modal_summary": {
            str(d): {
                str(m): {
                    k2: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k2, v in s.items()
                    if k2 not in ("freqs", "dampings", "shapes",
                                  "mpc_vals", "mac_to_ref_vals")
                }
                for m, s in all_days_stats.get(d, {}).items()
            }
            for d in days
        },
        "freq_shifts_pct": {
            str(m): {str(d): v for d, v in shifts[m].items()}
            for m in range(tracker.n_modes)
        },
        "detection_metrics": {
            str(k): {
                op: {col: v for col, v in mets.items()}
                for op, mets in op_dict.items()
            }
            for k, op_dict in detection_metrics.items()
        },
    }
    json_path = OUTPUT_DIR / "modal_survey_results.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=_ser)
    print(f"\n  Results saved → {json_path}")

    print(f"\n✓ Modal survey complete.  Outputs in: {OUTPUT_DIR}\n")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Harness Bridge Modal Survey — 3-mode tracking & validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",   default=str(DATA_DIR))
    p.add_argument("--days",       nargs="+", type=int, default=ALL_DAYS)
    p.add_argument("--n-segs",     type=int,   default=None,
                   help="Max segments per day (None=all)")

    # Segmenter
    p.add_argument("--fs",         type=float, default=128.0)
    p.add_argument("--seg-length", type=float, default=60.0,
                   help="Segment duration [s]")
    p.add_argument("--overlap",    type=float, default=0.5)

    # SRIM
    p.add_argument("--i-factor",   type=int,   default=10)
    p.add_argument("--max-order",  type=int,   default=80,
                   help="Higher order captures more modes")
    p.add_argument("--min-freq",   type=float, default=0.5)
    p.add_argument("--max-freq",   type=float, default=64.0,
                   help="Upper frequency bound [Hz] — Default 64.0 (Nyquist)")

    # Clusterer
    p.add_argument("--lim-f",      type=float, default=0.02)
    p.add_argument("--lim-mac", type=float, default=0.05, help="MAC stability limit (1 - lim)")
    p.add_argument("--mpc-threshold", type=float, default=0.50, help="MPC cutoff for Type 3")
    p.add_argument("--cluster-thresh", type=float, default=0.15, help="Cutoff for hierarchical clustering")
    p.add_argument("--min-cluster-size", type=int, default=5, help="Min poles per cluster")
    p.add_argument("--max-damping", type=float, default=0.10, help="Maximum physical damping ratio")
    
    p.add_argument("--merge-k", type=float, default=3.0, help="k-sigma boundary for mode merging")
    p.add_argument("--merge-gate", type=str, default="OR", choices=["AND", "OR"], help="Logic gate for distinctness")
    p.add_argument("--merge-mac", type=float, default=0.85, help="MAC threshold for distinctness")

    # Tracker
    p.add_argument("--mac-min",          type=float, default=0.70)
    p.add_argument("--freq-window",      type=float, default=0.15,
                   help="±15% frequency window for MAC assignment")
    p.add_argument("--global-cluster-thresh", type=float, default=0.25,
                   help="Distance cut for Day-1 global clustering")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_survey(args)
