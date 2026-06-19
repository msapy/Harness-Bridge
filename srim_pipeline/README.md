# SRIM SHM Pipeline — Harness Bridge

A fully automated Python implementation of the **simultaneous modal parameter anomaly detection** methodology described in:

> **Tran, T.T.X. & Ozer, E. (2020)**. *Automated and Model-Free Bridge Damage Indicators with Simultaneous Multiparameter Modal Anomaly Detection.* Sensors, 20(17), 4752. https://doi.org/10.3390/s20174752

Applied to the **Harness Bridge** 10-day accelerometer campaign (May–June 2026) to detect structural state changes caused by concrete pours.

---

## File Structure

```
srim_pipeline/
├── __init__.py              # Package exports
├── data_segmenter.py        # DataSegmenter class
├── srim_identifier.py       # SRIMIdentifier class (output-only SRIM)
├── modal_clusterer.py       # ModalClusterer class
├── anomaly_detector.py      # AnomalyDetector class
├── metrics_evaluator.py     # MetricsEvaluator class
├── srim_shm_pipeline.py     # Main runner script
├── README.md                # This file
└── output/                  # Generated plots and JSON (created at runtime)
```

---

## Dataset

| Day | Date          | Duration | Notes |
|-----|---------------|----------|-------|
| 1   | 20 May 2026   | 72 min   | **Baseline** (pre-lower-deck pour) |
| 2   | 21 May 2026   | 52 min   | Lower deck concrete poured → state change |
| 3   | 22 May 2026   | 12 min   | Post lower deck pour |
| 4   | 25 May 2026   | 14 min   | Post lower deck pour |
| 5   | 27 May 2026   | 13 min   | Post lower deck pour |
| 6   | 29 May 2026   | 60 min   | Post lower deck pour |
| 7   |  2 Jun 2026   | 38 min   | Post lower deck pour (J7 interpolated) |
| 8   |  3 Jun 2026   | 3h 34m   | **Upper deck pour activity** (mixed state) |
| 9   |  8 Jun 2026   | 25 min   | Post upper deck pour |
| 10  | 10 Jun 2026   | 55 min   | Post upper deck pour |

- **7 sensors** (29124–29130 `:ch3`) at **128 Hz**
- Concrete pours add mass → expected **frequency decrease** and potential damping changes

---

## Quick Start

```bash
# From the 'Harness Bridge' directory:

# Quick smoke-test on Day 1 (20 segments, ~40 seconds)
python srim_pipeline/srim_shm_pipeline.py --days 1 --n-segs 20

# Days 1 and 2 only (baseline + first state change)
python srim_pipeline/srim_shm_pipeline.py --days 1 2

# Full 10-day run (may take 20–60 minutes depending on hardware)
python srim_pipeline/srim_shm_pipeline.py

# Shorter 30s windows for better frequency resolution
python srim_pipeline/srim_shm_pipeline.py --seg-length 30 --n-baseline 20
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seg-length` | 20.0 s | Window duration (paper: 20 s for Bridge 2) |
| `--overlap` | 0.5 | Fractional overlap between windows |
| `--i-factor` | 10 | Block rows in SRIM Hankel matrix |
| `--max-order` | 40 | Maximum system order for stabilization diagram |
| `--lim-f` | 0.01 | Frequency stability limit (1%) |
| `--lim-mac` | 0.05 | MAC stability limit (5% → MAC > 0.95) |
| `--mpc-threshold` | 0.5 | Minimum MPC value to keep a pole |
| `--cluster-threshold` | 0.5 | Euclidean distance for hierarchical clustering |
| `--max-damping` | 0.10 | Hard upper bound on damping ratio (10%) |
| `--n-baseline` | 30 | Number of Day-1 segments for baseline Gaussian |

---

## Algorithm Summary

```
For each time segment:
  1. SRIMIdentifier.identify(data)
     └── Build block Hankel Y_i  (output-only, Eq. 4)
     └── R_yy = (1/j) Y_i Y_i^T  (Eq. 5)
     └── SVD → observability matrix O  (Eq. 13)
     └── For each order n: extract A, C → eigendecompose → (f, ξ, φ)

  2. ModalClusterer.process(poles_by_order)
     └── Build stabilization diagram (Eqs. 19, 21) — type-3 stable poles
     └── Clear: drop ξ ≤ 0, apply MPC filter (Eqs. 22–26)
     └── Hierarchical clustering (single-linkage, threshold 0.5)
     └── MAD outlier removal (Eq. 27), drop ξ > 10%
     └── Aggregate: median f and ξ per cluster

For the full time-series:
  3. AnomalyDetector.fit_baseline(Day_1_results)
     └── Fit Gaussian μ_j, σ²_j per feature (Eqs. 29–30)
     └── Compute thresholds ε_k = p(μ − kσ) for k = 1, 2, 3

  4. AnomalyDetector.run(all_results)
     └── p(x_j) for each new segment feature (Eq. 28/31)
     └── Flag if p(x_j) < ε_k
     └── AND / OR / FREQ_ONLY / DAMP_ONLY operators

  5. MetricsEvaluator.evaluate_all(anomaly_flags)
     └── TPR, TNR, PPV, NPV, FPR, FDR, FNR, FOR (Table 1)
```

---

## Outputs

After a run, `srim_pipeline/output/` will contain:

| File | Description |
|------|-------------|
| `stabilization_day1_seg0.png` | Stabilization diagram for segment 0 of Day 1 |
| `modal_timeseries.png` | Frequency and damping vs. time across all days |
| `anomaly_flags_k1.png` | Binary flags at 1σ threshold |
| `anomaly_flags_k2.png` | Binary flags at 2σ threshold |
| `anomaly_flags_k3.png` | Binary flags at 3σ threshold |
| `metrics_heatmap.png` | Performance metrics heatmap |
| `pipeline_results.json` | Full results in machine-readable format |

---

## Dependencies

```
numpy
scipy
scikit-learn  # (used indirectly via scipy clustering)
pandas
matplotlib
```

Install with:
```bash
pip install numpy scipy pandas matplotlib
```
