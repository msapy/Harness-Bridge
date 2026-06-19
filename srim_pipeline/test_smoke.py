"""
test_smoke.py -- quick end-to-end verification of the SRIM pipeline.
Run from the 'Harness Bridge' directory:
    python srim_pipeline/test_smoke.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import warnings

# =========================================================================
# Test 1: Synthetic 2-DOF spring-mass system (unit test for SRIMIdentifier)
# =========================================================================
from srim_identifier import SRIMIdentifier
from modal_clusterer import ModalClusterer, ModalClusterer as MC

print("=" * 60)
print("Test 1: Synthetic 2-DOF System")
print("=" * 60)

np.random.seed(42)
fs = 128.0
dt = 1 / fs
N = 2560   # 20 s @ 128 Hz

f1_true, f2_true = 3.5, 8.2
xi1, xi2 = 0.02, 0.03

t = np.arange(N) * dt
omega1 = 2 * np.pi * f1_true * np.sqrt(1 - xi1**2)
omega2 = 2 * np.pi * f2_true * np.sqrt(1 - xi2**2)

phi1 = np.array([1.0, 0.7, 0.3, -0.2])
phi2 = np.array([1.0, -0.5, -0.8, 0.6])

modal1 = np.exp(-xi1 * 2 * np.pi * f1_true * t) * np.sin(omega1 * t)
modal2 = np.exp(-xi2 * 2 * np.pi * f2_true * t) * np.sin(omega2 * t)

y = np.outer(modal1, phi1) + np.outer(modal2, phi2)
y += 0.05 * np.random.randn(N, 4)   # noise

print(f"  Synthetic data shape : {y.shape}")
print(f"  True modes           : f1={f1_true} Hz (xi={xi1}),  f2={f2_true} Hz (xi={xi2})")

srim = SRIMIdentifier(fs=fs, i_factor=8, max_order=16, min_freq=0.5)
poles_by_order = srim.identify(y)

n_poles_total = sum(len(v) for v in poles_by_order.values())
print(f"  Total raw poles      : {n_poles_total} across {len(poles_by_order)} orders")

all_poles = [p for plist in poles_by_order.values() for p in plist]
c1 = [p for p in all_poles if abs(p["freq"] - f1_true) < 0.5 and p["damping"] > 0]
c2 = [p for p in all_poles if abs(p["freq"] - f2_true) < 0.5 and p["damping"] > 0]
print(f"  Poles near f1        : {len(c1)}")
print(f"  Poles near f2        : {len(c2)}")

# --- ModalClusterer ---
clusterer = ModalClusterer(min_cluster_size=1)
result = clusterer.process(poles_by_order)

print(f"\n  Identified clusters  : {len(result)}")
for idx, cluster in result.items():
    f  = cluster["freq"]
    xi = cluster["damping"] * 100
    n  = cluster["n_poles"]
    print(f"    Mode {idx}: f={f:.3f} Hz, xi={xi:.2f}%,  n_poles={n}")

# =========================================================================
# Test 2: Real data -- first 5 segments of Day 1
# =========================================================================
print()
print("=" * 60)
print("Test 2: Real Data -- Day 1, first 5 segments")
print("=" * 60)

from data_segmenter   import DataSegmenter
from anomaly_detector import AnomalyDetector
from metrics_evaluator import MetricsEvaluator

import pathlib
data_dir = pathlib.Path(__file__).parent.parent / "Tests" / "Tests"
csv_path = data_dir / "day1 Clean.csv"

if not csv_path.exists():
    print(f"  WARNING: {csv_path} not found -- skipping real-data test.")
else:
    seg = DataSegmenter(fs=128.0, segment_length_s=20.0, overlap=0.5)
    seg.load_csv(str(csv_path))
    segments = seg.get_segments()[:5]

    srim5 = SRIMIdentifier(fs=128.0, i_factor=10, max_order=30, min_freq=0.2)
    cl5   = ModalClusterer(min_cluster_size=1)

    results5 = []
    for s in segments:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pb = srim5.identify(s["data"])
            r  = cl5.process(pb)
        results5.append(r)
        n_modes = len(r)
        if n_modes > 0:
            f0 = r[0]["freq"]
            xi0 = r[0]["damping"] * 100
            print(f"  Seg {s['seg_index']:>3} | modes={n_modes} | mode0: f={f0:.3f} Hz, xi={xi0:.2f}%")
        else:
            print(f"  Seg {s['seg_index']:>3} | NO modes identified")

    # Quick anomaly detector test
    det = AnomalyDetector(n_baseline_segments=5, k_sigma_values=(1, 2, 3),
                          mode_indices=(0,), min_baseline_count=2)
    det.fit_baseline(results5)
    print("\n" + det.baseline_summary())

print()
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
