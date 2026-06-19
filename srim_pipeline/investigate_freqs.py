"""
investigate_freqs.py - diagnostic: what frequencies does SRIM find?
"""
import sys
sys.path.insert(0, r'c:\Users\Win10\Desktop\Harness Bridge\srim_pipeline')
import numpy as np
from data_segmenter import DataSegmenter
from srim_identifier import SRIMIdentifier
from modal_clusterer import ModalClusterer

seg = DataSegmenter(fs=128.0, segment_length_s=60.0, overlap=0.5)
seg.load_csv(r'c:\Users\Win10\Desktop\Harness Bridge\Tests\Tests\day1 Clean.csv')
segments = seg.get_segments()[:2]

# Try with max_freq=20 Hz to focus on structural modes
for max_f in [20.0, 30.0, 64.0]:
    print(f"\n--- max_freq={max_f} Hz ---")
    srim = SRIMIdentifier(fs=128.0, i_factor=10, max_order=30, min_freq=0.1, max_freq=max_f)
    cl   = ModalClusterer(min_cluster_size=1)

    data = segments[0]['data']
    poles_by_order = srim.identify(data)
    all_poles = [p for plist in poles_by_order.values() for p in plist]
    pos_damp = [(round(p['freq'],3), round(p['damping']*100,3)) for p in all_poles
                if 0 < p['damping'] < 0.20 and p['freq'] < max_f]
    pos_damp.sort()
    print(f"  Positive-damped poles in [0.1, {max_f}] Hz: {len(pos_damp)}")
    for fd in pos_damp[:20]:
        print(f"    f={fd[0]:.3f} Hz, xi={fd[1]:.3f}%")

    result = cl.process(poles_by_order)
    print(f"  Clusters found: {len(result)}")
    for k, v in result.items():
        print(f"    Mode {k}: f={v['freq']:.3f} Hz, xi={v['damping']*100:.2f}%")
