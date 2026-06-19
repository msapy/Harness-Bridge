import sys
sys.path.append('srim_pipeline')
from data_segmenter import DataSegmenter
from srim_identifier import SRIMIdentifier
from modal_clusterer import ModalClusterer

seg = DataSegmenter(r'Tests\Tests\day2 Clean.csv', 60, 0.5)
segs = seg.get_segments()
srim = SRIMIdentifier(i_factor=10, min_order=2, max_order=80, min_freq=0.5, max_freq=64.0)

poles = srim.identify(segs[0])
clusterer = ModalClusterer(lim_f=0.02, lim_mac=0.05, mpc_thresh=0.50, max_damping=0.10)
stable = clusterer.build_stabilization(poles)
cleared = clusterer.clear_diagram(stable)

print('Day 2 Segment 0 Cleared Modes:')
for c in cleared:
    if c['n_poles'] >= 5:
        print(f"Freq: {c['freq']:.3f} Hz, Damping: {c['damping']*100:.2f}%, Poles: {c['n_poles']}")
