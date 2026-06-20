import numpy as np
from pathlib import Path
import sys
sys.path.append('srim_pipeline')
from run_modal_survey import process_day_raw
from modal_clusterer import ModalClusterer

day1_poles, _, _ = process_day_raw(1, Path('srim_pipeline/data'), {'fs': 128.0, 'seg_length': 60.0, 'overlap': 0.5, 'detrend': True}, {'fs': 128.0, 'i_factor': 10, 'max_order': 80, 'min_freq': 0.5, 'max_freq': 64.0}, max_segments=2)

clusterer = ModalClusterer(min_cluster_size=10, cluster_threshold=0.15, lim_mac=0.1)
stable0 = clusterer.build_stabilization(day1_poles[0])
cleared0 = clusterer.clear_diagram(stable0)
raw_clusters = clusterer.cluster_poles(cleared0)
clean_clusters = clusterer.remove_outliers(raw_clusters)
cluster_modes = sorted(clean_clusters.items(), key=lambda kv: np.median([p['freq'] for p in kv[1]]))
final_modes = [clusterer.aggregate_cluster(poles) for _, poles in cluster_modes]

print('Found', len(final_modes), 'modes')
for i, m in enumerate(final_modes):
    f = m['freq']
    poles = len(cluster_modes[i][1])
    print(f'Mode {i+1}: {f:.3f} Hz, poles: {poles}')
