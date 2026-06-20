import numpy as np
import sys
from pathlib import Path
sys.path.append('srim_pipeline')
from run_modal_survey import process_day_raw
from modal_clusterer import ModalClusterer

def main():
    import warnings
    warnings.filterwarnings('ignore')
    
    day1_poles, _, _ = process_day_raw(1, Path("C:/Users/Win10/Desktop/Harness Bridge/Tests/Tests"), {'fs': 128.0, 'segment_length': 60.0, 'overlap': 0.5, 'detrend': True}, {'fs': 128.0, 'i_factor': 10, 'max_order': 80, 'min_freq': 0.5, 'max_freq': 64.0}, max_segments=2)

    for thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        clusterer = ModalClusterer(min_cluster_size=10, cluster_threshold=thresh, lim_mac=0.05)
        stable0 = clusterer.build_stabilization(day1_poles[0])
        cleared0 = clusterer.clear_diagram(stable0)
        raw_clusters = clusterer.cluster_poles(cleared0)
        clean_clusters = clusterer.remove_outliers(raw_clusters)
        cluster_modes = sorted(clean_clusters.items(), key=lambda kv: np.median([p['freq'] for p in kv[1]]))
        final_modes = [clusterer.aggregate_cluster(poles) for _, poles in cluster_modes]
        
        freqs = [m['freq'] for m in final_modes]
        print(f"thresh={thresh:.2f} -> {len(final_modes)} modes: {[round(f, 3) for f in freqs]}")

if __name__ == '__main__':
    main()
