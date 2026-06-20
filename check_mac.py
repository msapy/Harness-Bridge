import numpy as np
import json
import sys
from pathlib import Path
sys.path.append('srim_pipeline')
from mode_tracker import ModeTracker
from run_modal_survey import process_day_raw
from modal_clusterer import ModalClusterer

def main():
    import warnings
    warnings.filterwarnings('ignore')
    
    # We will just parse the config from the script or run the clustering
    clusterer = ModalClusterer(min_cluster_size=5, cluster_threshold=0.15, lim_mac=0.05)
    # The data is in c:\Users\Win10\Desktop\Harness Bridge\srim_pipeline\data
    day1_poles, _, _ = process_day_raw(1, Path('srim_pipeline/data'), {'fs': 128.0, 'seg_length': 60.0, 'overlap': 0.5, 'detrend': True}, {'fs': 128.0, 'i_factor': 10, 'max_order': 80, 'min_freq': 0.5, 'max_freq': 64.0}, max_segments=2)

    stable0 = clusterer.build_stabilization(day1_poles[0])
    cleared0 = clusterer.clear_diagram(stable0)
    raw_clusters = clusterer.cluster_poles(cleared0)
    clean_clusters = clusterer.remove_outliers(raw_clusters)
    cluster_modes = sorted(clean_clusters.items(), key=lambda kv: np.median([p['freq'] for p in kv[1]]))
    final_modes = [clusterer.aggregate_cluster(poles) for _, poles in cluster_modes]

    print("Total modes found:", len(final_modes))
    for i in range(len(final_modes)):
        f = final_modes[i]['freq']
        n = final_modes[i]['n_poles']
        print(f"Mode {i+1}: {f:.3f} Hz, {n} poles")

    print("\nMAC Matrix:")
    for i in range(len(final_modes)):
        for j in range(i+1, len(final_modes)):
            s1 = final_modes[i]['shape']
            s2 = final_modes[j]['shape']
            # Get real shape to align phases
            s1_real = clusterer.get_real_shape(s1)
            s2_real = clusterer.get_real_shape(s2)
            mac = abs(s1_real @ s2_real)**2 / (np.linalg.norm(s1_real)**2 * np.linalg.norm(s2_real)**2 + 1e-15)
            print(f"MAC(Mode {i+1}, Mode {j+1}) = {mac:.3f}")

if __name__ == '__main__':
    main()
