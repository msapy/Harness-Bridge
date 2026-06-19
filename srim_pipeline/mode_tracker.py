"""
mode_tracker.py
===============
Reference-anchored modal tracking for the Harness Bridge 10-day campaign.

Problem with naive cluster-index ordering
------------------------------------------
The existing ModalClusterer assigns cluster indices (0, 1, 2, ...) purely by
ascending median frequency within each segment.  This means the same index can
refer to completely different physical modes in adjacent segments whenever the
clearer produces a different number of clusters or the same set of modes is
present but in a different ordering.

Solution: two-phase MAC-anchored tracking
-----------------------------------------

Phase 1 — Reference consensus (Day 1 only, unsupervised)
    Pool every cleared pole from every Day 1 segment into one large set.
    Apply global hierarchical clustering on that pool to find the stable
    frequency bands that persist across segments.  The top n_modes clusters
    become the "reference modes": each gets a consensus shape phi_ref_i
    (mean of all cleared pole shapes in that cluster, real part, normalised)
    and a reference frequency f_ref_i.

Phase 2 — Per-segment MAC-anchored assignment (all days)
    For every cleared pole in a new segment, compute MAC(phi_pole, phi_ref_i)
    for all i.  Assign the pole to the mode with the highest MAC, subject to:
        MAC  >= mac_min          (default 0.60)
        |f - f_ref_i| / f_ref_i <= freq_window   (default 20 %)
    Aggregate assigned poles per mode: median frequency & damping, mean shape.
    Modes with no valid assignment are returned as None for that segment.

This guarantees that mode index 0 always refers to the first bending mode
identified on Day 1, regardless of how many other modes happen to be found
in subsequent segments.

Follows the methodology of Bridge 1 in Tran & Ozer (2020), Sensors 20:4752.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Dict, List, Optional, Tuple

from modal_clusterer import ModalClusterer

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Pole          = Dict   # {'freq', 'damping', 'shape', 'order', ...}
ClusterResult = Dict   # {'freq', 'damping', 'shape', 'n_poles', 'freq_std', 'damping_std', 'mpc'}
TrackResult   = Dict   # {mode_idx (int): ClusterResult or None}


class ModeTracker:
    """
    Two-phase reference-anchored modal tracker.

    Parameters
    ----------
    n_modes : int
        Number of physical modes to track (default 3 for first three bending modes).
    mac_min : float
        Minimum MAC between a pole and a reference shape for the pole to be
        assigned to that mode (default 0.60).
    freq_window : float
        Maximum relative frequency deviation from the reference frequency for
        assignment: |f - f_ref| / f_ref <= freq_window (default 0.20 = 20 %).
    global_cluster_threshold : float
        Euclidean distance cut-off for the global Day-1 hierarchical clustering
        used to build reference modes (default 0.30).  Smaller values produce
        tighter, better-resolved reference clusters.
    mpc_threshold : float
        MPC threshold used during clearance (passed to the ModalClusterer
        helper used internally).
    max_damping : float
        Hard upper limit on damping ratio (10 % default).
    """

    def __init__(
        self,
        n_modes: int = 3,
        mac_min: float = 0.60,
        freq_window: float = 0.20,
        global_cluster_threshold: float = 0.30,
        mpc_threshold: float = 0.50,
        max_damping: float = 0.10,
    ):
        self.n_modes = n_modes
        self.mac_min = mac_min
        self.freq_window = freq_window
        self.global_cluster_threshold = global_cluster_threshold
        self.mpc_threshold = mpc_threshold
        self.max_damping = max_damping

        # Set in build_references()
        self._ref_shapes: List[np.ndarray] = []   # (n_modes,) of shape arrays
        self._ref_freqs:  List[float]      = []   # reference frequency per mode
        self._ref_damping: List[float]     = []   # reference damping per mode
        self._ref_n_poles: List[int]       = []   # pool size per reference cluster

        # Helper for MAC / MPC
        self._clusterer = ModalClusterer(mpc_threshold=mpc_threshold,
                                         max_damping=max_damping)

    # ================================================================== #
    #  Phase 1 — Reference mode shape consensus from Day 1               #
    # ================================================================== #

    def build_references(
        self,
        day1_poles_list: List[Dict[int, List[Pole]]],
    ) -> "ModeTracker":
        """
        Build n_modes reference mode shapes from Day 1 data.

        Uses a two-level approach:
          1. Segment-level: apply full ModalClusterer.process() to each segment
             to extract clean mode centroids (free of noise poles).
          2. Global-level: pool all centroids from all Day 1 segments and cluster
             them by MAC to find the persistent reference modes.

        Parameters
        ----------
        day1_poles_list : list of dicts
            One element per Day 1 segment.  Each element is the raw
            ``poles_by_order`` dict returned by SRIMIdentifier.identify().

        Returns
        -------
        self  (for method chaining)
        """
        # --- Collect all clean mode centroids from every Day 1 segment ---
        all_centroids: List[ClusterResult] = []

        for seg_idx, poles_by_order in enumerate(day1_poles_list):
            # Level 1: per-segment automated clustering
            seg_modes = self._clusterer.process(poles_by_order)
            
            for res in seg_modes.values():
                # res is a ClusterResult dict with 'freq', 'damping', 'shape', etc.
                if res["damping"] > self.max_damping:
                    continue
                # process() already filters by MPC and max_damping during clearance
                all_centroids.append(res)

        if len(all_centroids) < self.n_modes:
            raise RuntimeError(
                f"Only {len(all_centroids)} mode centroids from Day 1; "
                f"cannot build {self.n_modes} reference modes.  "
                "Try lowering mpc_threshold or increasing segment count."
            )

        print(
            f"[ModeTracker] Phase 1: {len(all_centroids)} clean mode centroids "
            f"extracted from {len(day1_poles_list)} Day-1 segments."
        )

        # --- Global hierarchical clustering on the full Day-1 centroid pool ---
        ref_clusters = self._global_cluster(all_centroids)

        # Drop trivially small clusters (likely noise/spurious)
        # Require at least 5% of segments to have found this mode, or min 3
        min_size = max(3, int(0.05 * len(day1_poles_list)))
        ref_clusters = {
            cid: centroids for cid, centroids in ref_clusters.items()
            if len(centroids) >= min_size
        }

        if len(ref_clusters) < self.n_modes:
            print(
                f"[ModeTracker] WARNING: only {len(ref_clusters)} reference "
                f"clusters found after noise filtering (wanted {self.n_modes}).  "
            )

        # Sort clusters by median frequency (ascending)
        # Take up to 15 candidate clusters for full tracking
        sorted_clusters = sorted(
            ref_clusters.items(),
            key=lambda kv: np.median([p["freq"] for p in kv[1]]),
        )[:15]

        # Temporarily set n_modes to track all candidates
        self.n_modes = len(sorted_clusters)

        # --- Compute consensus reference shapes ---------------------------
        self._ref_shapes  = []
        self._ref_freqs   = []
        self._ref_damping = []
        self._ref_n_poles = []

        for cid, poles in sorted_clusters:
            freqs  = np.array([p["freq"]    for p in poles])
            damps  = np.array([p["damping"] for p in poles])
            shapes = [np.real(p["shape"])   for p in poles]

            f_ref = float(np.median(freqs))
            d_ref = float(np.median(damps))

            # Consensus shape: mean of real parts, normalised to unit max
            if shapes:
                ref_s = shapes[0]
                aligned_shapes = []
                for s in shapes:
                    if np.real(s @ ref_s) < 0:
                        aligned_shapes.append(-s)
                    else:
                        aligned_shapes.append(s)
                mean_shape = np.mean(np.vstack(aligned_shapes), axis=0)
            else:
                mean_shape = np.mean(np.vstack(shapes), axis=0)
            norm = np.max(np.abs(mean_shape))
            mean_shape = mean_shape / norm if norm > 0 else mean_shape

            self._ref_shapes.append(mean_shape)
            self._ref_freqs.append(f_ref)
            self._ref_damping.append(d_ref)
            self._ref_n_poles.append(len(poles))

            print(
                f"  Mode {len(self._ref_shapes)}: f_ref = {f_ref:.3f} Hz, "
                f"ξ_ref = {d_ref*100:.2f}%,  n_poles = {len(poles)}"
            )

        return self

    def _global_cluster(
        self, poles: List[Pole]
    ) -> Dict[int, List[Pole]]:
        """
        MAC-dominated hierarchical clustering on the full Day-1 pole pool.

        Key insight: real physical bending modes have geometrically distinct,
        near-orthogonal shapes (inter-mode MAC << 0.5), so MAC distance is the
        PRIMARY separator.  Closely-spaced spurious/noise poles at similar
        frequencies but with random shapes will scatter across clusters.

        Distance metric (for poles i and j):
            d(i,j) = sqrt( (mac_weight * mac_diff)^2 + (freq_weight * f_diff)^2 )

        where:
            mac_diff = 1 - MAC(phi_i, phi_j)   in [0, 1]
            f_diff   = |f_i - f_j| / f_max     in [0, 1]
            mac_weight  = 1.0   (dominant term)
            freq_weight = 0.15  (soft regularisation only)

        For two poles from the same physical mode:   mac_diff ≈ 0.05-0.15
        For two poles from different physical modes: mac_diff ≈ 0.40-0.90
        => cluster threshold of ~0.30 on this distance cleanly separates modes.
        """
        K = len(poles)
        if K == 0:
            return {}
        if K == 1:
            return {1: poles}

        freqs = np.array([p["freq"] for p in poles])
        f_max = freqs.max() if freqs.max() > 0 else 1.0

        MAC_W  = 1.00   # MAC is the primary separator
        FREQ_W = 0.15   # frequency is a soft secondary regulariser

        # Build condensed distance matrix in O(K^2)
        # For large pools (K > 5000), subsample to avoid memory issues
        if K > 4000:
            rng  = np.random.default_rng(42)
            idx  = rng.choice(K, size=4000, replace=False)
            idx  = np.sort(idx)
            sub  = [poles[i] for i in idx]
            print(
                f"[ModeTracker] Large pool ({K} poles); "
                f"subsampling to {len(sub)} for reference clustering."
            )
            return self._global_cluster(sub)

        dists = []
        for i in range(K):
            for j in range(i + 1, K):
                f_diff   = abs(poles[i]["freq"] - poles[j]["freq"]) / f_max
                mac_diff = 1.0 - self._clusterer.mac(
                    poles[i]["shape"], poles[j]["shape"]
                )
                d = np.sqrt((MAC_W * mac_diff) ** 2 + (FREQ_W * f_diff) ** 2)
                dists.append(d)

        dists  = np.array(dists)
        Z      = linkage(dists, method="complete")
        labels = fcluster(
            Z, t=self.global_cluster_threshold, criterion="distance"
        )

        clusters: Dict[int, List[Pole]] = {}
        for pole, lbl in zip(poles, labels):
            clusters.setdefault(int(lbl), []).append(pole)

        return clusters

    # ================================================================== #
    #  Phase 2 — Per-segment MAC-anchored mode assignment                #
    # ================================================================== #

    def assign_modes(
        self,
        poles_by_order: Dict[int, List[Pole]],
        cleared_poles: Optional[List[Pole]] = None,
    ) -> TrackResult:
        """
        Assign cleared poles from one segment to reference modes via MAC.

        Parameters
        ----------
        poles_by_order : dict
            Raw output of SRIMIdentifier.identify() for one segment.
        cleared_poles : list[Pole] or None
            Pre-computed cleared poles.  If None, stabilisation + clearance
            is run internally.

        Returns
        -------
        dict: {mode_idx: ClusterResult or None}
            mode_idx in range(self.n_modes).  Value is None when no pole
            could be assigned to that mode in this segment.
        """
        if not self._ref_shapes:
            raise RuntimeError("Call build_references() before assign_modes().")

        # Get cleared poles
        if cleared_poles is None:
            stable  = self._clusterer.build_stabilization(poles_by_order)
            cleared = self._clusterer.clear_diagram(stable)
        else:
            cleared = cleared_poles

        # Filter by max_damping
        cleared = [p for p in cleared if 0 < p["damping"] <= self.max_damping]

        if not cleared:
            return {i: None for i in range(self.n_modes)}

        # --- Assign each cleared pole to best reference mode --------------
        # assignment[i] = list of poles assigned to mode i
        assignment: Dict[int, List[Pole]] = {i: [] for i in range(self.n_modes)}

        for pole in cleared:
            best_mode = -1
            best_mac  = -1.0

            phi_pole = pole["shape"]
            f_pole   = pole["freq"]

            for i, (phi_ref, f_ref) in enumerate(
                zip(self._ref_shapes, self._ref_freqs)
            ):
                # Frequency window gate
                rel_df = abs(f_pole - f_ref) / f_ref
                if rel_df > self.freq_window:
                    continue

                # MAC with reference shape (use real part comparison)
                # Real-valued reference vs complex pole shape: use real part
                phi_pole_real = np.real(phi_pole)
                norm1 = np.linalg.norm(phi_pole_real)
                norm2 = np.linalg.norm(phi_ref)
                if norm1 < 1e-15 or norm2 < 1e-15:
                    continue

                dot = abs(phi_pole_real @ phi_ref)
                mac_val = (dot ** 2) / (norm1**2 * norm2**2)
                mac_val = float(np.clip(mac_val, 0.0, 1.0))

                if mac_val > best_mac:
                    best_mac  = mac_val
                    best_mode = i

            if best_mode >= 0 and best_mac >= self.mac_min:
                assignment[best_mode].append(pole)

        # --- Aggregate assigned poles into one result per mode ------------
        result: TrackResult = {}
        for i in range(self.n_modes):
            poles_i = assignment[i]
            if not poles_i:
                result[i] = None
                continue

            # Prevent intra-segment mode mixing due to wide freq-window:
            # Find the pole with the highest MAC to the reference
            ref_s = self._ref_shapes[i]
            
            def _mac_to_ref(p):
                ps = np.real(p["shape"])
                return (abs(ps @ ref_s)**2) / (np.linalg.norm(ps)**2 * np.linalg.norm(ref_s)**2 + 1e-30)
                
            best_pole = max(poles_i, key=_mac_to_ref)
            f_best = best_pole["freq"]
            
            # Keep only poles within ±5% of the best pole's frequency
            filtered_poles = [p for p in poles_i if abs(p["freq"] - f_best) / f_best < 0.05]
            
            freqs  = np.array([p["freq"]    for p in filtered_poles])
            damps  = np.array([p["damping"] for p in filtered_poles])
            shapes = [np.real(p["shape"])   for p in filtered_poles]

            # Median aggregation (robust)
            f_med  = float(np.median(freqs))
            xi_med = float(np.median(damps))

            if shapes:
                ref_s = self._ref_shapes[i]
                aligned_shapes = []
                for s in shapes:
                    if np.real(s @ ref_s) < 0:
                        aligned_shapes.append(-s)
                    else:
                        aligned_shapes.append(s)
                mean_shape = np.mean(np.vstack(aligned_shapes), axis=0)
            else:
                mean_shape = np.mean(np.vstack(shapes), axis=0)
            norm = np.max(np.abs(mean_shape))
            mean_shape = mean_shape / norm if norm > 0 else mean_shape

            # MPC of mean shape
            mpc_val = self._clusterer.mpc(
                mean_shape.astype(complex)
            )

            result[i] = {
                "freq":         f_med,
                "damping":      xi_med,
                "shape":        mean_shape,
                "freq_std":     float(np.std(freqs)),
                "damping_std":  float(np.std(damps)),
                "n_poles":      len(poles_i),
                "mpc":          mpc_val,
                "mac_to_ref":   float(
                    (abs(mean_shape @ self._ref_shapes[i])**2)
                    / (np.linalg.norm(mean_shape)**2 * np.linalg.norm(self._ref_shapes[i])**2 + 1e-30)
                ),
            }

        return result

    def track_all_days(
        self,
        all_poles_by_order: List[Dict[int, List[Pole]]],
    ) -> List[TrackResult]:
        """
        Apply assign_modes() to every segment in the full dataset.

        Parameters
        ----------
        all_poles_by_order : list
            One poles_by_order dict per segment (across all days, concatenated).

        Returns
        -------
        list of TrackResult, same length as input.
        """
        results = []
        for seg_poles in all_poles_by_order:
            tr = self.assign_modes(seg_poles)
            results.append(tr)
        return results

    # ================================================================== #
    #  Phase 3 — Select best physical modes across all days                #
    # ================================================================== #

    def select_best_modes(
        self, all_track_results: List[TrackResult], top_n: int = 3
    ) -> List[TrackResult]:
        """
        Phase 3: Select the true physical bending modes based on cross-day stability.
        
        Evaluates all tracked candidates across the full dataset and selects the 
        top_n candidates with the highest total Identification Rate (most presence).
        """
        if self.n_modes <= top_n:
            self.n_modes = top_n
            return all_track_results
            
        counts = {i: 0 for i in range(self.n_modes)}
        for tr in all_track_results:
            for i in range(self.n_modes):
                if tr.get(i) is not None:
                    counts[i] += 1
                    
        # Sort mode indices by their presence count (descending)
        sorted_by_presence = sorted(range(self.n_modes), key=lambda i: counts[i], reverse=True)
        
        best_indices = []
        for idx in sorted_by_presence:
            if len(best_indices) >= top_n:
                break
                
            # Enforce MAC-based spatial diversity to avoid duplicate/aliased mode shapes
            shape_idx = self._ref_shapes[idx]
            is_duplicate = False
            for selected_idx in best_indices:
                shape_sel = self._ref_shapes[selected_idx]
                dot = abs(shape_idx @ shape_sel)
                norm1 = np.linalg.norm(shape_idx)
                norm2 = np.linalg.norm(shape_sel)
                mac_val = (dot**2) / (norm1**2 * norm2**2 + 1e-30)
                if mac_val > 0.80:  # 80% MAC similarity threshold
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                best_indices.append(idx)
        
        # Sort the chosen indices by their original frequency (ascending)
        best_indices = sorted(best_indices, key=lambda i: self._ref_freqs[i])
        
        print(f"\n[ModeTracker] Phase 3: Selected {top_n} best modes from {self.n_modes} candidates:")
        print("  " + "-" * 50)
        for new_idx, old_idx in enumerate(best_indices):
            print(f"  Mode {new_idx+1} (was Candidate {old_idx+1}): "
                  f"f_ref = {self._ref_freqs[old_idx]:.3f} Hz, "
                  f"Found in {counts[old_idx]} segments")
        print("  " + "-" * 50 + "\n")
                  
        # Update references
        self._ref_shapes = [self._ref_shapes[i] for i in best_indices]
        self._ref_freqs = [self._ref_freqs[i] for i in best_indices]
        self._ref_damping = [self._ref_damping[i] for i in best_indices]
        self._ref_n_poles = [self._ref_n_poles[i] for i in best_indices]
        self.n_modes = top_n
        
        # Filter all_track_results mapping the old indices to 0..top_n-1
        filtered_results = []
        for tr in all_track_results:
            new_tr = {new_idx: tr.get(old_idx) for new_idx, old_idx in enumerate(best_indices)}
            filtered_results.append(new_tr)
            
        return filtered_results

    # ================================================================== #
    #  Properties / diagnostics                                           #
    # ================================================================== #

    @property
    def reference_frequencies(self) -> List[float]:
        return list(self._ref_freqs)

    @property
    def reference_shapes(self) -> List[np.ndarray]:
        return [s.copy() for s in self._ref_shapes]

    def reference_summary(self) -> str:
        if not self._ref_freqs:
            return "No references built yet."
        lines = ["Mode | f_ref (Hz) | ξ_ref (%) | n_poles"]
        lines.append("-" * 40)
        for i, (f, d, n) in enumerate(
            zip(self._ref_freqs, self._ref_damping, self._ref_n_poles)
        ):
            lines.append(f"  {i+1}  | {f:10.4f} | {d*100:9.3f} | {n}")
        return "\n".join(lines)
