"""
modal_clusterer.py
==================
ModalClusterer class: implements the full automated modal parameter extraction
pipeline from a stabilization diagram following Tran & Ozer (2020), S.2.2.

Pipeline stages (paper S.2.2.2)
-------------------------------

**Step 0 - Build Stabilization Diagram** (S.2.2.1, Eqs 19-21)
    For consecutive system orders n and n+1, a pole is "stable" if:
        freq : |f_{n+1} - f_n| / f_n  < lim_f       (Eq. 19)
        damp : |xi_{n+1} - xi_n| / xi_n < lim_xi       (Eq. 20)
        MAC  : 1 - MAC(phi_n, phi_{n+1})  < lim_MAC     (Eq. 21)

    This study uses "type-3" stable poles: stable in frequency AND mode shape,
    NOT necessarily in damping (paper S.2.2.2 first paragraph - avoids shortage
    of poles in small datasets).

**Step 1 - Diagram Clearance** (S.2.2.2 Step 1)
    a) Drop poles with damping xi <= 0 (non-physical).
    b) Apply Modal Phase Collinearity (MPC) (Eqs. 22-26) to remove spurious
       poles with incoherent phase structure.

**Step 2 - Hierarchical Clustering** (S.2.2.2 Step 2)
    Group similar poles using single-linkage hierarchical clustering.
    Feature vector: [f / f_max,  1 - MAC_ref].
    Distance threshold fixed at 0.5 (paper default).

**Step 3 - MAD Outlier Removal** (S.2.2.2 Step 3, Eq. 27)
    Within each cluster remove poles whose damping ratio is an outlier:
        MAD_j = 1.482 * median(|xi_i - median(xi_i)|)
    Also drop any pole with xi > max_damping (default 10 %).
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Pole = Dict          # {'freq', 'damping', 'shape', 'order', 'eigenvalue'}
ClusterResult = Dict  # {'freq', 'damping', 'shape', 'n_poles', 'cluster_id'}


class ModalClusterer:
    """
    Automated stabilization, clearance, clustering, and outlier removal.

    Parameters
    ----------
    lim_f : float
        Max relative change in frequency between consecutive orders for a
        pole to be "stable" (default 0.01 = 1 %, paper default).
    lim_xi : float
        Max relative change in damping ratio (default 0.05 = 5 %).
    lim_mac : float
        Max (1 - MAC) between consecutive orders (default 0.05 = 5 %,
        equivalent to requiring MAC > 0.95).
    mpc_threshold : float
        Minimum MPC value [0, 1] to keep a pole (default 0.5).
    cluster_threshold : float
        Euclidean distance cut for hierarchical clustering (default 0.5).
    max_damping : float
        Hard upper bound on damping ratio; poles above this are discarded
        (default 0.10 = 10 %).
    min_cluster_size : int
        Minimum number of poles required to accept a cluster.  The paper
        uses an adaptive value to ensure at least one cluster per segment;
        set to 1 to maximise recall on short windows.
    f_normalise_max : float or None
        If set, normalise frequencies by this value when computing cluster
        feature vectors.  If None, uses the maximum frequency seen in the
        current pole set.
    """

    def __init__(
        self,
        lim_f: float = 0.01,
        lim_xi: float = 0.05,
        lim_mac: float = 0.05,
        mpc_threshold: float = 0.5,
        cluster_threshold: float = 0.5,
        max_damping: float = 0.10,
        min_cluster_size: int = 3,
        f_normalise_max: Optional[float] = None,
    ):
        self.lim_f = lim_f
        self.lim_xi = lim_xi
        self.lim_mac = lim_mac
        self.mpc_threshold = mpc_threshold
        self.cluster_threshold = cluster_threshold
        self.max_damping = max_damping
        self.min_cluster_size = min_cluster_size
        self.f_normalise_max = f_normalise_max

    # ================================================================== #
    #  MAC and MPC helper functions                                        #
    # ================================================================== #

    @staticmethod
    def mac(phi1: np.ndarray, phi2: np.ndarray) -> float:
        """
        Modal Assurance Criterion (Eq. 21).

        MAC(phi₁, phi₂) = |phi₁ᴴ phi₂|^2 / (phi₁ᴴ phi₁ * phi₂ᴴ phi₂)

        Range [0, 1]: 0 -> orthogonal (different modes),
                      1 -> collinear (same mode).

        (.)ᴴ denotes complex conjugate transpose.

        Parameters
        ----------
        phi1, phi2 : complex np.ndarray  (m,)

        Returns
        -------
        float in [0, 1]
        """
        num = np.abs(phi1.conj() @ phi2) ** 2
        den = np.real(phi1.conj() @ phi1) * np.real(phi2.conj() @ phi2)
        if den < 1e-15:
            return 0.0
        return float(np.clip(num / den, 0.0, 1.0))

    @staticmethod
    def mpc(phi: np.ndarray) -> float:
        """
        Modal Phase Collinearity (Eqs. 22-26).

        Evaluates the spatial coherence of the phase angles across sensor
        channels.  Physical modes have nearly real mode shapes (all sensors
        vibrating in-phase or 180° out-of-phase), giving MPC ~= 1.
        Spurious numerical modes have random phase -> MPC ~= 0.

        Algorithm:
            Let phi_r = Re(phi),  phi_i = Im(phi).
            S_rr = phi_rᵀ phi_r   (Eq. 22)
            S_ii = phi_iᵀ phi_i   (Eq. 23)
            S_ri = phi_rᵀ phi_i   (Eq. 24)
            η    = (S_rr - S_ii) / (2 S_ri)
            lambda₁,₂ = (S_rr + S_ii)/2 +/- S_ri sqrt(η^2 + 1)   (Eq. 25)
            MPC  = ((lambda₁ - lambda₂) / (lambda₁ + lambda₂))^2 * 100 %  (Eq. 26)

        Parameters
        ----------
        phi : complex np.ndarray  (m,)

        Returns
        -------
        float in [0, 1]
        """
        phi_r = np.real(phi)
        phi_i = np.imag(phi)

        S_rr = float(phi_r @ phi_r)
        S_ii = float(phi_i @ phi_i)
        S_ri = float(phi_r @ phi_i)

        if abs(S_ri) < 1e-15:
            # Purely real mode shape -> MPC = 1 by definition
            return 1.0

        eta = (S_rr - S_ii) / (2.0 * S_ri)
        disc = S_ri * np.sqrt(eta ** 2 + 1.0)

        lam1 = (S_rr + S_ii) / 2.0 + disc
        lam2 = (S_rr + S_ii) / 2.0 - disc

        denom = abs(lam1 + lam2)
        if denom < 1e-15:
            return 0.0

        mpc_val = ((lam1 - lam2) / (lam1 + lam2)) ** 2
        return float(np.clip(mpc_val, 0.0, 1.0))

    # ================================================================== #
    #  Step 0: Stabilization Diagram                                       #
    # ================================================================== #

    def build_stabilization(
        self, poles_by_order: Dict[int, List[Pole]]
    ) -> List[Pole]:
        """
        Identify poles that are "type-3 stable": stable in both frequency and
        mode shape across consecutive system orders (Eqs. 19 and 21).

        The paper deliberately avoids requiring damping stability in this step
        to avoid pole shortages in short data segments (S.2.2.2).

        Parameters
        ----------
        poles_by_order : dict
            Output of SRIMIdentifier.identify().
            Key = system order n, value = list of Pole dicts.

        Returns
        -------
        List of Pole dicts flagged as stable (type-3).
        Each stable pole also carries a 'mac_prev' field for diagnostics.
        """
        orders = sorted(poles_by_order.keys())
        stable_poles: List[Pole] = []

        for idx in range(1, len(orders)):
            n_prev = orders[idx - 1]
            n_curr = orders[idx]

            poles_prev = poles_by_order.get(n_prev, [])
            poles_curr = poles_by_order.get(n_curr, [])

            if not poles_prev or not poles_curr:
                continue

            for p_curr in poles_curr:
                f_curr = p_curr["freq"]
                phi_curr = p_curr["shape"]

                # Find closest pole in previous order by frequency
                best_mac = -1.0
                best_p_prev = None

                for p_prev in poles_prev:
                    f_prev = p_prev["freq"]

                    # Frequency stability check (Eq. 19)
                    if f_prev == 0:
                        continue
                    rel_df = abs(f_curr - f_prev) / f_prev
                    if rel_df >= self.lim_f:
                        continue

                    # MAC check (Eq. 21)
                    mac_val = self.mac(phi_curr, p_prev["shape"])
                    if mac_val > best_mac:
                        best_mac = mac_val
                        best_p_prev = p_prev

                # Accept if MAC passes
                if best_p_prev is not None and (1.0 - best_mac) < self.lim_mac:
                    p_out = dict(p_curr)
                    p_out["mac_prev"] = best_mac
                    stable_poles.append(p_out)

        return stable_poles

    # ================================================================== #
    #  Step 1: Diagram Clearance                                           #
    # ================================================================== #

    def clear_diagram(self, stable_poles: List[Pole]) -> List[Pole]:
        """
        Remove non-physical poles from the stabilization diagram.

        Two filters are applied (S.2.2.2 Step 1):
        a) Drop poles with xi <= 0   (negative or zero damping -> unstable/non-physical).
        b) Drop poles with MPC < mpc_threshold  (incoherent phase -> spurious).

        Parameters
        ----------
        stable_poles : List[Pole]

        Returns
        -------
        List[Pole] after clearance.
        """
        cleared: List[Pole] = []

        for p in stable_poles:
            # (a) Positive damping check
            if p["damping"] <= 0.0:
                continue

            # (b) MPC check (Eqs. 22-26)
            mpc_val = self.mpc(p["shape"])
            if mpc_val < self.mpc_threshold:
                continue

            p_out = dict(p)
            p_out["mpc"] = mpc_val
            cleared.append(p_out)

        return cleared

    # ================================================================== #
    #  Step 2: Hierarchical Clustering                                     #
    # ================================================================== #

    def _make_feature_vectors(
        self, poles: List[Pole]
    ) -> Tuple[np.ndarray, float]:
        """
        Build the feature matrix for clustering.

        Feature vector for pole k:
            [f_k / f_max ,  1 - MAC(phi_k, phi_ref)]

        where phi_ref is the mode shape of the pole with the highest MAC
        to its nearest frequency neighbour (a rough reference).  In practice,
        using the mean frequency bin as reference gives stable results.

        Returns
        -------
        X : np.ndarray, shape (K, 2)   - normalised features
        f_max : float                  - normalisation constant used
        """
        freqs = np.array([p["freq"] for p in poles])
        shapes = [p["shape"] for p in poles]
        K = len(poles)

        # Normalise frequency to [0, 1]
        f_max = self.f_normalise_max if self.f_normalise_max else (freqs.max() or 1.0)

        # MAC distance from pole to the "best neighbour" in frequency
        # (acts as a shape-similarity normaliser)
        mac_dist = np.zeros(K)
        for k in range(K):
            # Find closest pole in frequency (excluding self)
            df = np.abs(freqs - freqs[k])
            df[k] = np.inf
            if df.min() == np.inf:
                continue
            j_near = int(np.argmin(df))
            mac_dist[k] = 1.0 - self.mac(shapes[k], shapes[j_near])

        X = np.column_stack([freqs / f_max, mac_dist])
        return X, f_max

    def cluster_poles(self, cleared_poles: List[Pole]) -> Dict[int, List[Pole]]:
        """
        Apply complete-linkage hierarchical clustering using custom Euclidean distance 
        of normalized frequency difference and MAC shape difference to identify 
        distinct modes and prevent chaining.
        
        Returns
        -------
        Dict mapping cluster_id (1-based) -> List[Pole]
        """
        K = len(cleared_poles)

        if K == 0:
            return {}

        if K == 1:
            return {1: list(cleared_poles)}

        # Compute the custom distance matrix: d(i, j) = sqrt((df/f_max)^2 + (1-MAC)^2)
        f_max = max(p["freq"] for p in cleared_poles)
        dists = []
        for i in range(K):
            for j in range(i + 1, K):
                f_diff = abs(cleared_poles[i]["freq"] - cleared_poles[j]["freq"]) / f_max
                mac_diff = 1.0 - self.mac(cleared_poles[i]["shape"], cleared_poles[j]["shape"])
                d = np.sqrt(f_diff**2 + mac_diff**2)
                dists.append(d)
        dists = np.array(dists)

        # Complete-linkage (furthest-neighbour) to avoid chaining
        Z = linkage(dists, method="complete")

        # Cut dendrogram at fixed threshold
        labels = fcluster(Z, t=self.cluster_threshold, criterion="distance")

        # Group poles by cluster label
        clusters: Dict[int, List[Pole]] = {}
        for pole, label in zip(cleared_poles, labels):
            clusters.setdefault(int(label), []).append(pole)

        return clusters

    # ================================================================== #
    #  Step 3: MAD Outlier Removal                                         #
    # ================================================================== #

    def remove_outliers(
        self, clusters: Dict[int, List[Pole]]
    ) -> Dict[int, List[Pole]]:
        """
        Remove extreme damping-ratio outliers within each cluster using
        the Median Absolute Deviation (Eq. 27).

        MAD_j = 1.482 * median(|xi_i - median(xi_i)|)

        A pole is an outlier if:
            |xi_i - median(xi_i)| > MAD_j    OR    xi_i > max_damping

        The constant 1.482 makes MAD a consistent estimator of sigma under
        a normal distribution (scaling factor for normal consistency).

        Parameters
        ----------
        clusters : dict  cluster_id -> List[Pole]

        Returns
        -------
        Cleaned dict (clusters with fewer than min_cluster_size poles removed).
        """
        cleaned: Dict[int, List[Pole]] = {}

        for cid, poles in clusters.items():
            xis = np.array([p["damping"] for p in poles])

            # Absolute bound
            keep_mask = xis <= self.max_damping

            if keep_mask.sum() == 0:
                continue

            # MAD filter within remaining poles
            xis_kept = xis[keep_mask]
            med_xi = np.median(xis_kept)
            mad = 1.482 * np.median(np.abs(xis_kept - med_xi))

            inlier_mask = keep_mask.copy()
            if mad > 1e-10:
                inlier_mask[keep_mask] &= (
                    np.abs(xis_kept - med_xi) <= mad
                )

            accepted = [p for p, keep in zip(poles, inlier_mask) if keep]

            if len(accepted) >= self.min_cluster_size:
                cleaned[cid] = accepted
            elif accepted:
                # Adaptive fallback: keep even small clusters to avoid segments
                # with zero identifications (paper S.2.2.2 note on trade-off)
                cleaned[cid] = accepted

        return cleaned

    # ================================================================== #
    #  Aggregation: representative mode per cluster                        #
    # ================================================================== #

    @staticmethod
    def aggregate_cluster(poles: List[Pole]) -> ClusterResult:
        """
        Compute the representative modal parameters for a cluster.

        Uses the median frequency and damping (robust to remaining outliers)
        and the mode shape from the pole whose MAC to the cluster centroid
        shape is highest.

        Returns
        -------
        Dict with keys: 'freq', 'damping', 'shape', 'n_poles',
                        'freq_std', 'damping_std'
        """
        freqs = np.array([p["freq"] for p in poles])
        damps = np.array([p["damping"] for p in poles])
        shapes = [p["shape"] for p in poles]

        med_freq = float(np.median(freqs))
        med_damp = float(np.median(damps))
        std_freq = float(np.std(freqs))
        std_damp = float(np.std(damps))

        # Representative shape: average of real parts, normalised
        # (MAC-based selection would be ideal but mean-real is robust)
        mean_shape = np.mean(np.array([np.real(s) for s in shapes]), axis=0)
        norm = np.max(np.abs(mean_shape))
        if norm > 0:
            mean_shape = mean_shape / norm

        return {
            "freq": med_freq,
            "damping": med_damp,
            "shape": mean_shape,
            "freq_std": std_freq,
            "damping_std": std_damp,
            "n_poles": len(poles),
        }

    # ================================================================== #
    #  Main entry point                                                    #
    # ================================================================== #

    def process(
        self, poles_by_order: Dict[int, List[Pole]]
    ) -> Dict[int, ClusterResult]:
        """
        Run the full automated modal identification pipeline for one segment.

        Stages: stabilization -> clearance -> clustering -> MAD outlier removal
                -> aggregation.

        Parameters
        ----------
        poles_by_order : dict
            Raw output from SRIMIdentifier.identify().

        Returns
        -------
        Dict mapping  cluster_index (0-based) -> ClusterResult
            Clusters are sorted by median frequency (ascending).
            Returns {} if no physical modes could be identified.
        """
        # -- Stage 0: Stabilization diagram -------------------------------
        stable = self.build_stabilization(poles_by_order)

        if not stable:
            return {}

        # -- Stage 1: Clearance --------------------------------------------
        cleared = self.clear_diagram(stable)

        if not cleared:
            return {}

        # -- Stage 2: Hierarchical clustering -----------------------------
        raw_clusters = self.cluster_poles(cleared)

        if not raw_clusters:
            return {}

        # -- Stage 3: MAD outlier removal ---------------------------------
        clean_clusters = self.remove_outliers(raw_clusters)

        if not clean_clusters:
            return {}

        # -- Aggregation: one representative mode per cluster -------------
        results: Dict[int, ClusterResult] = {}
        cluster_modes = sorted(
            clean_clusters.items(),
            key=lambda kv: np.median([p["freq"] for p in kv[1]]),
        )

        for mode_idx, (_, poles) in enumerate(cluster_modes):
            res = self.aggregate_cluster(poles)
            res["cluster_id"] = mode_idx
            results[mode_idx] = res

        return results

    # ================================================================== #
    #  Diagnostics                                                         #
    # ================================================================== #

    def stabilization_diagram_data(
        self, poles_by_order: Dict[int, List[Pole]]
    ) -> Dict:
        """
        Return all raw poles grouped by stability type for plotting.

        Returns
        -------
        dict with keys:
            'all'      - every pole across all orders (List[Pole])
            'stable'   - type-3 stable poles
            'cleared'  - after MPC/positive-damping clearance
        """
        all_poles = [p for plist in poles_by_order.values() for p in plist]
        stable = self.build_stabilization(poles_by_order)
        cleared = self.clear_diagram(stable)

        return {
            "all": all_poles,
            "stable": stable,
            "cleared": cleared,
        }
