"""
srim_identifier.py
==================
SRIMIdentifier class: implements the output-only System Realization Using
Information Matrix (SRIM) algorithm following Tran & Ozer (2020), Sensors
20(17):4752, Equations (3)-(18).

Mathematical background
-----------------------
For a linear time-invariant (LTI) system excited by unmeasured ambient forces
(output-only scenario), the state-space model is:

    x_{k+1} = A x_k + w_k       (process noise w_k models ambient excitation)
    y_k      = C x_k + v_k       (measurement noise v_k)

where y_k in R^m is the m-channel acceleration vector at time step k.

**Output-only SRIM derivation:**

1.  Build the block Hankel matrix of outputs Y_i  (Eq. 4):
        Y_i = [[y_0, y_1, ..., y_{j-1}  ],   <- row block 0
               [y_1, y_2, ..., y_j      ],   <- row block 1
               [  ...                     ],
               [y_{i-1}, ..., y_{j+i-2}]]    <- row block i-1
    Shape: (i*m) * j,  where j = N_samples - i + 1

2.  Compute the symmetric autocorrelation matrix (Eq. 5):
        R_yy = (1/j) * Y_i * Y_i^T             shape: (i*m) * (i*m)

    In the noise-free case R_yy = O * R_xx * O^T,  where O is the
    extended observability matrix  O = [C; CA; CA^2; ...; CA^{i-1}].

3.  Singular value decomposition  (Eq. 13):
        R_yy = U * Σ * V^T  (symmetric, so U = V)
    For system order n, retain the top-n left singular vectors:
        O_n  =  U[:, 0:n]     shape: (i*m) * n

4.  Extract system matrices from O_n  (Eqs. 15-18):
        C  =  O_n[0:m, :]                       shape: m * n
        A  =  pinv(O_n[0:(i-1)*m, :]) @ O_n[m:i*m, :]   shape: n * n

5.  Eigendecompose A  ->  discrete-time poles mu_k  (Eq. 15):
        A = V_eig * diag(mu) * V_eig^{-1}

6.  Convert to continuous-time modal parameters  (Eqs. 16-18):
        lambda_k  = log(mu_k) / Deltat              (continuous-time eigenvalue)
        f_k  = |Im(lambda_k)| / (2pi)          [Hz]
        xi_k  = -Re(lambda_k) / |lambda_k|          (damping ratio, dimensionless)
        phi_k  = C * v_k                    (mode shape at sensor DOFs)

Note: Physical modes have positive damping (xi > 0) and real positive frequency.
Spurious numerical modes must be removed downstream (ModalClusterer).
"""

import numpy as np
from numpy.linalg import svd, eig, pinv
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Type alias: a single identified pole
#   freq     - natural frequency  [Hz]
#   damping  - damping ratio      [-]
#   shape    - mode shape         complex np.ndarray (m,)
#   order    - system order at which this pole was identified
# ---------------------------------------------------------------------------
Pole = Dict  # {'freq': float, 'damping': float, 'shape': np.ndarray, 'order': int}


class SRIMIdentifier:
    """
    Output-only SRIM system identification for one data segment.

    Parameters
    ----------
    fs : float
        Sampling frequency [Hz].
    i_factor : int
        Number of block rows in the Hankel matrix Y_i.  Must satisfy
        i_factor > n_max / m + 1  (paper constraint below Eq. 4).
        Default 10 is safe for n_max=40, m=7 sensors.
    system_orders : list of int or None
        System orders n to try.  If None, defaults to even integers from 2 to
        2*m*i_factor (up to max_order).
    max_order : int
        Upper bound on automatically generated system_orders.
    min_freq : float
        Lower cutoff for physical frequencies [Hz] (removes near-DC poles).
    max_freq : float
        Upper cutoff for physical frequencies [Hz] (removes aliased poles).
    """

    def __init__(
        self,
        fs: float = 128.0,
        i_factor: int = 10,
        system_orders: Optional[List[int]] = None,
        max_order: int = 40,
        min_freq: float = 0.2,
        max_freq: Optional[float] = None,
    ):
        self.fs = fs
        self.dt = 1.0 / fs
        self.i_factor = i_factor
        self.max_order = max_order
        self.min_freq = min_freq
        self.max_freq = max_freq if max_freq is not None else fs / 2.0

        # Will be resolved after we know the number of channels m
        self._system_orders_input = system_orders

    def _get_orders(self, m: int) -> List[int]:
        """
        Return list of system orders to evaluate.

        The maximum useful order is 2*i*m (the rank of R_yy), but we cap it
        at self.max_order.  We always use even orders so that complex conjugate
        pole pairs are complete.
        """
        if self._system_orders_input is not None:
            return [n for n in self._system_orders_input if n <= 2 * self.i_factor * m]

        upper = min(self.max_order, 2 * self.i_factor * m - 2)
        return list(range(2, upper + 1, 2))

    # ------------------------------------------------------------------ #
    #  Core Algorithm                                                      #
    # ------------------------------------------------------------------ #

    def _build_hankel(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Construct the block Hankel output matrix Y_i.

        Parameters
        ----------
        data : np.ndarray, shape (N, m)
            N time samples, m sensor channels.

        Returns
        -------
        Y_i : np.ndarray, shape (i*m, j)
        j   : int  (number of time columns)
        """
        N, m = data.shape
        i = self.i_factor
        j = N - i + 1   # number of usable time columns

        if j < 2 * i * m:
            raise ValueError(
                f"Segment too short: j={j} < 2*i*m={2*i*m}. "
                f"Increase segment_length_s or reduce i_factor."
            )

        # Allocate (i*m * j)
        Y = np.zeros((i * m, j), dtype=np.float64)

        for s in range(i):           # s = block row index (0-based)
            Y[s * m : (s + 1) * m, :] = data[s : s + j, :].T   # m * j slice

        return Y, j

    def _compute_poles(
        self, U_full: np.ndarray, n: int, m: int
    ) -> List[Pole]:
        """
        Extract continuous-time modal parameters for a given system order n.

        Uses the top-n left singular vectors of R_yy as the observability matrix.

        Parameters
        ----------
        U_full : np.ndarray, shape (i*m, i*m)
            Full left singular matrix from SVD of R_yy.
        n : int
            Desired system order.
        m : int
            Number of sensor channels.

        Returns
        -------
        List of Pole dicts (one per identified mode, both physical and spurious).
        """
        i = self.i_factor
        im = i * m

        # Observability matrix O ~= U[:, 0:n]  (first n columns)
        O_n = U_full[:, :n]   # (i*m) * n

        # -- Extract C and A ----------------------------------------------
        # C: top m rows of O_n
        C = O_n[:m, :]                          # m * n

        # A: from shift property O[m:] ~= O[:-m] * A
        O_upper = O_n[: (i - 1) * m, :]         # (i-1)*m * n
        O_lower = O_n[m:, :]                    # (i-1)*m * n

        # Least-squares solution: A = pinv(O_upper) @ O_lower
        # Uses SVD-based pseudo-inverse for numerical stability
        A = pinv(O_upper) @ O_lower             # n * n

        # -- Eigendecomposition of A --------------------------------------
        # mu_k are discrete-time poles (complex numbers)
        # V_eig columns are the corresponding discrete-time eigenvectors
        mu, V_eig = eig(A)                      # mu: (n,),  V_eig: (n,n)

        # -- Convert discrete -> continuous modal parameters ---------------
        poles: List[Pole] = []

        for k in range(n):
            mu_k = mu[k]
            v_k = V_eig[:, k]    # n-dimensional eigenvector

            # Continuous-time eigenvalue: lambda = log(mu) / Deltat
            # np.log returns principal value; for stable poles |mu| < 1 -> Re(lambda) < 0
            lam_k = np.log(mu_k) / self.dt

            # Natural frequency [Hz]
            freq_k = np.abs(np.imag(lam_k)) / (2.0 * np.pi)

            # Damping ratio (Eq. 17 via continuous-time eigenvalue)
            # xi = -Re(lambda) / |lambda|   ->  positive for stable modes
            abs_lam = np.abs(lam_k)
            if abs_lam < 1e-12:
                continue          # degenerate pole; skip

            damp_k = -np.real(lam_k) / abs_lam

            # Mode shape at sensor DOFs: phi = C * v_k   (Eq. 18)
            shape_k = C @ v_k   # complex (m,)

            # Frequency-band gating (remove DC and super-Nyquist artefacts)
            if not (self.min_freq <= freq_k <= self.max_freq):
                continue

            poles.append(
                {
                    "freq": float(freq_k),
                    "damping": float(damp_k),
                    "shape": shape_k,    # keep complex for MPC computation
                    "order": n,
                    "eigenvalue": mu_k,
                }
            )

        return poles

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def identify(self, data: np.ndarray) -> Dict[int, List[Pole]]:
        """
        Run output-only SRIM on one data segment across all system orders.

        Algorithm
        ---------
        1. Build block Hankel matrix Y_i  (single pass).
        2. Compute R_yy = (1/j) Y_i Y_i^T  (single large matrix multiply).
        3. SVD of R_yy -- **computed once**, then truncated to each order.
        4. For each order n: extract A, C from O_n -> eigendecompose -> poles.

        Parameters
        ----------
        data : np.ndarray, shape (N_samples, m)
            Zero-mean acceleration segment (one time window).

        Returns
        -------
        Dict mapping  system_order -> List[Pole]
            Poles are raw (include spurious, negative-damping modes).
            Pass to ModalClusterer for filtering and clustering.
        """
        N, m = data.shape
        orders = self._get_orders(m)

        if not orders:
            raise ValueError("No valid system orders. Check i_factor and max_order.")

        # -- Step 1: Block Hankel matrix -----------------------------------
        Y_i, j = self._build_hankel(data)          # (i*m) * j

        # -- Step 2: Autocorrelation matrix --------------------------------
        # R_yy = (1/j) * Y_i * Y_i^T
        # This is an (i*m) * (i*m) symmetric positive semi-definite matrix.
        # In the noise-free case R_yy = O * R_xx * O^T, where O is the
        # extended observability matrix.  SVD of R_yy recovers O up to a
        # right orthogonal factor that does not affect eigenvalues.
        R_yy = (Y_i @ Y_i.T) / j                   # (i*m) * (i*m)

        # -- Step 3: SVD (single computation for all orders) ---------------
        # R_yy = U * S * U^T  (symmetric -> U = V, S >= 0)
        # U_full[:, 0:n] gives the observability matrix for order n.
        U_full, S_vals, _ = svd(R_yy, full_matrices=True)

        # -- Step 4: Extract poles for each system order -------------------
        poles_by_order: Dict[int, List[Pole]] = {}

        for n in orders:
            if n >= Y_i.shape[0]:
                continue    # order exceeds Hankel rank; skip
            try:
                poles = self._compute_poles(U_full, n, m)
                poles_by_order[n] = poles
            except np.linalg.LinAlgError:
                poles_by_order[n] = []

        return poles_by_order

    def singular_values(self, data: np.ndarray) -> np.ndarray:
        """
        Return the singular values of R_yy for system order selection guidance.

        A sudden drop in singular values suggests the true system order.

        Parameters
        ----------
        data : np.ndarray, shape (N, m)

        Returns
        -------
        np.ndarray of singular values, sorted descending.
        """
        Y_i, j = self._build_hankel(data)
        R_yy = (Y_i @ Y_i.T) / j
        _, S, _ = svd(R_yy, full_matrices=False)
        return S
