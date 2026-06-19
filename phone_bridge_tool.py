"""
phone_bridge_tool.py
====================
Reconstruct Harness Bridge Deflections & Mode Shapes from Phone Accelerometer Data.

Each phone (harness1–harness7) is stored as a zip file in the Phones/ directory.
The zip contains Accelerometer.csv with columns: time, seconds_elapsed, z, y, x
Sample rate: 100 Hz (10 ms intervals) as configured by the PhyPhox app.

Sensor-to-joint mapping (same layout as harness_bridge_tool.py):
  harness1 → Joint 2
  harness2 → Joint 3
  harness3 → Joint 5
  harness4 → EXCLUDED (only ~54 s of data — phone stopped early)
  harness5 → Joint 7
  harness6 → Joint 9
  harness7 → Joint 10

Usage:
    python phone_bridge_tool.py
    python phone_bridge_tool.py --phones-dir Phones --axis z --zero-points 4 8 --fixed-points 1 11
    python phone_bridge_tool.py --method EFDD --num-modes 3
"""

import argparse
import os
import zipfile
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, csd, find_peaks
from scipy.interpolate import CubicSpline

# ─────────────────────────────────────────────────────────────────────────────
#  Ordered phone filenames → joint numbers
# ─────────────────────────────────────────────────────────────────────────────
PHONE_ORDER = [
    "harness1",  # Joint 2
    "harness2",  # Joint 3
    "harness3",  # Joint 5
    # harness4 excluded — only ~54 s recorded (phone stopped early)
    "harness5",  # Joint 7
    "harness6",  # Joint 9
    "harness7",  # Joint 10
]
SENSOR_JOINTS = [2, 3, 5, 7, 9, 10]

# Bridge geometry (joint spacings in metres)
SPACINGS = [1.82, 1.82, 1.82, 2.0, 2.0, 2.0, 2.0, 1.25, 1.25, 1.25]


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def joint_positions():
    """Return array of 11 joint x-positions (metres)."""
    pos = [0.0]
    for s in SPACINGS:
        pos.append(pos[-1] + s)
    return np.array(pos)


def cumtrapz(y, dx=1.0, initial=0):
    """Cumulative trapezoidal integration (avoids SciPy deprecation)."""
    res = np.zeros(len(y))
    res[1:] = np.cumsum((y[:-1] + y[1:]) / 2.0) * dx
    return res


def double_integrate(accel, fs, cutoff=0.1):
    """
    High-pass filter → integrate to velocity → high-pass filter →
    integrate to displacement → high-pass filter → detrend.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype='high', analog=False)

    accel_filt = filtfilt(b, a, accel)
    vel = cumtrapz(accel_filt, dx=1 / fs, initial=0)
    vel_filt = filtfilt(b, a, vel)
    disp = cumtrapz(vel_filt, dx=1 / fs, initial=0)
    disp_filt = filtfilt(b, a, disp)

    t = np.arange(len(disp_filt)) / fs
    poly = np.polyfit(t, disp_filt, 1)
    return disp_filt - np.polyval(poly, t)


def mac(u1, u2):
    """Modal Assurance Criterion between two complex vectors."""
    num = np.abs(np.conj(u1).T @ u2) ** 2
    den = (np.conj(u1).T @ u1) * (np.conj(u2).T @ u2)
    return num / den


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def find_zip(phones_dir: str, prefix: str) -> str:
    """Find the zip file whose name starts with *prefix* (case-insensitive)."""
    for fname in os.listdir(phones_dir):
        if fname.lower().startswith(prefix.lower()) and fname.endswith('.zip'):
            return os.path.join(phones_dir, fname)
    raise FileNotFoundError(
        f"No zip file starting with '{prefix}' found in '{phones_dir}'."
    )


def load_phone_data(phones_dir: str, axis: str = 'z') -> tuple[np.ndarray, float, list[str]]:
    """
    Load accelerometer data from all 7 phone zips.

    Returns
    -------
    data : ndarray, shape (N, 7)  – one column per sensor in SENSOR_JOINTS order
    fs   : float                  – sample rate (Hz), inferred from timestamps
    device_names : list[str]      – device name from each phone's Metadata.csv
    """
    series = []
    device_names = []
    raw_fs_list = []

    for prefix in PHONE_ORDER:
        zip_path = find_zip(phones_dir, prefix)
        print(f"  Loading {os.path.basename(zip_path)} …")
        with zipfile.ZipFile(zip_path) as z:
            # Device name from metadata
            try:
                with z.open('Metadata.csv') as mf:
                    meta_lines = mf.read().decode('utf-8').strip().split('\n')
                    if len(meta_lines) >= 2:
                        headers = [h.strip() for h in meta_lines[0].split(',')]
                        values  = [v.strip() for v in meta_lines[1].split(',')]
                        meta_d  = dict(zip(headers, values))
                        device_names.append(meta_d.get('device name', 'unknown'))
                    else:
                        device_names.append('unknown')
            except Exception:
                device_names.append('unknown')

            # Accelerometer
            with z.open('Accelerometer.csv') as af:
                df_acc = pd.read_csv(af)

        # Column names: time, seconds_elapsed, z, y, x  (order may vary → use names)
        if axis not in df_acc.columns:
            available = [c for c in df_acc.columns if c not in ('time', 'seconds_elapsed')]
            raise ValueError(
                f"Axis '{axis}' not found in Accelerometer.csv. "
                f"Available axes: {available}"
            )

        # Infer fs from timestamps (nanoseconds)
        timestamps_ns = df_acc['time'].values.astype(np.float64)
        dt_ns = np.median(np.diff(timestamps_ns))
        raw_fs = 1e9 / dt_ns
        raw_fs_list.append(raw_fs)

        series.append(df_acc[axis].values)

    # Use median inferred fs across phones, rounded to nearest integer
    fs = round(float(np.median(raw_fs_list)))
    print(f"  Inferred sample rate: {fs:.1f} Hz (median across phones)")

    # Align to the shortest recording
    min_len = min(len(s) for s in series)
    data = np.column_stack([s[:min_len] for s in series])
    print(f"  Data shape after alignment: {data.shape}  ({min_len / fs:.1f} s per channel)")

    return data, float(fs), device_names


# ─────────────────────────────────────────────────────────────────────────────
#  Bridge layout diagram
# ─────────────────────────────────────────────────────────────────────────────

def draw_bridge_layout(zero_joints, fixed_joints, sensor_joints, device_names, output_path):
    jp = joint_positions()

    fig, ax = plt.subplots(figsize=(16, 7))

    # Draw horizontal decks
    # Lower Deck (y = 0)
    ax.plot([jp[0], jp[-1]], [0, 0], color='#2c3e50', linewidth=4, zorder=1)
    ax.text(jp[-1] + 0.2, 0, "Lower Deck", va='center', ha='left', fontsize=10, fontweight='bold', color='#2c3e50')
    # Upper Deck (y = 1.5)
    ax.plot([jp[0], jp[-1]], [1.5, 1.5], color='#2980b9', linewidth=4, zorder=1)
    ax.text(jp[-1] + 0.2, 1.5, "Upper Deck", va='center', ha='left', fontsize=10, fontweight='bold', color='#2980b9')

    # Draw columns erected from joints 1, 4, 5, 6, 8, 11 (upwards to upper deck at y = 1.5)
    upper_col_joints = [1, 4, 5, 6, 8, 11]
    for joint in upper_col_joints:
        x = jp[joint - 1]
        ax.plot([x, x], [0, 1.5], color='#7f8c8d', linewidth=4, zorder=2)

    # Draw columns going to ground at joint 4 and joint 8 (downwards to y = -1.5)
    ground_col_joints = [4, 8]
    for joint in ground_col_joints:
        x = jp[joint - 1]
        ax.plot([x, x], [0, -1.5], color='#34495e', linewidth=5, zorder=2)
        # Draw foundation horizontal plate
        ax.plot([x - 0.4, x + 0.4], [-1.5, -1.5], color='#2c3e50', linewidth=3, zorder=3)
        # Draw soil hatch lines
        for h in np.linspace(-0.3, 0.3, 5):
            ax.plot([x + h, x + h - 0.1], [-1.5, -1.7], color='#7f8c8d', linewidth=1.5)
        stype = "Zero Point (Ground Support)" if joint in zero_joints else "Ground Support"
        ax.text(x, -1.9, f"Joint {joint}\n({stype})", ha='center', va='top', fontsize=9, color='#2c3e50', fontweight='bold')

    # Draw end supports directly at lower deck level for Joint 1 and Joint 11
    for joint in [1, 11]:
        x = jp[joint - 1]
        # Draw a support triangle underneath the node at y = -0.3
        ax.plot([x - 0.25, x + 0.25, x, x - 0.25], [-0.4, -0.4, 0, -0.4], color='#2c3e50', linewidth=2, zorder=3)
        # Draw ground line under the triangle
        ax.plot([x - 0.4, x + 0.4], [-0.4, -0.4], color='#34495e', linewidth=2, zorder=3)
        for h in np.linspace(-0.3, 0.3, 5):
            ax.plot([x + h, x + h - 0.08], [-0.4, -0.52], color='#7f8c8d', linewidth=1)
        stype = "Fixed Support" if joint in fixed_joints else "Support"
        ax.text(x, -0.7, f"Joint {joint}\n({stype})", ha='center', va='top', fontsize=9, color='#2c3e50', fontweight='bold')

    # Draw lower deck joints (nodes) - all uninstrumented (white/gray)
    for i, x in enumerate(jp):
        joint_num = i + 1
        ax.scatter(x, 0, s=600, color='#ecf0f1', edgecolors='#7f8c8d', linewidth=2, zorder=4)
        ax.text(x, 0, str(joint_num), ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50', zorder=5)

    # Draw upper deck joints (nodes) - sensors are red, column connections are dark gray, others are light gray
    for i, x in enumerate(jp):
        joint_num = i + 1
        is_sensor = joint_num in sensor_joints
        is_column = joint_num in upper_col_joints
        
        if is_sensor:
            face_color = '#e74c3c'
            edge_color = '#c0392b'
            textcolor = 'white'
        elif is_column:
            face_color = '#34495e'
            edge_color = '#2c3e50'
            textcolor = 'white'
        else:
            face_color = '#ecf0f1'
            edge_color = '#7f8c8d'
            textcolor = '#2c3e50'
            
        ax.scatter(x, 1.5, s=600, color=face_color, edgecolors=edge_color, linewidth=2, zorder=4)
        ax.text(x, 1.5, f"U{joint_num}", ha='center', va='center', fontsize=9, fontweight='bold', color=textcolor, zorder=5)

        if is_sensor:
            sensor_idx = sensor_joints.index(joint_num)
            dev = device_names[sensor_idx] if sensor_idx < len(device_names) else ''
            # Draw sensor indicator pointing to node
            ax.scatter(x, -0.4 if joint_num in [1, 11] else -0.5, marker='^', color='#e74c3c', s=120, zorder=3)
            ax.text(x, -0.65 if joint_num in [1, 11] else -0.75, f"📱 {dev}", ha='center', va='top',
                    fontsize=7.5, color='#e74c3c', fontweight='bold')

    # Draw spacings at the top of the upper deck
    for i in range(len(SPACINGS)):
        x_mid = (jp[i] + jp[i + 1]) / 2.0
        ax.annotate('', xy=(jp[i], 2.1), xytext=(jp[i + 1], 2.1),
                    arrowprops=dict(arrowstyle='<->', color='#7f8c8d', lw=1.5))
        ax.text(x_mid, 2.25, f"{SPACINGS[i]} m", ha='center', va='bottom', fontsize=10, color='#7f8c8d')

    ax.set_xlim(-1.5, jp[-1] + 2.0)
    ax.set_ylim(-2.5, 3.0)
    ax.axis('off')
    ax.set_title("Harness Bridge (Phone Sensors): 2-Deck Joint, Sensor & Column Layout",
                 fontsize=14, fontweight='bold', pad=20)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Layout sketch saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Modal Analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_modal_analysis(data, fs, sensor_joints, zero_joints, fixed_joints,
                       out_dir, nperseg=2048, num_modes=3, method="FDD"):
    """
    Operational Modal Analysis on phone accelerometer data.

    Parameters
    ----------
    data : ndarray (N, n_channels)
    """
    print(f"\nPerforming Operational Modal Analysis ({method} Method)…")
    n_channels = data.shape[1]

    # 1. CSD matrix
    f, _ = csd(data[:, 0], data[:, 0], fs=fs, nperseg=nperseg)
    n_freqs = len(f)
    G = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    for i in range(n_channels):
        for j in range(n_channels):
            _, temp = csd(data[:, i], data[:, j], fs=fs, nperseg=nperseg)
            G[:, i, j] = temp

    # 2. SVD of CSD matrix at each frequency
    singular_values  = np.zeros((n_freqs, n_channels))
    singular_vectors = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    for k in range(n_freqs):
        U, S, VH = np.linalg.svd(G[k, :, :])
        singular_values[k, :] = S
        singular_vectors[k, :, :] = U

    # 3. Peak detection on first SV (FDD reference)
    search_idx = (f >= 0.5) & (f <= fs / 2)
    f_search   = f[search_idx]
    sv1_search = singular_values[search_idx, 0]

    for prom in [0.3, 0.2, 0.1, 0.05, 0.01]:
        peaks, props = find_peaks(
            np.log10(np.maximum(sv1_search, 1e-30)),
            distance=max(1, int(0.5 / (f[1] - f[0]))),
            prominence=prom,
        )
        if len(peaks) >= num_modes:
            break

    peak_freqs       = f_search[peaks]
    peak_prominences = props['prominences']

    all_sorted = np.argsort(peak_prominences)[::-1]
    print(f"\nAll detected peaks (ranked by prominence):")
    for rank, i in enumerate(all_sorted):
        print(f"  #{rank + 1}: {peak_freqs[i]:.3f} Hz  (prominence={peak_prominences[i]:.3f})")

    sorted_idx    = np.argsort(peak_prominences)[::-1]
    top_peaks_fdd = sorted(peak_freqs[sorted_idx[:num_modes]])

    mode_freqs  = []
    mode_damps  = []
    mode_shapes = []

    # ── FDD ──────────────────────────────────────────────────────────────────
    if method == "FDD":
        for freq in top_peaks_fdd:
            f_idx    = np.argmin(np.abs(f - freq))
            u_vector = singular_vectors[f_idx, :, 0]
            max_idx  = np.argmax(np.abs(u_vector))
            u_rot    = u_vector * np.exp(-1j * np.angle(u_vector[max_idx]))
            real_u   = np.real(u_rot)
            nf = np.max(np.abs(real_u))
            if nf > 0:
                real_u /= nf
            mode_freqs.append(freq)
            mode_damps.append(None)
            mode_shapes.append(real_u)

    # ── EFDD ─────────────────────────────────────────────────────────────────
    elif method == "EFDD":
        print("Running Enhanced Frequency Domain Decomposition (EFDD)…")
        efdd_candidates = []
        for freq in top_peaks_fdd:
            f_idx = np.argmin(np.abs(f - freq))
            phi   = singular_vectors[f_idx, :, 0]

            # SDOF bell via MAC > 0.90
            sdof_idx = []
            for idx in range(f_idx, 0, -1):
                if mac(phi, singular_vectors[idx, :, 0]) > 0.90:
                    sdof_idx.append(idx)
                else:
                    break
            for idx in range(f_idx + 1, n_freqs):
                if mac(phi, singular_vectors[idx, :, 0]) > 0.90:
                    sdof_idx.append(idx)
                else:
                    break
            sdof_idx = sorted(sdof_idx)

            S_sdof = np.zeros(n_freqs)
            S_sdof[sdof_idx] = singular_values[sdof_idx, 0]

            G_double = np.zeros(nperseg, dtype=complex)
            G_double[:n_freqs] = S_sdof
            for i in range(1, n_freqs - 1):
                G_double[nperseg - i] = np.conj(S_sdof[i])
            R_tau = np.real(np.fft.ifft(G_double))

            decay = R_tau[:nperseg // 2]
            decay = decay / np.max(np.abs(decay))

            expected_period_samples = fs / freq
            damping      = None
            refined_freq = freq
            efdd_ok      = False

            if expected_period_samples >= 5:
                dist = max(5, int(expected_period_samples * 0.8))
                decay_peaks, _ = find_peaks(decay, distance=dist)
                valid_peaks = [p for p in decay_peaks if decay[p] > 0.1][:15]

                if len(valid_peaks) >= 3:
                    periods = np.diff(valid_peaks)
                    valid_periods = periods[
                        (periods >= expected_period_samples * 0.5) &
                        (periods <= expected_period_samples * 1.5)
                    ]
                    if len(valid_periods) >= 2:
                        avg_period = np.mean(valid_periods) / fs
                        fd   = 1.0 / avg_period
                        amps = decay[valid_peaks[:len(valid_periods) + 1]]
                        if np.all(amps > 0) and len(amps) >= 2:
                            x_fit = np.arange(len(amps))
                            y_fit = np.log(amps)
                            slope, _ = np.polyfit(x_fit, y_fit, 1)
                            delta = -slope
                            if delta > 0:
                                damp_est = delta / np.sqrt(4 * np.pi ** 2 + delta ** 2)
                                freq_est = fd / np.sqrt(max(1e-6, 1.0 - damp_est ** 2))
                                if abs(freq_est - freq) / freq < 0.15 and damp_est >= 1e-4:
                                    damping      = damp_est
                                    refined_freq = freq_est
                                    efdd_ok      = True

            if not efdd_ok:
                damping      = None
                refined_freq = freq

            max_idx  = np.argmax(np.abs(phi))
            phi_rot  = phi * np.exp(-1j * np.angle(phi[max_idx]))
            real_u   = np.real(phi_rot)
            nf = np.max(np.abs(real_u))
            if nf > 0:
                real_u /= nf

            efdd_candidates.append((refined_freq, damping, real_u, efdd_ok))

        # Deduplicate
        used_freqs = []
        for (rf, damp, shape, ok) in sorted(efdd_candidates, key=lambda x: x[0]):
            if all(abs(rf - uf) > 0.5 for uf in used_freqs):
                mode_freqs.append(rf)
                mode_damps.append(damp)
                mode_shapes.append(shape)
                used_freqs.append(rf)

    # ── SSI-COV ──────────────────────────────────────────────────────────────
    elif method == "SSI-COV":
        print("Running Stochastic Subspace Identification (SSI-COV) stabilisation loop…")
        y = data.T
        y = y - np.mean(y, axis=1, keepdims=True)

        N_full = y.shape[1]
        seg_len = min(40000, N_full)
        start_idx = max(0, N_full // 2 - seg_len // 2)
        y_seg = y[:, start_idx:start_idx + seg_len]
        N_seg = y_seg.shape[1]

        p_lags = 60
        R_cov = []
        for lag in range(2 * p_lags):
            if lag == 0:
                cov = (y_seg @ y_seg.T) / N_seg
            else:
                cov = (y_seg[:, lag:] @ y_seg[:, :-lag].T) / N_seg
            R_cov.append(cov)

        H_toep = np.zeros((p_lags * n_channels, p_lags * n_channels))
        for i in range(p_lags):
            for j in range(p_lags):
                H_toep[i * n_channels:(i + 1) * n_channels,
                       j * n_channels:(j + 1) * n_channels] = R_cov[i + j + 1]

        U_t, S_t, _ = np.linalg.svd(H_toep)

        orders    = range(10, 81, 2)
        all_poles = []
        for order in orders:
            U_n = U_t[:, :order]
            S_n = S_t[:order]
            O_n      = U_n @ np.diag(np.sqrt(S_n))
            O_n_down = O_n[:-n_channels, :]
            O_n_up   = O_n[n_channels:, :]
            A_mat    = np.linalg.pinv(O_n_down) @ O_n_up
            C_mat    = O_n[:n_channels, :]
            eigenvals, eigenvects = np.linalg.eig(A_mat)
            dt = 1.0 / fs
            poles_ct = np.log(eigenvals) / dt

            for idx, pole in enumerate(poles_ct):
                freq    = np.abs(pole) / (2 * np.pi)
                damping = -np.real(pole) / np.abs(pole)
                if 0.5 <= freq <= fs / 2 and 0 < damping <= 0.15:
                    shape = C_mat @ eigenvects[:, idx]
                    max_idx   = np.argmax(np.abs(shape))
                    shape     = shape * np.exp(-1j * np.angle(shape[max_idx]))
                    real_shape = np.real(shape)
                    norm_shape = (real_shape / np.max(np.abs(real_shape))
                                  if np.max(np.abs(real_shape)) > 0 else real_shape)
                    all_poles.append({'freq': freq, 'damping': damping, 'shape': norm_shape})

        clusters = []
        for pole in all_poles:
            placed = False
            for c in clusters:
                avg_freq = np.mean([p['freq'] for p in c])
                if np.abs(pole['freq'] - avg_freq) < 0.25:
                    c.append(pole)
                    placed = True
                    break
            if not placed:
                clusters.append([pole])

        stable_clusters = [c for c in clusters if len(c) >= 10]
        stable_clusters = sorted(stable_clusters,
                                 key=lambda c: np.mean([p['freq'] for p in c]))

        matched_clusters     = []
        used_cluster_indices = set()
        for target_f in top_peaks_fdd:
            best_idx  = -1
            best_dist = 999.0
            for idx, c in enumerate(stable_clusters):
                avg_freq = np.mean([p['freq'] for p in c])
                dist     = np.abs(avg_freq - target_f)
                if dist < 1.5 and dist < best_dist and idx not in used_cluster_indices:
                    best_dist = dist
                    best_idx  = idx
            if best_idx != -1:
                matched_clusters.append(stable_clusters[best_idx])
                used_cluster_indices.add(best_idx)

        if len(matched_clusters) < num_modes:
            remaining = [c for idx, c in enumerate(stable_clusters)
                         if idx not in used_cluster_indices]
            remaining = sorted(remaining, key=lambda c: len(c), reverse=True)
            for c in remaining:
                if len(matched_clusters) >= num_modes:
                    break
                matched_clusters.append(c)

        matched_clusters = sorted(matched_clusters,
                                  key=lambda c: np.mean([p['freq'] for p in c]))
        for c in matched_clusters:
            freqs = [p['freq'] for p in c]
            damps = [p['damping'] for p in c]
            avg_freq = np.mean(freqs)
            avg_damp = np.mean(damps)
            best_pole = c[np.argmin([np.abs(p['freq'] - avg_freq) for p in c])]
            mode_freqs.append(avg_freq)
            mode_damps.append(max(0.0, avg_damp))
            mode_shapes.append(best_pole['shape'])

    print(f"\nIdentified Top {len(mode_freqs)} Natural Frequencies ({method}):")
    for idx, freq in enumerate(mode_freqs):
        damp_str = (f" | Damping: {mode_damps[idx] * 100:.2f}%"
                    if mode_damps[idx] is not None
                    else " | Damping: N/A (FDD fallback)")
        print(f"  Mode {idx + 1}: {freq:.3f} Hz{damp_str}")

    # ── Singular Value Plot ──────────────────────────────────────────────────
    sv_plot_path = os.path.join(out_dir, "phone_fdd_sv_plot.png")
    plt.figure(figsize=(12, 7))
    for i in range(min(3, n_channels)):
        plt.semilogy(f, singular_values[:, i], label=f"SV {i + 1}", linewidth=1.5)
    for idx, freq in enumerate(mode_freqs):
        f_idx = np.argmin(np.abs(f - freq))
        val   = singular_values[f_idx, 0]
        plt.scatter(freq, val, color='red', s=80, zorder=5)
        plt.axvline(freq, color='red', linestyle='--', alpha=0.5, linewidth=1.2)
        plt.text(freq, val * 1.5, f"Mode {idx + 1}\n{freq:.3f} Hz",
                 ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    plt.xlim(0, fs / 2)
    plt.xlabel("Frequency (Hz)", fontsize=11)
    plt.ylabel("Singular Value Amplitude (PSD)", fontsize=11)
    plt.title(f"Phone FDD Singular Value Plot ({method} Modes Highlighted)",
              fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(sv_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Phone FDD SV plot saved to: {sv_plot_path}")

    # ── Mode Shape Plots ─────────────────────────────────────────────────────
    jp           = joint_positions()
    all_supports = set(zero_joints + fixed_joints)
    num_modes_plot = len(mode_freqs)
    fig, axes = plt.subplots(num_modes_plot, 1,
                              figsize=(12, 3 * num_modes_plot + 1), sharex=True)
    if num_modes_plot == 1:
        axes = [axes]

    colors = ['#2980b9', '#27ae60', '#8e44ad', '#d35400', '#f39c12', '#16a085', '#2c3e50']

    for mode_idx, (freq, shape) in enumerate(zip(mode_freqs, mode_shapes)):
        f_idx    = np.argmin(np.abs(f - freq))
        u_vector = shape
        max_val_idx = np.argmax(np.abs(u_vector))
        ref_phase   = np.angle(u_vector[max_val_idx])
        rotated_u   = u_vector * np.exp(-1j * ref_phase)
        real_u      = np.real(rotated_u)
        nf = np.max(np.abs(real_u))
        if nf > 0:
            real_u /= nf

        known_x = []
        known_y = []
        sensor_x = []
        sensor_y = []

        for j in range(1, 12):
            x = jp[j - 1]
            if j in all_supports:
                known_x.append(x)
                known_y.append(0.0)
            elif j in sensor_joints:
                s_idx = sensor_joints.index(j)
                val   = real_u[s_idx]
                known_x.append(x)
                known_y.append(val)
                sensor_x.append(x)
                sensor_y.append(val)

        sort_idx = np.argsort(known_x)
        known_x  = np.array(known_x)[sort_idx]
        known_y  = np.array(known_y)[sort_idx]

        cs     = CubicSpline(known_x, known_y, bc_type='natural')
        x_fine = np.linspace(0, jp[-1], 200)
        y_fine = cs(x_fine)

        ax    = axes[mode_idx]
        color = colors[mode_idx % len(colors)]

        ax.plot(x_fine, y_fine, color=color, linewidth=3, label=f"Mode Shape {mode_idx + 1}")
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)

        for j in all_supports:
            if j < 1 or j > 11:
                continue
            ax.plot(jp[j - 1], 0, marker='^', color='#2c3e50', markersize=12, zorder=5)

        ax.scatter(sensor_x, sensor_y, color='#e74c3c', s=80, zorder=6)
        s_joints_no_sup = [j for j in sensor_joints if j not in all_supports]
        for sx, sy, sj in zip(sensor_x, sensor_y, s_joints_no_sup):
            va  = 'bottom' if sy >= 0 else 'top'
            off = 0.08 if sy >= 0 else -0.08
            ax.text(sx, sy + off, f"J{sj}\n{sy:.2f}",
                    ha='center', va=va, fontsize=8, color='#c0392b', fontweight='bold')

        damp_str = (f" | ζ={mode_damps[mode_idx] * 100:.2f}%"
                    if mode_damps[mode_idx] is not None else "")
        ax.set_title(f"Mode {mode_idx + 1}: {freq:.3f} Hz{damp_str}",
                     fontsize=11, fontweight='bold', color=color, loc='left')
        ax.set_ylabel("Norm. Amplitude", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(-1.3, 1.3)

    axes[-1].set_xlabel("Bridge Span (metres)", fontsize=11)
    axes[0].plot([], [], marker='^', color='#2c3e50', ls='', markersize=10, label="Supports")
    axes[0].scatter([], [], color='#e74c3c', label="Phone Sensors")
    axes[0].legend(loc='upper right', ncol=3)
    fig.suptitle(f"Bridge Mode Shapes — Phone Accelerometers ({method})",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    mode_plot_path = os.path.join(out_dir, "phone_bridge_mode_shapes.png")
    plt.savefig(mode_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Phone mode shapes plot saved to: {mode_plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Displacement reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_displacements(data, fs, sensor_joints, zero_joints, fixed_joints,
                               out_dir, cutoff=0.1):
    """Double-integrate each phone channel to get displacement (metres)."""
    print("\nCalculating displacements via double integration…")
    jp           = joint_positions()
    all_supports = set(zero_joints + fixed_joints)

    displacements = {}
    for ch_idx, joint in enumerate(sensor_joints):
        accel      = data[:, ch_idx]
        accel_dyn  = accel - np.mean(accel)
        disp       = double_integrate(accel_dyn, fs, cutoff=cutoff)
        disp_m     = disp * 9.81          # g → m/s² → after integration → m
        displacements[joint] = disp_m

    # Time-series plot
    t_series_path = os.path.join(out_dir, "phone_displacements_time_series.png")
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(len(data)) / fs / 60.0  # minutes
    for joint in sensor_joints:
        plt.plot(time_axis, displacements[joint] * 1000,
                 label=f"Joint {joint} (Phone)", alpha=0.8)
    plt.xlabel("Time (minutes)", fontsize=11)
    plt.ylabel("Displacement (mm)", fontsize=11)
    plt.title("Phone Sensor Reconstructed Displacements (Double Integrated & Filtered)",
              fontsize=13, fontweight='bold')
    plt.legend(ncol=4, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(t_series_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Time-series plot saved to: {t_series_path}")

    # Peak vibration deflection profile
    N = len(data)
    rms_disp = np.zeros(N)
    for t_idx in range(N):
        vals = [displacements[j][t_idx] for j in sensor_joints]
        rms_disp[t_idx] = np.sqrt(np.mean(np.array(vals) ** 2))

    max_t_idx   = np.argmax(rms_disp)
    max_time_s  = max_t_idx / fs
    print(f"Maximum vibration at index {max_t_idx} ({max_time_s:.2f} s)")

    known_x = []
    known_y = []
    for j in range(1, 12):
        x = jp[j - 1]
        if j in all_supports:
            known_x.append(x)
            known_y.append(0.0)
        elif j in sensor_joints:
            known_x.append(x)
            known_y.append(displacements[j][max_t_idx])

    sort_idx = np.argsort(known_x)
    known_x  = np.array(known_x)[sort_idx]
    known_y  = np.array(known_y)[sort_idx]

    cs     = CubicSpline(known_x, known_y, bc_type='natural')
    x_fine = np.linspace(0, jp[-1], 200)
    y_fine = cs(x_fine)

    profile_path = os.path.join(out_dir, "phone_bridge_deflection_profile.png")
    plt.figure(figsize=(14, 5))
    plt.plot(x_fine, y_fine * 1000, color='#2980b9', linewidth=3,
             label="Reconstructed Deflection Shape")
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)

    for j in all_supports:
        if j < 1 or j > 11:
            continue
        x = jp[j - 1]
        lbl = "Support" if j == list(all_supports)[0] else ""
        plt.plot(x, 0, marker='^', color='#2c3e50', markersize=15, zorder=5, label=lbl)
        plt.text(x, -0.15, f"J{j}", ha='center', va='top', fontsize=9, fontweight='bold')

    for joint in sensor_joints:
        x = jp[joint - 1]
        if joint in all_supports:
            continue
        val = displacements[joint][max_t_idx] * 1000
        lbl = "Phone Sensor" if joint == sensor_joints[0] else ""
        plt.scatter(x, val, color='#e74c3c', s=100, zorder=6, label=lbl)
        va  = 'bottom' if val >= 0 else 'top'
        off = 0.15 if val >= 0 else -0.15
        plt.text(x, val + off, f"J{joint}\n{val:.2f} mm",
                 ha='center', va=va, fontsize=8, color='#c0392b', fontweight='bold')

    plt.xlabel("Bridge Span (metres)", fontsize=11)
    plt.ylabel("Deflection (mm)", fontsize=11)
    plt.title(f"Phone Sensor Bridge Deflection Profile at t = {max_time_s:.2f} s",
              fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(profile_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Deflection profile saved to: {profile_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct Harness Bridge Deflections & Mode Shapes from Phone Accelerometer Data."
    )
    parser.add_argument("--phones-dir", default="Phones",
                        help="Directory containing harness1–harness7 zip files (default: Phones).")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z",
                        help="Accelerometer axis to use for vertical vibration (default: z).")
    parser.add_argument("--zero-points", nargs="*", type=int, default=[4, 8],
                        help="Joint numbers acting as Zero Points (default: 4 8).")
    parser.add_argument("--fixed-points", nargs="*", type=int, default=[1, 11],
                        help="Joint numbers acting as Fixed Points (default: 1 11).")
    parser.add_argument("--cutoff", type=float, default=0.1,
                        help="High-pass filter cutoff frequency in Hz (default: 0.1).")
    parser.add_argument("--nperseg", type=int, default=2048,
                        help="Window size for CSD estimation (default: 2048).")
    parser.add_argument("--method", choices=["FDD", "EFDD", "SSI-COV"], default="FDD",
                        help="Modal analysis method: FDD, EFDD, or SSI-COV (default: FDD).")
    parser.add_argument("--num-modes", type=int, default=3,
                        help="Number of modes to extract (default: 3).")
    parser.add_argument("--out-dir", default=".",
                        help="Base directory for output files (default: current directory).")

    args = parser.parse_args()

    # Auto-name output subdirectory
    zp_str   = "-".join(map(str, args.zero_points))
    fp_str   = "-".join(map(str, args.fixed_points))
    auto_dir = f"phone_z{zp_str}_f{fp_str}_n{args.num_modes}_{args.method}"
    out_dir  = os.path.abspath(os.path.join(args.out_dir, auto_dir))
    os.makedirs(out_dir, exist_ok=True)

    phones_dir = os.path.abspath(args.phones_dir)
    if not os.path.isdir(phones_dir):
        print(f"Error: Phones directory '{phones_dir}' not found.")
        return

    print(f"Phones directory : {phones_dir}")
    print(f"Output directory : {out_dir}")
    print(f"Axis             : {args.axis}")
    print(f"Zero joints      : {args.zero_points}")
    print(f"Fixed joints     : {args.fixed_points}")
    print(f"Method           : {args.method}  |  Modes: {args.num_modes}")

    # 1. Load data
    print("\nLoading phone data…")
    data, fs, device_names = load_phone_data(phones_dir, axis=args.axis)
    print(f"Sample rate: {fs} Hz")

    # 2. Bridge layout diagram
    layout_path = os.path.join(out_dir, "phone_bridge_layout.png")
    draw_bridge_layout(args.zero_points, args.fixed_points,
                       SENSOR_JOINTS, device_names, layout_path)

    # 3. Modal analysis
    run_modal_analysis(data, fs, SENSOR_JOINTS, args.zero_points, args.fixed_points,
                       out_dir, nperseg=args.nperseg,
                       num_modes=args.num_modes, method=args.method)

    # 4. Displacement reconstruction
    reconstruct_displacements(data, fs, SENSOR_JOINTS, args.zero_points, args.fixed_points,
                               out_dir, cutoff=args.cutoff)

    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
