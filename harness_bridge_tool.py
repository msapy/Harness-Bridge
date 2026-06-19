import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, csd, find_peaks
from scipy.interpolate import CubicSpline

# Custom cumulative trapezoidal integration using numpy to avoid deprecation issues in newer SciPy versions
def cumtrapz(y, dx=1.0, initial=0):
    res = np.zeros(len(y))
    res[1:] = np.cumsum((y[:-1] + y[1:]) / 2.0) * dx
    return res

def double_integrate(accel, fs, cutoff=0.1):
    # High-pass filter parameter (Butterworth 4th-order)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype='high', analog=False)
    
    # 1. Filter raw acceleration
    accel_filt = filtfilt(b, a, accel)
    
    # 2. Integrate to velocity and filter
    vel = cumtrapz(accel_filt, dx=1/fs, initial=0)
    vel_filt = filtfilt(b, a, vel)
    
    # 3. Integrate to displacement and filter
    disp = cumtrapz(vel_filt, dx=1/fs, initial=0)
    disp_filt = filtfilt(b, a, disp)
    
    # Linear detrending to eliminate remaining drift
    t = np.arange(len(disp_filt)) / fs
    poly = np.polyfit(t, disp_filt, 1)
    disp_clean = disp_filt - np.polyval(poly, t)
    
    return disp_clean

def draw_bridge_layout(zero_joints, fixed_joints, sensor_joints, output_path):
    spacings = [1.82, 1.82, 1.82, 2.0, 2.0, 2.0, 2.0, 1.25, 1.25, 1.25]
    joint_positions = [0.0]
    for s in spacings:
        joint_positions.append(joint_positions[-1] + s)
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Draw horizontal decks
    # Lower Deck (y = 0)
    ax.plot([joint_positions[0], joint_positions[-1]], [0, 0], color='#2c3e50', linewidth=4, zorder=1)
    ax.text(joint_positions[-1] + 0.2, 0, "Lower Deck", va='center', ha='left', fontsize=10, fontweight='bold', color='#2c3e50')
    # Upper Deck (y = 1.5)
    ax.plot([joint_positions[0], joint_positions[-1]], [1.5, 1.5], color='#2980b9', linewidth=4, zorder=1)
    ax.text(joint_positions[-1] + 0.2, 1.5, "Upper Deck", va='center', ha='left', fontsize=10, fontweight='bold', color='#2980b9')
    
    # Draw columns erected from joints 1, 4, 5, 6, 8, 11 (upwards to upper deck at y = 1.5)
    upper_col_joints = [1, 4, 5, 6, 8, 11]
    for joint in upper_col_joints:
        x = joint_positions[joint - 1]
        ax.plot([x, x], [0, 1.5], color='#7f8c8d', linewidth=4, zorder=2)
        
    # Draw columns going to ground at joint 4 and joint 8 (downwards to y = -1.5)
    ground_col_joints = [4, 8]
    for joint in ground_col_joints:
        x = joint_positions[joint - 1]
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
        x = joint_positions[joint - 1]
        # Draw a support triangle underneath the node at y = -0.3
        ax.plot([x - 0.25, x + 0.25, x, x - 0.25], [-0.4, -0.4, 0, -0.4], color='#2c3e50', linewidth=2, zorder=3)
        # Draw ground line under the triangle
        ax.plot([x - 0.4, x + 0.4], [-0.4, -0.4], color='#34495e', linewidth=2, zorder=3)
        for h in np.linspace(-0.3, 0.3, 5):
            ax.plot([x + h, x + h - 0.08], [-0.4, -0.52], color='#7f8c8d', linewidth=1)
        stype = "Fixed Support" if joint in fixed_joints else "Support"
        ax.text(x, -0.7, f"Joint {joint}\n({stype})", ha='center', va='top', fontsize=9, color='#2c3e50', fontweight='bold')
        
    # Draw lower deck joints (nodes) - all uninstrumented (white/gray)
    for i, x in enumerate(joint_positions):
        joint_num = i + 1
        ax.scatter(x, 0, s=600, color='#ecf0f1', edgecolors='#7f8c8d', linewidth=2, zorder=4)
        ax.text(x, 0, str(joint_num), ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50', zorder=5)

    # Draw upper deck joints (nodes) - sensors are red, column connections are dark gray, others are light gray
    for i, x in enumerate(joint_positions):
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
            # Draw sensor indicator pointing to node on top deck
            ax.scatter(x, 1.9, marker='v', color='#e74c3c', s=120, zorder=3)
            ax.text(x, 2.1, "SENSOR", ha='center', va='bottom', fontsize=8, color='#e74c3c', fontweight='bold')

    # Draw spacings at the top of the upper deck
    for i in range(len(spacings)):
        x_mid = (joint_positions[i] + joint_positions[i+1]) / 2.0
        ax.annotate('', xy=(joint_positions[i], 2.4), xytext=(joint_positions[i+1], 2.4),
                    arrowprops=dict(arrowstyle='<->', color='#7f8c8d', lw=1.5))
        ax.text(x_mid, 2.5, f"{spacings[i]} m", ha='center', va='bottom', fontsize=10, color='#7f8c8d')
        
    ax.set_xlim(-1.5, joint_positions[-1] + 2.0)
    ax.set_ylim(-2.5, 3.2)
    ax.axis('off')
    ax.set_title("Harness Bridge: 2-Deck Joint, Upper Sensor, and Column Layout", fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Layout sketch saved to: {output_path}")

def mac(u1, u2):
    num = np.abs(np.conj(u1).T @ u2)**2
    den = (np.conj(u1).T @ u1) * (np.conj(u2).T @ u2)
    return num / den

def run_modal_analysis(df, fs, sensor_cols, sensor_joints, zero_joints, fixed_joints, out_dir, nperseg=4096, num_modes=3, method="FDD"):
    print(f"Performing Operational Modal Analysis ({method} Method)...")
    n_channels = len(sensor_cols)
    
    # 1. Compute CSD matrix (always computed for plotting reference SV spectrum)
    f, _ = csd(df[sensor_cols[0]].values, df[sensor_cols[0]].values, fs=fs, nperseg=nperseg)
    n_freqs = len(f)
    G = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    for i in range(n_channels):
        for j in range(n_channels):
            _, temp = csd(df[sensor_cols[i]].values, df[sensor_cols[j]].values, fs=fs, nperseg=nperseg)
            G[:, i, j] = temp
            
    singular_values = np.zeros((n_freqs, n_channels))
    singular_vectors = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    for k in range(n_freqs):
        if np.any(np.isnan(G[k, :, :])) or np.any(np.isinf(G[k, :, :])):
            continue
        try:
            U, S, VH = np.linalg.svd(G[k, :, :])
            singular_values[k, :] = S
            singular_vectors[k, :, :] = U
        except np.linalg.LinAlgError:
            continue

    # Detect reference peaks using FDD (always useful for plotting and initial reference)
    search_idx = (f >= 0.5) & (f <= fs / 2)  # Full Nyquist range
    f_search = f[search_idx]
    sv1_search = singular_values[search_idx, 0]

    # Adaptive prominence threshold: start at 0.3, lower until we have enough peaks
    min_prominence = 0.3
    safe_sv1_search = np.maximum(sv1_search, 1e-12)
    for prom in [0.3, 0.2, 0.1, 0.05, 0.01]:
        peaks, props = find_peaks(np.log10(safe_sv1_search), distance=int(0.5 / (f[1] - f[0])), prominence=prom)
        if len(peaks) >= num_modes:
            min_prominence = prom
            break
    else:
        # Use whatever we found at lowest threshold
        peaks, props = find_peaks(np.log10(safe_sv1_search), distance=int(0.5 / (f[1] - f[0])), prominence=0.01)

    peak_freqs = f_search[peaks]
    peak_prominences = props['prominences']

    # Print ALL detected peaks ranked by prominence so user can see the full picture
    all_sorted = np.argsort(peak_prominences)[::-1]
    print(f"\nAll detected peaks (prominence threshold={min_prominence:.2f}, ranked by prominence):")
    for rank, i in enumerate(all_sorted):
        print(f"  #{rank+1}: {peak_freqs[i]:.3f} Hz  (prominence={peak_prominences[i]:.3f})")

    # Select top num_modes peaks by prominence
    sorted_idx = np.argsort(peak_prominences)[::-1]
    top_peaks_fdd = sorted(peak_freqs[sorted_idx[:num_modes]])

    # Populate these based on selected method
    mode_freqs = []
    mode_damps = [] 
    mode_shapes = [] 
    
    if method == "FDD":
        for freq in top_peaks_fdd:
            f_idx = np.argmin(np.abs(f - freq))
            u_vector = singular_vectors[f_idx, :, 0]
            max_idx = np.argmax(np.abs(u_vector))
            u_rot = u_vector * np.exp(-1j * np.angle(u_vector[max_idx]))
            real_u = np.real(u_rot)
            norm_factor = np.max(np.abs(real_u))
            if norm_factor > 0:
                real_u = real_u / norm_factor
            mode_freqs.append(freq)
            mode_damps.append(None)
            mode_shapes.append(real_u)
            
    elif method == "EFDD":
        print("Running Enhanced Frequency Domain Decomposition (EFDD) for refined frequency and damping...")
        efdd_candidates = []
        for freq in top_peaks_fdd:
            f_idx = np.argmin(np.abs(f - freq))
            phi = singular_vectors[f_idx, :, 0]
            
            # Identify SDOF bell: MAC > 0.90
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
            
            # Inverse FFT to get Autocorrelation decay curve
            G_double = np.zeros(nperseg, dtype=complex)
            G_double[:n_freqs] = S_sdof
            for i in range(1, n_freqs - 1):
                G_double[nperseg - i] = np.conj(S_sdof[i])
            R_tau = np.real(np.fft.ifft(G_double))
            
            decay = R_tau[:nperseg // 2]
            decay = decay / np.max(np.abs(decay))
            
            expected_period_samples = fs / freq
            damping = None
            refined_freq = freq  # default fallback to FDD peak
            efdd_ok = False
            # EFDD needs at least 5 samples per cycle to be reliable (avoids high-freq aliasing)
            if expected_period_samples < 5:
                # Too close to Nyquist - fall back to FDD
                efdd_ok = False
            else:
                dist = max(5, int(expected_period_samples * 0.8))
                decay_peaks, _ = find_peaks(decay, distance=dist)
                valid_peaks = [p for p in decay_peaks if decay[p] > 0.1][:15]
                
                if len(valid_peaks) >= 3:
                    periods = np.diff(valid_peaks)
                    valid_periods = periods[(periods >= expected_period_samples * 0.5) & (periods <= expected_period_samples * 1.5)]
                    if len(valid_periods) >= 2:
                        avg_period = np.mean(valid_periods) / fs
                        fd = 1.0 / avg_period
                        amps = decay[valid_peaks[:len(valid_periods)+1]]
                        if np.all(amps > 0) and len(amps) >= 2:
                            x_fit = np.arange(len(amps))
                            y_fit = np.log(amps)
                            slope, _ = np.polyfit(x_fit, y_fit, 1)
                            delta = -slope
                            # Require actual decay (slope must be negative i.e. delta > 0)
                            # and damping must be at least 0.001% to be meaningful
                            if delta > 0:
                                damp_est = delta / np.sqrt(4 * np.pi**2 + delta**2)
                                freq_est = fd / np.sqrt(max(1e-6, 1.0 - damp_est**2))
                                # Validate: refined freq must be within 15% of original FDD peak
                                if abs(freq_est - freq) / freq < 0.15 and damp_est >= 1e-4:
                                    damping = damp_est
                                    refined_freq = freq_est
                                    efdd_ok = True
                            
            if not efdd_ok:
                # Fall back to FDD frequency with no damping estimate
                damping = None
                refined_freq = freq
            
            # Mode shape from peak singular vector
            max_idx = np.argmax(np.abs(phi))
            phi_rot = phi * np.exp(-1j * np.angle(phi[max_idx]))
            real_u = np.real(phi_rot)
            norm_factor = np.max(np.abs(real_u))
            if norm_factor > 0:
                real_u = real_u / norm_factor
                
            efdd_candidates.append((refined_freq, damping, real_u, efdd_ok))
        
        # Deduplicate: remove modes within 0.5 Hz of a higher-prominence mode
        used_freqs = []
        for (rf, damp, shape, ok) in sorted(efdd_candidates, key=lambda x: x[0]):
            if all(abs(rf - uf) > 0.5 for uf in used_freqs):
                mode_freqs.append(rf)
                mode_damps.append(damp)
                mode_shapes.append(shape)
                used_freqs.append(rf)
            
    elif method == "SSI-COV":
        print("Running Stochastic Subspace Identification (SSI-COV) stabilization loop...")
        y = df[sensor_cols].values.T
        y = y - np.mean(y, axis=1, keepdims=True)
        
        # Segment 40,000 samples around peak vibration
        max_vib_idx = 207415
        start_idx = max(0, max_vib_idx - 20000)
        end_idx = min(y.shape[1], max_vib_idx + 20000)
        y_seg = y[:, start_idx:end_idx]
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
                H_toep[i*n_channels:(i+1)*n_channels, j*n_channels:(j+1)*n_channels] = R_cov[i + j + 1]
                
        U_t, S_t, VH_t = np.linalg.svd(H_toep)
        
        orders = range(10, 81, 2)
        all_poles = []
        for order in orders:
            U_n = U_t[:, :order]
            S_n = S_t[:order]
            O_n = U_n @ np.diag(np.sqrt(S_n))
            O_n_down = O_n[:-n_channels, :]
            O_n_up = O_n[n_channels:, :]
            A_mat = np.linalg.pinv(O_n_down) @ O_n_up
            C_mat = O_n[:n_channels, :]
            eigenvals, eigenvects = np.linalg.eig(A_mat)
            dt = 1.0 / fs
            poles_ct = np.log(eigenvals) / dt
            
            for idx, pole in enumerate(poles_ct):
                freq = np.abs(pole) / (2 * np.pi)
                damping = -np.real(pole) / np.abs(pole)
                if 0.5 <= freq <= fs / 2 and 0 < damping <= 0.15:
                    shape = C_mat @ eigenvects[:, idx]
                    max_idx = np.argmax(np.abs(shape))
                    shape = shape * np.exp(-1j * np.angle(shape[max_idx]))
                    real_shape = np.real(shape)
                    norm_shape = real_shape / np.max(np.abs(real_shape)) if np.max(np.abs(real_shape)) > 0 else real_shape
                    all_poles.append({
                        'freq': freq,
                        'damping': damping,
                        'shape': norm_shape
                    })
                    
        # Cluster stable poles
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
        stable_clusters = sorted(stable_clusters, key=lambda c: np.mean([p['freq'] for p in c]))
        
        # Match closest stable clusters to the target FDD peak frequencies
        matched_clusters = []
        used_cluster_indices = set()
        for target_f in top_peaks_fdd:
            best_idx = -1
            best_dist = 999.0
            for idx, c in enumerate(stable_clusters):
                avg_freq = np.mean([p['freq'] for p in c])
                dist = np.abs(avg_freq - target_f)
                if dist < 1.5 and dist < best_dist and idx not in used_cluster_indices:
                    best_dist = dist
                    best_idx = idx
            if best_idx != -1:
                matched_clusters.append(stable_clusters[best_idx])
                used_cluster_indices.add(best_idx)
                
        # Fill in remaining modes if matched_clusters is less than num_modes
        if len(matched_clusters) < num_modes:
            remaining_clusters = [c for idx, c in enumerate(stable_clusters) if idx not in used_cluster_indices]
            remaining_clusters = sorted(remaining_clusters, key=lambda c: len(c), reverse=True)
            for c in remaining_clusters:
                if len(matched_clusters) >= num_modes:
                    break
                matched_clusters.append(c)
                
        matched_clusters = sorted(matched_clusters, key=lambda c: np.mean([p['freq'] for p in c]))
        
        for idx, c in enumerate(matched_clusters):
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
        damp_str = f" | Damping: {mode_damps[idx]*100:.2f}%" if mode_damps[idx] is not None else " | Damping: N/A (FDD fallback)"
        print(f"  Mode {idx+1}: {freq:.3f} Hz{damp_str}")

    # 4. Plot SV plot with annotated mode frequencies
    sv_plot_path = os.path.join(out_dir, "fdd_sv_plot.png")
    plt.figure(figsize=(12, 7))
    for i in range(min(3, n_channels)):
        plt.semilogy(f, singular_values[:, i], label=f"SV {i+1}", linewidth=1.5)
        
    for idx, freq in enumerate(mode_freqs):
        f_idx = np.argmin(np.abs(f - freq))
        val = singular_values[f_idx, 0]
        plt.scatter(freq, val, color='red', s=80, zorder=5)
        plt.axvline(freq, color='red', linestyle='--', alpha=0.5, linewidth=1.2)
        plt.text(freq, val * 1.5, f"Mode {idx+1}\n{freq:.3f} Hz", 
                 ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
                 
    plt.xlim(0, fs/2)  # Show full frequency range up to Nyquist (64 Hz for 128 Hz sampling)
    plt.xlabel("Frequency (Hz)", fontsize=11)
    plt.ylabel("Singular Value Amplitude (PSD)", fontsize=11)
    plt.title(f"Frequency Domain Decomposition (FDD) Singular Value Plot ({method} Modes Highlighted)", fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(sv_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> FDD Singular Value plot saved to: {sv_plot_path}")
        
    # 5. Extract and Plot Mode Shapes
    spacings = [1.82, 1.82, 1.82, 2.0, 2.0, 2.0, 2.0, 1.25, 1.25, 1.25]
    joint_positions = [0.0]
    for s in spacings:
        joint_positions.append(joint_positions[-1] + s)
    joint_positions = np.array(joint_positions)
    
    all_supports = set(zero_joints + fixed_joints)
    
    num_modes_plot = len(mode_freqs)
    if num_modes_plot == 0:
        print("No prominent mode shapes found. Skipping mode shape plot.")
        return

    fig, axes = plt.subplots(num_modes_plot, 1, figsize=(12, 3 * num_modes_plot + 1), sharex=True)
    if num_modes_plot == 1:
        axes = [axes]

    colors = ['#2980b9', '#27ae60', '#8e44ad', '#d35400', '#f39c12', '#16a085', '#2c3e50']
    mode_styles = []
    for m in range(num_modes_plot):
        c = colors[m % len(colors)]
        mode_styles.append({'color': c, 'title_color': c})

    for mode_idx, (freq, shape) in enumerate(zip(mode_freqs, mode_shapes)):
        # Use the refined frequency for plotting reference
        f_idx = np.argmin(np.abs(f - freq))
        u_vector = shape  # already normalized real mode shape
        max_val_idx = np.argmax(np.abs(u_vector))
        ref_phase = np.angle(u_vector[max_val_idx])
        rotated_u = u_vector * np.exp(-1j * ref_phase)
        real_u = np.real(rotated_u)
        
        # Normalize maximum magnitude to 1.0
        norm_factor = np.max(np.abs(real_u))
        if norm_factor > 0:
            real_u = real_u / norm_factor
            
        known_x = []
        known_y = []
        sensor_x = []
        sensor_y = []
        
        for j in range(1, 12):
            x = joint_positions[j - 1]
            if j in all_supports:
                known_x.append(x)
                known_y.append(0.0)
            elif j in sensor_joints:
                sensor_idx = sensor_joints.index(j)
                val = real_u[sensor_idx]
                known_x.append(x)
                known_y.append(val)
                sensor_x.append(x)
                sensor_y.append(val)
                
        sorted_indices = np.argsort(known_x)
        known_x = np.array(known_x)[sorted_indices]
        known_y = np.array(known_y)[sorted_indices]
        
        # Cubic spline interpolation of the mode shape
        cs = CubicSpline(known_x, known_y, bc_type='natural')
        x_fine = np.linspace(0, joint_positions[-1], 200)
        y_fine = cs(x_fine)
        
        # Plot sub-modes
        ax = axes[mode_idx]
        style = mode_styles[mode_idx]
        
        ax.plot(x_fine, y_fine, color=style['color'], linewidth=3, label=f"Mode Shape {mode_idx+1}")
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        
        # Plot supports
        for j in all_supports:
            if j < 1 or j > 11:
                continue
            x_sup = joint_positions[j - 1]
            ax.plot(x_sup, 0, marker='^', color='#2c3e50', markersize=12, zorder=5)
            
        # Plot sensor locations
        ax.scatter(sensor_x, sensor_y, color='#e74c3c', s=80, zorder=6)
        for sx, sy, sj in zip(sensor_x, sensor_y, [j for j in sensor_joints if j not in all_supports]):
            ax.text(sx, sy + 0.08 if sy >= 0 else sy - 0.08, f"J{sj}\n{sy:.2f}", ha='center', va='bottom' if sy >= 0 else 'top', fontsize=8, color='#c0392b', fontweight='bold')
            
        ax.set_title(f"Mode {mode_idx+1}: {freq:.3f} Hz", fontsize=11, fontweight='bold', color=style['title_color'], loc='left')
        ax.set_ylabel("Norm. Amplitude", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(-1.3, 1.3)
        
    axes[-1].set_xlabel("Bridge Span (meters)", fontsize=11)
    
    # Legend on first subplot
    axes[0].plot([], [], marker='^', color='#2c3e50', ls='', markersize=10, label="Supports")
    axes[0].scatter([], [], color='#e74c3c', label="Sensors")
    axes[0].legend(loc='upper right', ncol=3)
    
    fig.suptitle(f"Identified Bridge Mode Shapes ({method} Method)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    mode_plot_path = os.path.join(out_dir, "bridge_mode_shapes.png")
    plt.savefig(mode_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Mode shapes plot saved to: {mode_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Reconstruct Harness Bridge Deflections & Mode Shapes from Clean Sensor Data.")
    parser.add_argument("--clean-csv", default="Harness Bridge Data Clean.csv", help="Path to clean CSV file.")
    parser.add_argument("--zero-points", nargs="*", type=int, default=[4, 8], help="Joint numbers acting as Zero Points (default: 4 8).")
    parser.add_argument("--fixed-points", nargs="*", type=int, default=[1, 11], help="Joint numbers acting as Fixed Points (default: 1 11).")
    parser.add_argument("--cutoff", type=float, default=0.1, help="High-pass filter cutoff frequency in Hz (default: 0.1).")
    parser.add_argument("--nperseg", type=int, default=4096, help="Window size for CSD estimation in FDD (default: 4096).")
    parser.add_argument("--method", choices=["FDD","EFDD","SSI-COV"], default="FDD", help="Select modal analysis method: FDD, EFDD, or SSI-COV.")
    parser.add_argument("--num-modes", type=int, default=3, help="Number of bridge modes to extract and plot (default: 3).")
    parser.add_argument("--out-dir", default=".", help="Directory to save output files (default: current directory).")
    
    args = parser.parse_args()
    
    # Automatically compute output directory based on zero points, fixed points, num_modes, and method
    zp_str = "-".join(map(str, args.zero_points))
    fp_str = "-".join(map(str, args.fixed_points))
    auto_dir = f"z{zp_str}_f{fp_str}_n{args.num_modes}_{args.method}"
    args.out_dir = os.path.abspath(os.path.join(args.out_dir, auto_dir))
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Validate CSV file
    if not os.path.exists(args.clean_csv):
        print(f"Error: CSV file '{args.clean_csv}' not found. Please specify the path with --clean-csv.")
        return
        
    print(f"Configured Zero Points: {args.zero_points}")
    print(f"Configured Fixed Points: {args.fixed_points}")
    
    sensor_joints = [2, 3, 5, 6, 7, 9, 10]
    
    # 1. Plot bridge layout
    layout_path = os.path.join(args.out_dir, "bridge_layout.png")
    draw_bridge_layout(args.zero_points, args.fixed_points, sensor_joints, layout_path)
    
    # 2. Load clean CSV
    print(f"Loading data from {args.clean_csv}...")
    df = pd.read_csv(args.clean_csv, skiprows=25)
    fs = 128.0
    sensor_cols = [col for col in df.columns if col not in ['Time', 'ParsedTime']]
    df = df.dropna(subset=sensor_cols)
    col_to_joint = dict(zip(sensor_cols, sensor_joints))
    
    # 3. Run Modal Analysis (FDD)
    run_modal_analysis(df, fs, sensor_cols, sensor_joints, args.zero_points, args.fixed_points, args.out_dir, nperseg=args.nperseg, num_modes=args.num_modes, method=args.method)
    
    # 4. Double-integrate to get displacements
    print("\nCalculating displacements using double integration with high-pass filtering...")
    displacements = {}
    for col in sensor_cols:
        accel_data = df[col].values
        accel_dyn = accel_data - np.mean(accel_data)
        
        # Double integrate (convert g to m/s^2)
        disp = double_integrate(accel_dyn, fs, cutoff=args.cutoff)
        disp_m = disp * 9.81
        displacements[col_to_joint[col]] = disp_m
        
    # 5. Generate displacement time series plot
    t_series_path = os.path.join(args.out_dir, "displacements_time_series.png")
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(len(df)) / fs / 60.0  # in minutes
    for j in sensor_joints:
        plt.plot(time_axis, displacements[j] * 1000, label=f"Joint {j} (Sensor)", alpha=0.8)
    plt.xlabel("Time (minutes)", fontsize=11)
    plt.ylabel("Displacement (mm)", fontsize=11)
    plt.title("Reconstructed Sensor Displacements (Double Integrated & Filtered)", fontsize=13, fontweight='bold')
    plt.legend(ncol=4, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(t_series_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> Time-series plot saved to: {t_series_path}")
    
    # 6. Spatial deflection profile at maximum vibration instant
    num_steps = len(df)
    rms_disp = np.zeros(num_steps)
    for t_idx in range(num_steps):
        s_vals = [displacements[j][t_idx] for j in sensor_joints]
        rms_disp[t_idx] = np.sqrt(np.mean(np.array(s_vals)**2))
        
    max_t_idx = np.argmax(rms_disp)
    max_time_str = df['Time'].iloc[max_t_idx]
    print(f"Maximum vibration activity detected at index {max_t_idx} (Time: {max_time_str})")
    
    spacings = [1.82, 1.82, 1.82, 2.0, 2.0, 2.0, 2.0, 1.25, 1.25, 1.25]
    joint_positions = [0.0]
    for s in spacings:
        joint_positions.append(joint_positions[-1] + s)
    joint_positions = np.array(joint_positions)
    
    all_supports = set(args.zero_points + args.fixed_points)
    
    known_x = []
    known_y = []
    
    for j in range(1, 12):
        x = joint_positions[j - 1]
        if j in all_supports:
            known_x.append(x)
            known_y.append(0.0)
        elif j in sensor_joints:
            known_x.append(x)
            known_y.append(displacements[j][max_t_idx])
            
    sorted_indices = np.argsort(known_x)
    known_x = np.array(known_x)[sorted_indices]
    known_y = np.array(known_y)[sorted_indices]
    
    if np.any(np.isnan(known_y)) or np.any(np.isinf(known_y)):
        print("Deflection profile contains non-finite values. Skipping plot.")
        return

    cs = CubicSpline(known_x, known_y, bc_type='natural')
    x_fine = np.linspace(0, joint_positions[-1], 200)
    y_fine = cs(x_fine)
    
    profile_path = os.path.join(args.out_dir, "bridge_deflection_profile.png")
    plt.figure(figsize=(14, 5))
    plt.plot(x_fine, y_fine * 1000, color='#2980b9', linewidth=3, label="Reconstructed Deflection Shape")
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    for j in all_supports:
        if j < 1 or j > 11:
            continue
        x = joint_positions[j - 1]
        plt.plot(x, 0, marker='^', color='#2c3e50', markersize=15, zorder=5, label="Support" if j == list(all_supports)[0] else "")
        plt.text(x, -0.2, f"J{j}", ha='center', va='top', fontsize=9, fontweight='bold')
        
    for j in sensor_joints:
        x = joint_positions[j - 1]
        if j in all_supports:
            continue
        val = displacements[j][max_t_idx] * 1000
        plt.scatter(x, val, color='#e74c3c', s=100, zorder=6, label="Sensor Measurement" if j == sensor_joints[0] else "")
        plt.text(x, val + 0.15 if val >= 0 else val - 0.15, f"J{j}\n{val:.2f}mm", ha='center', va='bottom' if val >= 0 else 'top', fontsize=8, color='#c0392b', fontweight='bold')
        
    plt.xlabel("Bridge Span (meters)", fontsize=11)
    plt.ylabel("Deflection (mm)", fontsize=11)
    plt.title(f"Reconstructed Bridge Deflection Profile at t = {max_time_str}", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(profile_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"-> Deflection profile plot saved to: {profile_path}")
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()
