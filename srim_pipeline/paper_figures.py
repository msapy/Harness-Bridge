import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from scipy.signal import welch

def plot_fig01_methodology_4panel(meth_data_dict: dict, out_dir: Path):
    """Fig 01: The 4-panel methodology steps from Day 1 Segment 0."""
    if not meth_data_dict:
        return
    
    poles_by_order = meth_data_dict['poles_by_order']
    all_poles = meth_data_dict['all_poles']
    stable_poles = meth_data_dict['stable_poles']
    cleared_poles = meth_data_dict['cleared_poles']
    clean_clusters = meth_data_dict['clean_clusters']
    final_modes = meth_data_dict['final_modes']
    orders = sorted(poles_by_order.keys())
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    titles = [
        "(a) Raw SRIM Poles",
        "(b) Stage 0: Type-3 Stable Poles (Freq & MAC)",
        "(c) Stage 1: Cleared Diagram (MPC > 0.5, $\\xi > 0$)",
        "(d) Stage 2 & 3: Clustered Modes & Inliers"
    ]
    
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlim(0, 64)
        ax.set_ylim(0, max(orders) + 2)
        ax.grid(True, linestyle='--', alpha=0.6)
        if ax in [axes[2], axes[3]]:
            ax.set_xlabel("Frequency (Hz)", fontsize=12)
        if ax in [axes[0], axes[2]]:
            ax.set_ylabel("System Order", fontsize=12)
        ax.tick_params(labelsize=11)

    for p in all_poles:
        axes[0].scatter(p["freq"], p["order"], c="grey", s=10, alpha=0.5, marker=".")
    
    for p in all_poles:
        axes[1].scatter(p["freq"], p["order"], c="lightgrey", s=10, alpha=0.3, marker=".", zorder=1)
    for p in stable_poles:
        axes[1].scatter(p["freq"], p["order"], c="royalblue", s=25, marker="+", alpha=0.8, zorder=2)
        
    for p in all_poles:
        axes[2].scatter(p["freq"], p["order"], c="lightgrey", s=10, alpha=0.3, marker=".", zorder=1)
    for p in cleared_poles:
        axes[2].scatter(p["freq"], p["order"], c="darkorange", s=30, marker="o", alpha=0.8, edgecolors='none', zorder=2)
        
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
    for p in all_poles:
        axes[3].scatter(p["freq"], p["order"], c="lightgrey", s=10, alpha=0.3, marker=".", zorder=1)
    
    color_idx = 0
    for cid, plist in clean_clusters.items():
        c = colors[color_idx % len(colors)]
        for p in plist:
            axes[3].scatter(p["freq"], p["order"], c=c, s=35, marker="o", alpha=0.7, edgecolors='none', zorder=2)
        color_idx += 1
        
    for i, fm in enumerate(final_modes):
        x_val = fm["freq"]
        n_poles = fm.get("n_poles", 0)
        axes[3].axvline(x=x_val, color='k', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)
        
        # Stagger the text vertically to prevent overlapping for closely spaced modes
        y_offset = (max(orders) + 1) if (i % 2 == 0) else (max(orders) - 5)
        axes[3].text(x_val + 0.5, y_offset, f"n={n_poles}", rotation=90, va='top', ha='left', fontsize=9, color='black', fontweight='bold', zorder=5)
        
    axes[0].legend(handles=[mpatches.Patch(color="grey", label="Raw Pole")], loc="upper right")
    axes[1].legend(handles=[mpatches.Patch(color="royalblue", label="Stable Pole")], loc="upper right")
    axes[2].legend(handles=[mpatches.Patch(color="darkorange", label="Cleared Pole (MPC)")], loc="upper right")
    handles_d = [
        mpatches.Patch(color="#e41a1c", label="Cluster Inliers"),
        plt.Line2D([0], [0], color='k', linestyle='--', linewidth=1.5, label="Final Median Freq")
    ]
    axes[3].legend(handles=handles_d, loc="upper right")
    
    plt.tight_layout()
    fig_path = out_dir / "Fig_01_Methodology_4Panel.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Fig 01] {fig_path.name}")

def compute_psd(data, fs=128.0):
    nperseg = min(1024, data.shape[0])
    freqs, Pxx = welch(data, fs=fs, axis=0, nperseg=nperseg)
    Pxx_mean = np.mean(Pxx, axis=1)
    Pxx_mean[Pxx_mean <= 0] = 1e-12
    return freqs, 10 * np.log10(Pxx_mean)

def categorize_poles(poles_by_order, clusterer):
    all_poles = [p for plist in poles_by_order.values() for p in plist]
    cat1_freq, cat2_freq_damp, cat3_freq_mac, cat4_all = [], [], [], []
    orders = sorted(poles_by_order.keys())
    for idx in range(1, len(orders)):
        n_prev, n_curr = orders[idx - 1], orders[idx]
        poles_prev = poles_by_order.get(n_prev, [])
        poles_curr = poles_by_order.get(n_curr, [])
        for p_curr in poles_curr:
            f_curr, d_curr, phi_curr = p_curr["freq"], p_curr["damping"], p_curr["shape"]
            stable_f, stable_d, stable_mac = False, False, False
            for p_prev in poles_prev:
                f_prev = p_prev["freq"]
                if f_prev == 0: continue
                if abs(f_curr - f_prev) / f_prev < clusterer.lim_f:
                    stable_f = True
                    if abs(d_curr - p_prev["damping"]) / max(1e-10, p_prev["damping"]) < clusterer.lim_xi:
                        stable_d = True
                    if (1.0 - clusterer.mac(phi_curr, p_prev["shape"])) < clusterer.lim_mac:
                        stable_mac = True
            if stable_f and stable_d and stable_mac: cat4_all.append(p_curr)
            elif stable_f and stable_mac: cat3_freq_mac.append(p_curr)
            elif stable_f and stable_d: cat2_freq_damp.append(p_curr)
            elif stable_f: cat1_freq.append(p_curr)
    return cat1_freq, cat2_freq_damp, cat3_freq_mac, cat4_all

def plot_fig02_to_04_multiday_clusters(seg_data_dict: dict, fs: float, out_dir: Path):
    if not seg_data_dict: return
    n = len(seg_data_dict)
    
    # Fig 02: Stabilization with PSD
    rows = int(np.ceil(n / 2.0))
    fig2, axes2 = plt.subplots(rows, 2, figsize=(14, rows * 5), squeeze=False)
    axes2 = axes2.flatten()
    
    # Fig 03 & 04: Clusters
    fig3, axes3 = plt.subplots(n, 2, figsize=(12, n * 4), squeeze=False)
    fig4, axes4 = plt.subplots(n, 2, figsize=(12, n * 4), squeeze=False)
    
    colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (seg_name, result) in enumerate(seg_data_dict.items()):
        # Fig 02 processing
        ax2 = axes2[i]
        ax2_twin = ax2.twinx()
        clusterer = result['clusterer']
        poles_by_order = result['poles_by_order']
        data = result['data']
        orders = list(poles_by_order.keys())
        freqs, psd = compute_psd(data, fs)
        ax2_twin.plot(freqs, psd, color='darkorange', alpha=0.3, linewidth=1.5, zorder=0)
        ax2_twin.set_ylabel("PSD (dB/Hz)", color='darkorange', fontsize=12)
        ax2_twin.tick_params(axis='y', labelcolor='darkorange')
        
        cat1, cat2, cat3, cat4 = categorize_poles(poles_by_order, clusterer)
        for p in cat1: ax2.plot(p["freq"], p["order"], 'o', color='red', markersize=3, alpha=0.6, zorder=1)
        for p in cat2: ax2.plot(p["freq"], p["order"], 'o', color='yellow', markersize=3, alpha=0.6, zorder=2)
        for p in cat3: ax2.plot(p["freq"], p["order"], 'o', markeredgecolor='blue', markerfacecolor='none', markersize=4, zorder=3)
        for p in cat4: ax2.plot(p["freq"], p["order"], 'x', color='green', markersize=4, zorder=4)
        ax2.set_title(seg_name, fontweight='bold', fontsize=12)
        ax2.set_xlim(0, fs/2)
        ax2.set_ylim(0, max(orders) + 5)
        if i >= len(axes2) - 2: ax2.set_xlabel("Frequency (Hz)", fontsize=11)
        if i % 2 == 0: ax2.set_ylabel("Model Order", fontsize=11)
        
        # Fig 03 & 04 processing
        stable = clusterer.build_stabilization(poles_by_order)
        cleared = clusterer.clear_diagram(stable)
        raw_clusters = clusterer.cluster_poles(cleared)
        clean_clusters = clusterer.remove_outliers(raw_clusters)
        
        ax3_before, ax3_after = axes3[i, 0], axes3[i, 1]
        ax4_before, ax4_after = axes4[i, 0], axes4[i, 1]
        
        for p in cleared:
            ax3_before.plot(p["freq"], p["order"], 'o', markeredgecolor='blue', markerfacecolor='none', markersize=4)
            ax4_before.plot(p["freq"], p["damping"], 'o', markeredgecolor='red', markerfacecolor='none', markersize=4)
            
        col_idx = 0
        for cid, plist in clean_clusters.items():
            c = colors[col_idx % len(colors)]
            for p in plist:
                ax3_after.plot(p["freq"], p["order"], 'o', markeredgecolor=c, markerfacecolor='none', markersize=4)
                ax4_after.plot(p["freq"], p["damping"], 'o', markeredgecolor=c, markerfacecolor='none', markersize=4)
            col_idx += 1
            
        ax3_before.set_title(f"{seg_name} Before Clustering")
        ax3_after.set_title(f"{seg_name} After Clustering")
        ax4_before.set_title(f"{seg_name} Before Clustering")
        ax4_after.set_title(f"{seg_name} After Clustering")
        
        for ax in [ax3_before, ax3_after]:
            ax.set_xlim(0, fs/2)
            ax.set_ylim(0, max(orders) + 5)
            ax.set_ylabel("Model Order")
        for ax in [ax4_before, ax4_after]:
            ax.set_xlim(0, fs/2)
            ax.set_ylim(0, 0.15)
            ax.set_ylabel("Damping Ratio")
        if i == n - 1:
            for ax in [ax3_before, ax3_after, ax4_before, ax4_after]: ax.set_xlabel("Frequency(Hz)")

    for j in range(i + 1, len(axes2)): fig2.delaxes(axes2[j])
    handles = [
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='red', label='stable pole (freq)', markersize=6),
        plt.Line2D([0],[0], marker='o', color='w', markeredgecolor='blue', markerfacecolor='none', label='stable freq. & MAC', markersize=6),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='yellow', label='stable freq. & damp.', markersize=6),
        plt.Line2D([0],[0], marker='x', color='w', markeredgecolor='green', label='stable freq. & damp. & MAC', markersize=6)
    ]
    fig2.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig2.tight_layout()
    fig2_path = out_dir / "Fig_02_Stabilization_with_PSD.png"
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"  [Fig 02] {fig2_path.name}")
    
    fig3.tight_layout()
    fig3_path = out_dir / "Fig_03_Freq_Clusters_MultiDay.png"
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"  [Fig 03] {fig3_path.name}")
    
    fig4.tight_layout()
    fig4_path = out_dir / "Fig_04_Damping_Clusters_MultiDay.png"
    fig4.savefig(fig4_path, dpi=300)
    plt.close(fig4)
    print(f"  [Fig 04] {fig4_path.name}")
