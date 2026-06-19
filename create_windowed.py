import sys

with open('multi_sv_explorer.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Extract the build_html function string
start_idx = code.find('def build_html(data_json):')
end_idx = code.find('def main():')
build_html_code = code[start_idx:end_idx]

# Replace title
build_html_code = build_html_code.replace('Multi-CSV SV Explorer', 'Windowed SV Explorer')

prefix = '''"""
windowed_sv_explorer.py
Generates an interactive HTML SV Mode Shape Explorer dividing a dataset into time windows.
"""

import pandas as pd
import numpy as np
from scipy.signal import csd
import json
import os
import argparse

def compute_sv_data_from_df(df_chunk, nperseg=4096, fs=128.0):
    sensor_cols = [col for col in df_chunk.columns if col not in ['Time', 'ParsedTime']]
    n_channels = len(sensor_cols)

    f, _ = csd(df_chunk[sensor_cols[0]].values, df_chunk[sensor_cols[0]].values, fs=fs, nperseg=nperseg)
    n_freqs = len(f)

    G = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    for i in range(n_channels):
        for j in range(n_channels):
            _, temp = csd(df_chunk[sensor_cols[i]].values, df_chunk[sensor_cols[j]].values, fs=fs, nperseg=nperseg)
            G[:, i, j] = temp

    sv = np.zeros((n_freqs, n_channels))
    shapes = np.zeros((n_freqs, n_channels))

    for k in range(n_freqs):
        if np.any(np.isnan(G[k])) or np.any(np.isinf(G[k])):
            continue
        try:
            U, S, VH = np.linalg.svd(G[k])
            sv[k] = S
            u = U[:, 0]
            max_idx = np.argmax(np.abs(u))
            u_rot = u * np.exp(-1j * np.angle(u[max_idx]))
            real_u = np.real(u_rot)
            norm = np.max(np.abs(real_u))
            shapes[k] = real_u / norm if norm > 0 else real_u
        except np.linalg.LinAlgError:
            continue

    return f, sv, shapes, n_channels

def generate_windowed_explorer(csv_path, out_path, window_sec, zero_joints=None, fixed_joints=None, nperseg=4096):
    if zero_joints is None:
        zero_joints = [4, 8]
    if fixed_joints is None:
        fixed_joints = [1, 11]

    fs = 128.0
    chunk_size = int(window_sec * fs)

    print(f"Loading sensor data from {csv_path}...")
    df = pd.read_csv(csv_path, skiprows=25)
    sensor_cols = [col for col in df.columns if col not in ['Time', 'ParsedTime']]
    
    datasets = []
    global_f = None
    
    total_rows = len(df)
    n_chunks = total_rows // chunk_size
    if total_rows % chunk_size > 0:
        n_chunks += 1
        
    print(f"Dividing into {n_chunks} windows of ~{window_sec} seconds ({chunk_size} samples each).")
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        df_chunk = df.iloc[start_idx:end_idx]
        
        if len(df_chunk) < nperseg:
            print(f"Skipping window {i+1} as it has fewer samples ({len(df_chunk)}) than nperseg ({nperseg}).")
            continue
            
        start_sec = start_idx / fs
        end_sec = end_idx / fs
        name = f"Window {i+1} ({start_sec:.0f}s - {end_sec:.0f}s)"
        print(f"Computing SV for {name} ({len(df_chunk)} samples)...")
        
        f, sv, shapes, n_channels = compute_sv_data_from_df(df_chunk, nperseg=nperseg, fs=fs)
        if global_f is None:
            global_f = f
            
        n_sv = min(3, n_channels)
        datasets.append({
            'id': i,
            'name': name,
            'sv': [[float(sv[k, j]) for j in range(n_sv)] for k in range(len(f))],
            'shapes': [[round(float(v), 5) for v in shapes[k]] for k in range(len(f))],
            'nSV': n_sv
        })

    sensor_joints = [2, 3, 5, 6, 7, 9, 10]
    spacings = [1.82, 1.82, 1.82, 2.0, 2.0, 2.0, 2.0, 1.25, 1.25, 1.25]
    jp = [0.0]
    for s in spacings:
        jp.append(round(jp[-1] + s, 4))

    all_supports = sorted(set(zero_joints + fixed_joints))

    knots = []
    for j in range(1, 12):
        pos = jp[j - 1]
        is_sup = j in all_supports
        s_idx = sensor_joints.index(j) if j in sensor_joints else None
        knots.append({'j': j, 'pos': pos, 'isSupport': is_sup, 'sensorIdx': s_idx})

    payload = {
        'freqs': [round(float(x), 4) for x in global_f],
        'datasets': datasets,
        'knots': knots,
        'fs': fs,
        'totalSpan': jp[-1],
        'sensorJoints': sensor_joints,
    }

    data_json = json.dumps(payload, separators=(',', ':'))
    html = build_html(data_json)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print(f"\\nInteractive windowed explorer saved to: {out_path}")
    print("  Open this file in your browser to explore mode shapes!")

'''

suffix = '''
def main():
    parser = argparse.ArgumentParser(description="Generate interactive Windowed SV Explorer HTML")
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument("--window-sec", type=float, default=3600.0, help="Window size in seconds")
    parser.add_argument("--zero-points", nargs="*", type=int, default=[4, 8])
    parser.add_argument("--fixed-points", nargs="*", type=int, default=[1, 11])
    parser.add_argument("--nperseg", type=int, default=4096)
    parser.add_argument("--out", default="windowed_sv_explorer.html")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: '{args.csv}' not found.")
        return

    generate_windowed_explorer(args.csv, args.out, args.window_sec, args.zero_points, args.fixed_points, args.nperseg)

if __name__ == "__main__":
    main()
'''

with open('windowed_sv_explorer.py', 'w', encoding='utf-8') as f:
    f.write(prefix + build_html_code + suffix)

print("Successfully created windowed_sv_explorer.py")
