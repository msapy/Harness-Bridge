"""
phone_sv_explorer.py
Generates an interactive HTML SV Mode Shape Explorer for Phone Data.
"""

import numpy as np
from scipy.signal import csd
import json
import os
import argparse

# Re-use the data loading from the new phone tool
import phone_bridge_tool as pbt
# Re-use the HTML builder from the original explorer tool
from sv_explorer import build_html

def compute_sv_data(data, fs, nperseg=4096):
    n_channels = data.shape[1]
    print(f"Computing cross-spectral density matrix ({n_channels}x{n_channels})...")
    f, _ = csd(data[:, 0], data[:, 0], fs=fs, nperseg=nperseg)
    n_freqs = len(f)

    G = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    for i in range(n_channels):
        for j in range(n_channels):
            _, temp = csd(data[:, i], data[:, j], fs=fs, nperseg=nperseg)
            G[:, i, j] = temp

    print(f"Computing SVD at {n_freqs} frequency bins...")
    sv = np.zeros((n_freqs, n_channels))
    shapes = np.zeros((n_freqs, n_channels))

    for k in range(n_freqs):
        U, S, VH = np.linalg.svd(G[k])
        sv[k] = S
        u = U[:, 0]
        max_idx = np.argmax(np.abs(u))
        u_rot = u * np.exp(-1j * np.angle(u[max_idx]))
        real_u = np.real(u_rot)
        norm = np.max(np.abs(real_u))
        shapes[k] = real_u / norm if norm > 0 else real_u

    return f, sv, shapes


def generate_explorer(phones_dir, out_path, axis='z', zero_joints=None, fixed_joints=None, nperseg=4096):
    if zero_joints is None:
        zero_joints = [4, 8]
    if fixed_joints is None:
        fixed_joints = [1, 11]

    print("Loading phone data...")
    data, fs, device_names = pbt.load_phone_data(phones_dir, axis=axis)
    n_channels = data.shape[1]
    
    f, sv, shapes = compute_sv_data(data, fs, nperseg=nperseg)

    sensor_joints = pbt.SENSOR_JOINTS
    jp = pbt.joint_positions()

    all_supports = sorted(set(zero_joints + fixed_joints))

    # Knot table
    knots = []
    for j in range(1, 12):
        pos = jp[j - 1]
        is_sup = j in all_supports
        s_idx = sensor_joints.index(j) if j in sensor_joints else None
        knots.append({'j': j, 'pos': pos, 'isSupport': is_sup, 'sensorIdx': s_idx})

    n_sv = min(3, n_channels)
    payload = {
        'freqs': [round(float(x), 4) for x in f],
        'sv': [[float(sv[k, i]) for i in range(n_sv)] for k in range(len(f))],
        'shapes': [[round(float(v), 5) for v in shapes[k]] for k in range(len(f))],
        'knots': knots,
        'fs': fs,
        'totalSpan': jp[-1],
        'sensorJoints': sensor_joints,
        'nSV': n_sv,
    }

    data_json = json.dumps(payload, separators=(',', ':'))
    # Use the HTML template from the original sv_explorer
    html = build_html(data_json)
    # Update the title to reflect Phone Data
    html = html.replace("🌉 Harness Bridge · SV Mode Shape Explorer", "📱 Phone Bridge · SV Mode Shape Explorer")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print(f"\nInteractive phone explorer saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate interactive SV Mode Shape Explorer HTML for Phone Data")
    parser.add_argument("--phones-dir", default="Phones")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--zero-points", nargs="*", type=int, default=[4, 8])
    parser.add_argument("--fixed-points", nargs="*", type=int, default=[1, 11])
    parser.add_argument("--nperseg", type=int, default=2048)
    parser.add_argument("--out", default="phone_sv_explorer.html")
    args = parser.parse_args()

    if not os.path.isdir(args.phones_dir):
        print(f"Error: Directory '{args.phones_dir}' not found.")
        return

    generate_explorer(args.phones_dir, args.out, args.axis, args.zero_points, args.fixed_points, args.nperseg)

if __name__ == "__main__":
    main()
