import pandas as pd
import numpy as np
import os

folder = r"C:\Users\Win10\Desktop\HarNess Bridge Test Campaign\Tests"

for day in range(1, 11):
    fpath = os.path.join(folder, f"day{day}.csv")
    if not os.path.exists(fpath):
        continue
        
    header_idx = None
    with open(fpath, 'r') as f:
        for idx, line in enumerate(f):
            if line.strip().startswith("Time,"):
                header_idx = idx
                break
    if header_idx is None:
        continue
        
    df = pd.read_csv(fpath, skiprows=header_idx)
    if '29131:ch3' in df.columns:
        df['29128:ch3'] = df['29131:ch3']
        df = df.drop(columns=['29131:ch3'])
        
    cols = [c for c in df.columns if c not in ['Time', 'ParsedTime']]
    active_cols = [c for c in cols if not df[c].isna().all()]
    
    # Intact check (no NaNs)
    intact = df[active_cols].notna().all(axis=1)
    
    # On-bridge check (value close to 1g, say between 0.85g and 1.15g)
    # We check this for all active sensors
    on_bridge = df[active_cols].apply(lambda x: (x >= 0.85) & (x <= 1.15)).all(axis=1)
    
    # Combine both checks
    valid = intact & on_bridge
    
    # Find contiguous groups of valid rows
    group_ids = (~valid).cumsum()
    valid_groups = df[valid].groupby(group_ids[valid])
    
    print(f"=== Day {day} valid on-bridge groups ===")
    
    largest_group_len = 0
    largest_group_info = None
    
    for name, group in valid_groups:
        if len(group) < 100: # skip tiny groups
            continue
        start_idx = group.index[0]
        end_idx = group.index[-1]
        
        if len(group) > largest_group_len:
            largest_group_len = len(group)
            largest_group_info = (start_idx, end_idx)
            
    if largest_group_info:
        s_idx, e_idx = largest_group_info
        start_line = header_idx + 2 + s_idx
        end_line = header_idx + 2 + e_idx
        print(f"  Largest block: lines {start_line} to {end_line} | Length: {largest_group_len}")
        print(f"  Timeframe: {df.loc[s_idx, 'Time']} -> {df.loc[e_idx, 'Time']}")
    else:
        print("  No valid block found.")
