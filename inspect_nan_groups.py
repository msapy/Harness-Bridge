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
    
    # Find contiguous groups of intact rows
    intact = df[active_cols].notna().all(axis=1)
    group_ids = (~intact).cumsum()
    intact_groups = df[intact].groupby(group_ids[intact])
    
    print(f"=== Day {day} groups ===")
    for name, group in intact_groups:
        if len(group) < 1000: # skip tiny noise groups
            continue
        start_idx = group.index[0]
        end_idx = group.index[-1]
        
        # Calculate means
        means = [group[c].mean() for c in active_cols]
        stds = [group[c].std() for c in active_cols]
        
        # Check if all means are around 1.0 (between 0.8 and 1.2)
        on_bridge = all(0.8 <= m <= 1.2 for m in means)
        
        print(f"  Lines {header_idx + 2 + start_idx} to {header_idx + 2 + end_idx} | Length: {len(group)}")
        print(f"    Means: {['{:.4f}'.format(m) for m in means]}")
        print(f"    Stds:  {['{:.4f}'.format(s) for s in stds]}")
        print(f"    On Bridge: {on_bridge}")
