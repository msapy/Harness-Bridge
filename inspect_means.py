import pandas as pd
import os

folder = r"C:\Users\Win10\Desktop\HarNess Bridge Test Campaign\Tests"

for day in range(1, 11):
    fpath = os.path.join(folder, f"day{day}.csv")
    if not os.path.exists(fpath):
        continue
        
    # Find header
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
    
    print(f"--- Day {day} entire means ---")
    for c in cols:
        mean_val = df[c].mean()
        std_val = df[c].std()
        print(f"  {c}: mean={mean_val:.4f}, std={std_val:.4f}")
