import pandas as pd
import numpy as np

fpath = r"c:\Users\Win10\Desktop\Harness Bridge\Tests\Tests\day2.csv"
df = pd.read_csv(fpath, skiprows=25)
cols = [c for c in df.columns if c not in ['Time', 'ParsedTime']]

print("--- Block A (0 to 404266) ---")
df_a = df.iloc[0:404267]
for c in cols:
    print(f"  {c}: mean={df_a[c].mean():.4f}")

print("\n--- Block B (437911 to 842177) ---")
df_b = df.iloc[437911:842178]
for c in cols:
    print(f"  {c}: mean={df_b[c].mean():.4f}")
