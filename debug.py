import pandas as pd
import os

root_dir = r"c:\Users\Win10\Desktop\Harness Bridge"
files = ["Harness Bridge Data Clean.csv", "SensorConnectData21052026 Clean.csv"]
for f in files:
    path = os.path.join(root_dir, f)
    if os.path.exists(path):
        df = pd.read_csv(path, skiprows=25)
        print(f"\n=== {f} info ===")
        print(df.shape)
        print(df.isna().sum())
