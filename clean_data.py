import pandas as pd
import sys
import os

def clean_csv(input_path, output_path):
    print(f"Cleaning {input_path} -> {output_path}...")
    
    # Read the first 25 rows for metadata
    with open(input_path, 'r') as f:
        metadata = [f.readline() for _ in range(25)]
        
    # Read the data
    df = pd.read_csv(input_path, skiprows=25)
    
    # Identify sensor columns
    sensor_cols = [col for col in df.columns if col not in ['Time', 'ParsedTime']]
    
    # Drop rows where any sensor is NaN
    initial_len = len(df)
    df = df.dropna(subset=sensor_cols)
    final_len = len(df)
    print(f"  Dropped {initial_len - final_len} rows containing missing values (NaNs).")
    
    # Write metadata
    with open(output_path, 'w', newline='') as f:
        for line in metadata:
            f.write(line)
            
    # Append data
    df.to_csv(output_path, mode='a', index=False)
    print("  Done.")

files = [
    'SensorConnectData21052026.csv',
    'SensorConnectData_day3.csv',
    'SensorConnectData_day4.csv'
]

for file in files:
    out_file = file.replace('.csv', ' Clean.csv')
    clean_csv(file, out_file)
