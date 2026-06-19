import os
import pandas as pd
from clean_data import clean_csv
from sv_explorer import generate_explorer

def clean_csv_day9(input_path, output_path):
    print(f"Special Cleaning for Day 9: {input_path} -> {output_path}...")
    
    # Read the first 26 rows for metadata
    with open(input_path, 'r') as f:
        metadata = [f.readline() for _ in range(26)]
    
    # Remove the blank line at index 24 so that we write exactly 25 lines of metadata.
    # This aligns the cleaned CSV with the standard 25-line skiprows format expected by other tools.
    if len(metadata) > 24 and metadata[24].strip() == '':
        metadata.pop(24)
        print("  Removed blank line to reduce metadata headers to 25 lines.")
        
    # Read the data (skip 26 lines, so row 27 is header)
    df = pd.read_csv(input_path, skiprows=26)
    
    # Copy 29131:ch3 to 29128:ch3 because signal was lost on 29128:ch3
    if '29131:ch3' in df.columns and '29128:ch3' in df.columns:
        print("  Using 29131:ch3 values for 29128:ch3...")
        df['29128:ch3'] = df['29131:ch3']
    else:
        print("  Warning: Columns 29131:ch3 or 29128:ch3 not found!")
    
    # Identify sensor columns
    sensor_cols = [col for col in df.columns if col not in ['Time', 'ParsedTime']]
    
    # Drop rows where any sensor is NaN
    initial_len = len(df)
    df = df.dropna(subset=sensor_cols)
    final_len = len(df)
    print(f"  Dropped {initial_len - final_len} rows containing missing values (NaNs).")
    
    # Write metadata (now exactly 25 lines)
    with open(output_path, 'w', newline='') as f:
        for line in metadata:
            f.write(line)
            
    # Append data
    df.to_csv(output_path, mode='a', index=False)
    print("  Done cleaning Day 9.")

def main():
    tests_dir = r"c:\Users\Win10\Desktop\Harness Bridge\Tests\Tests"
    for i in range(1, 10):
        input_csv = os.path.join(tests_dir, f"day{i}.csv")
        clean_csv_path = os.path.join(tests_dir, f"day{i} Clean.csv")
        out_html = os.path.join(tests_dir, f"day{i}_sv_explorer.html")
        
        print(f"\n=== Processing Day {i} ===")
        if os.path.exists(input_csv):
            if i == 9:
                clean_csv_day9(input_csv, clean_csv_path)
            else:
                clean_csv(input_csv, clean_csv_path)
            
            # Generate the SV Explorer HTML
            print(f"Generating explorer: {clean_csv_path} -> {out_html}")
            try:
                generate_explorer(clean_csv_path, out_html)
            except Exception as e:
                print(f"Error generating explorer for Day {i}: {e}")
        else:
            print(f"File {input_csv} does not exist. Skipping.")

if __name__ == "__main__":
    main()
