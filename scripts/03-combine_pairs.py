"""
Usage:
    python 03-combine_pairs.py
Example:
    python 03-combine_pairs.py
"""

import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "pairs.csv")


def main():
    # Find all pairs_*.csv files in plate subdirectories
    pairs_files = []
    for plate_dir in os.listdir(DATA_DIR):
        if not os.path.isdir(os.path.join(DATA_DIR, plate_dir)):
            continue

        pairs_file = os.path.join(DATA_DIR, plate_dir, f"pairs_{plate_dir}.csv")
        if os.path.exists(pairs_file):
            pairs_files.append(pairs_file)

    if not pairs_files:
        print("No pairs_*.csv files found in plate directories.")
        return

    print(f"Found {len(pairs_files)} pairs files to combine.")

    # Read and combine all pairs files
    dfs = []
    for pairs_file in pairs_files:
        print(f"Reading {pairs_file}...")
        df = pd.read_csv(pairs_file)
        dfs.append(df)

    # Concatenate all dataframes
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined {len(dfs)} files, total rows: {len(df_combined)}")

    # Save combined file
    df_combined.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote combined pairs file to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
