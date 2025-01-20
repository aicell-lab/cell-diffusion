"""
Usage:
    python 04-create_webdataset.py [--samples_per_shard 100] [--train_frac 0.8] [--val_frac 0.1]

Creates a WebDataset from the pairs.csv file, with train/val/test splits.
Each sample contains a 5-channel image stack and metadata about the compound.
"""

import argparse
import os

import numpy as np
import pandas as pd
import tifffile
import webdataset as wds
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Input/Output paths
PAIRS_CSV = os.path.join(DATA_DIR, "pairs.csv")
WDS_DIR = os.path.join(DATA_DIR, "webdataset")


def load_5ch_stack(row):
    """Load and stack the 5 channel images into a single (5, H, W) array."""
    channels = ["channelDNA", "channelER", "channelRNA", "channelAGP", "channelMito"]
    stack = []

    for channel in channels:
        img = tifffile.imread(row[channel]).astype(np.float32)

        # Basic normalization to 0-1 range
        if img.max() > 0:
            img = img / img.max()

        stack.append(img)

    return np.stack(stack, axis=0)


def create_writer(out_dir, shard_id):
    """Create a WebDataset TarWriter for a shard."""
    fname = os.path.join(out_dir, f"shard_{shard_id:06d}.tar")
    return wds.TarWriter(fname)


def write_split(df_split, split_dir, samples_per_shard):
    """Write a dataframe split to WebDataset shards."""
    os.makedirs(split_dir, exist_ok=True)

    shard_id = 0
    samples_in_shard = 0
    writer = create_writer(split_dir, shard_id)

    for idx, row in tqdm(
        df_split.iterrows(),
        total=len(df_split),
        desc=f"Creating {os.path.basename(split_dir)} split",
    ):
        try:
            # Load and stack images
            stack_5ch = load_5ch_stack(row)

            # Create sample key
            sample_key = f"{row['plate_id']}_{row['well']}_{row['site']}"

            # Prepare metadata
            meta = {
                "broad_id": row["BROAD_ID"] if pd.notna(row["BROAD_ID"]) else None,
                "compound_name": row["CPD_NAME"] if pd.notna(row["CPD_NAME"]) else None,
                "compound_type": (
                    row["CPD_NAME_TYPE"] if pd.notna(row["CPD_NAME_TYPE"]) else None
                ),
                "smiles": row["CPD_SMILES"] if pd.notna(row["CPD_SMILES"]) else None,
                "plate_id": row["plate_id"],
                "well": row["well"],
                "site": int(row["site"]),
            }

            # Write sample to shard
            writer.write(
                {"__key__": sample_key, "images.npy": stack_5ch, "meta.json": meta}
            )

            samples_in_shard += 1

            # Start new shard if needed
            if samples_in_shard >= samples_per_shard:
                writer.close()
                shard_id += 1
                writer = create_writer(split_dir, shard_id)
                samples_in_shard = 0

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    writer.close()
    print(f"Created {shard_id + 1} shards in {split_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create WebDataset from pairs.csv")
    parser.add_argument(
        "--samples_per_shard", type=int, default=100, help="Number of samples per shard"
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.8, help="Fraction of data for training"
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.1, help="Fraction of data for validation"
    )
    args = parser.parse_args()

    # Read pairs.csv
    print(f"Reading {PAIRS_CSV}...")
    df = pd.read_csv(PAIRS_CSV)
    print(f"Found {len(df)} samples")

    # Shuffle data
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Calculate split sizes
    n_total = len(df)
    n_train = int(n_total * args.train_frac)
    n_val = int(n_total * args.val_frac)

    # Split data
    df_train = df[:n_train]
    df_val = df[n_train : n_train + n_val]
    df_test = df[n_train + n_val :]

    print(f"Splits: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # Create WebDataset for each split
    for split_name, split_df in [
        ("train", df_train),
        ("val", df_val),
        ("test", df_test),
    ]:
        split_dir = os.path.join(WDS_DIR, split_name)
        write_split(split_df, split_dir, args.samples_per_shard)

    print("Done!")


if __name__ == "__main__":
    main()
