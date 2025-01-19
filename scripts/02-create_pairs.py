"""
Usage:
    python create_pairs.py [<chemical_annotations_csv>]
Example:
    python create_pairs.py chemical_annotations.csv
"""

import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Input/Output paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BARCODE_CSV = os.path.join(DATA_DIR, "barcode_platemap.csv")
DEFAULT_CHEM_ANNOT = os.path.join(DATA_DIR, "chemical_annotations.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "pairs.csv")


def build_abs_path(filename, plate):
    return os.path.join(DATA_DIR, str(plate), filename)


def process_plate(plate_id, load_data_csv, df_barcode, df_chem):
    print(f"Processing plate_id={plate_id}, file={load_data_csv}")

    # Find matching platemap
    row = df_barcode.loc[df_barcode["plate_id"] == plate_id]
    if row.empty:
        print(
            f"Warning: No row in barcode_platemap.csv for plate {plate_id}. Skipping."
        )
        return None

    platemap_name = row["Plate_Map_Name"].iloc[0]
    platemap_file = os.path.join(DATA_DIR, f"{platemap_name}.txt")

    if not os.path.exists(platemap_file):
        print(f"Warning: Platemap file {platemap_file} not found. Skipping.")
        return None

    # Read the load_data.csv
    df_load = pd.read_csv(load_data_csv, header=None, sep=",", skiprows=1).rename(
        columns={
            0: "channelDNA_filename",
            2: "channelER_filename",
            4: "channelRNA_filename",
            6: "channelAGP_filename",
            8: "channelMito_filename",
            10: "plate_id",
            11: "well",
            12: "site",
        }
    )

    # Build absolute file paths for each channel
    channels = ["DNA", "ER", "RNA", "AGP", "Mito"]
    for channel in channels:
        df_load[f"channel{channel}"] = df_load.apply(
            lambda row: build_abs_path(
                row[f"channel{channel}_filename"], row["plate_id"]
            ),
            axis=1,
        )

    # Read and merge platemap
    df_map = pd.read_csv(platemap_file, sep="\t")
    df_load["well"] = df_load["well"].str.upper()
    df_merged = pd.merge(
        df_load, df_map, left_on="well", right_on="well_position", how="left"
    ).rename(columns={"broad_sample": "BROAD_ID"})

    # Merge with chemical annotations
    df_final = pd.merge(df_merged, df_chem, on="BROAD_ID", how="left")

    return df_final[
        [
            "plate_id",
            "well",
            "site",
            "channelDNA",
            "channelER",
            "channelRNA",
            "channelAGP",
            "channelMito",
            "BROAD_ID",
            "CPD_NAME",
            "CPD_NAME_TYPE",
            "CPD_SMILES",
        ]
    ]


def main():
    # Parse command line arguments
    chem_annot_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHEM_ANNOT

    if not os.path.exists(chem_annot_csv):
        print(f"Error: Chemical annotations file {chem_annot_csv} not found.")
        sys.exit(1)

    # Read input files
    df_chem = pd.read_csv(chem_annot_csv)
    df_barcode = pd.read_csv(BARCODE_CSV)
    df_barcode["plate_id"] = (
        df_barcode["Assay_Plate_Barcode"].astype(str).str.replace(",", "")
    )

    # Process all load_data files
    all_dfs = []
    for fname in os.listdir(DATA_DIR):
        if not fname.startswith("load_data_") or not fname.endswith(".csv"):
            continue

        plate_id = fname.replace("load_data_", "").replace(".csv", "")
        load_data_csv = os.path.join(DATA_DIR, fname)

        df_plate = process_plate(plate_id, load_data_csv, df_barcode, df_chem)
        if df_plate is not None:
            all_dfs.append(df_plate)

    if not all_dfs:
        print("No plates processed. Exiting.")
        return

    # Merge and save results
    df_final = pd.concat(all_dfs, ignore_index=True)
    print(f"Merged total rows: {len(df_final)}")

    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} with {len(df_final)} rows.")


if __name__ == "__main__":
    main()
