"""
Usage:
    python 02-create_pairs.py <plate_id>
Example:
    python 02-create_pairs.py 24277
"""

import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Input/Output paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BARCODE_CSV = os.path.join(DATA_DIR, "barcode_platemap.csv")
CHEM_ANNOT_CSV = os.path.join(DATA_DIR, "chemical_annotations.csv")


def build_abs_path(filename, plate):
    """Build path to image file in the images subdirectory"""
    return os.path.join(DATA_DIR, str(plate), "images", filename)


def process_plate(plate_id, df_barcode, df_chem):
    plate_dir = os.path.join(DATA_DIR, str(plate_id))
    load_data_csv = os.path.join(plate_dir, "load_data.csv")
    if not os.path.exists(load_data_csv):
        print(f"Error: load_data.csv not found for plate {plate_id}")
        return None

    print(f"Processing plate_id={plate_id}, file={load_data_csv}")

    # Find matching platemap
    row = df_barcode.loc[df_barcode["plate_id"] == plate_id]
    if row.empty:
        print(f"Error: No row in barcode_platemap.csv for plate {plate_id}.")
        return None

    platemap_name = row["Plate_Map_Name"].iloc[0]
    platemap_file = os.path.join(plate_dir, f"{platemap_name}.txt")

    if not os.path.exists(platemap_file):
        print(f"Error: Platemap file {platemap_file} not found.")
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
    if len(sys.argv) < 2:
        print("Usage: create_pairs.py <plate_id>")
        sys.exit(1)

    plate_id = sys.argv[1]

    if not os.path.exists(CHEM_ANNOT_CSV):
        print(f"Error: Chemical annotations file {CHEM_ANNOT_CSV} not found.")
        sys.exit(1)

    # Read input files
    df_chem = pd.read_csv(CHEM_ANNOT_CSV)
    df_barcode = pd.read_csv(BARCODE_CSV)
    df_barcode["plate_id"] = (
        df_barcode["Assay_Plate_Barcode"].astype(str).str.replace(",", "")
    )

    # Process single plate
    df_plate = process_plate(plate_id, df_barcode, df_chem)
    if df_plate is None:
        print("Failed to process plate. Exiting.")
        sys.exit(1)

    # Save plate-specific pairs file
    output_csv = os.path.join(DATA_DIR, str(plate_id), f"pairs_{plate_id}.csv")
    df_plate.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} with {len(df_plate)} rows.")


if __name__ == "__main__":
    main()
