"""
Usage:
    python download_plate.py <plate_id> <barcode_platemap_csv> [<local_out_dir>]
Example:
    python download_plate.py 24277 barcode_platemap.csv .
"""

import sys
import os
import pandas as pd
import quilt3 as q3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

S3_BUCKET = "s3://cellpainting-gallery"
DATASET_PATH = "cpg0012-wawer-bioactivecompoundprofiling"
IMAGES_PATH = f"{DATASET_PATH}/broad/images/CDRP/images"
PLATEMAP_BASE = f"{DATASET_PATH}/broad/workspace/metadata/platemaps/CDRP"
PLATEMAP_SUBFOLDER = f"{PLATEMAP_BASE}/platemap"

BARCODE_CSV = os.path.join(PROJECT_ROOT, "data", "barcode_platemap.csv")
LOCAL_OUT_DIR = os.path.join(PROJECT_ROOT, "data")

# load_data.csv path pattern:
# e.g. "cpg0012-wawer-bioactivecompoundprofiling/broad/workspace/load_data_csv/CDRP/24277/load_data.csv"
LOAD_DATA_BASE = f"{DATASET_PATH}/broad/workspace/load_data_csv/CDRP"


def main():
    if len(sys.argv) < 2:
        print("Usage: download_plate.py <plate_id>")
        sys.exit(1)

    plate_id = str(sys.argv[1])

    df_barcode = pd.read_csv(BARCODE_CSV)
    df_barcode["Assay_Plate_Barcode_str"] = (
        df_barcode["Assay_Plate_Barcode"].astype(str).str.replace(",", "")
    )

    row = df_barcode.loc[df_barcode["Assay_Plate_Barcode_str"] == plate_id]
    if row.empty:
        print(f"Error: Plate ID {plate_id} not found in {BARCODE_CSV}.")
        sys.exit(1)

    platemap_name = row["Plate_Map_Name"].iloc[0]
    print(f"Plate ID {plate_id} → Platemap: {platemap_name}")

    bucket = q3.Bucket(S3_BUCKET)

    plate_image_path = f"{IMAGES_PATH}/{plate_id}/"
    local_plate_dir = os.path.join(LOCAL_OUT_DIR, plate_id)
    if not local_plate_dir.endswith("/"):
        local_plate_dir += "/"
    print(f"Downloading images from {plate_image_path} to {local_plate_dir} ...")
    bucket.fetch(plate_image_path, local_plate_dir)

    platemap_file_remote = f"{PLATEMAP_SUBFOLDER}/{platemap_name}.txt"
    platemap_file_local = os.path.join(LOCAL_OUT_DIR, f"{platemap_name}.txt")
    print(
        f"Downloading platemap from {platemap_file_remote} to {platemap_file_local} ..."
    )
    bucket.fetch(platemap_file_remote, platemap_file_local)

    load_data_remote = f"{LOAD_DATA_BASE}/{plate_id}/load_data.csv"
    load_data_local = os.path.join(LOCAL_OUT_DIR, f"load_data_{plate_id}.csv")
    print(
        f"Attempting to download load_data.csv from {load_data_remote} to {load_data_local} ..."
    )

    try:
        bucket.fetch(load_data_remote, load_data_local)
    except Exception as e:
        print(f"Warning: Could not download load_data.csv for plate {plate_id}.\n{e}")

    print("Done.")


if __name__ == "__main__":
    main()
