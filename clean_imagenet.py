import os
import shutil


def clean_validation_folders(train_path, val_path):
    train_folders = set(os.listdir(train_path))
    val_folders = set(os.listdir(val_path))
    folders_to_remove = val_folders - train_folders

    for folder in folders_to_remove:
        folder_path = os.path.join(val_path, folder)
        print(f"Removing {folder_path}")
        shutil.rmtree(folder_path)

    print(f"Removed {len(folders_to_remove)} folders from validation set")
    print(f"Validation folders remaining: {len(val_folders - folders_to_remove)}")


if __name__ == "__main__":
    train_path = "datasets/imagenet/train.X1"
    val_path = "datasets/imagenet/val.X"

    print(f"This will remove validation folders that don't exist in {train_path}")
    response = input("Do you want to proceed? (y/n): ")

    if response.lower() == "y":
        clean_validation_folders(train_path, val_path)
    else:
        print("Operation cancelled")
