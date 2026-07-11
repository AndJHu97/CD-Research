import os
from send2trash import send2trash


def remove_duplicate_folders(reference_folder, search_folder, dry_run=True):
    """
    reference_folder:
        Folder containing the folder names to remove.

    search_folder:
        Folder tree to search for matching folders.

    dry_run:
        If True, only prints what would be moved to Recycle Bin.
        If False, moves matching folders to Recycle Bin.
    """

    # Get folder names from reference folder
    reference_names = {
        name
        for name in os.listdir(reference_folder)
        if os.path.isdir(os.path.join(reference_folder, name))
    }

    print(f"Found {len(reference_names)} reference folder names.")

    matched_count = 0

    # Walk through all folders in search_folder
    for root, dirs, files in os.walk(search_folder, topdown=False):

        for dirname in dirs:
            if dirname in reference_names:

                full_path = os.path.join(root, dirname)

                if dry_run:
                    print(f"[DRY RUN] Would move to Recycle Bin: {full_path}")
                else:
                    print(f"Moving to Recycle Bin: {full_path}")
                    send2trash(full_path)

                matched_count += 1

    print(f"\nMatched folders: {matched_count}")


if __name__ == "__main__":

    reference_folder = r"run_incorrect"  # Update this path to your reference folder
    search_folder = r"evaluation/"

    remove_duplicate_folders(
        reference_folder,
        search_folder,
        dry_run=False  # Set to False after verifying results
    )