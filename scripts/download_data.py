#!/usr/bin/env python3
"""Download robomimic Lift low-dimensional dataset using the official utility."""
import argparse
import os
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download robomimic demonstration dataset.")
    parser.add_argument("--output", default="data/lift.hdf5", help="Target output HDF5 path.")
    parser.add_argument("--task", default="lift", help="Task name (default: lift).")
    parser.add_argument("--dataset_type", default="mg", help="Dataset type: ph (proficient-human) or mg (machine-generated).")
    parser.add_argument("--hdf5_type", default="low_dim_sparse", help="HDF5 type (e.g., low_dim_sparse, low_dim_dense, low_dim).")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from robomimic import DATASET_REGISTRY
        import robomimic.utils.file_utils as FileUtils
    except ImportError as e:
        sys.exit(f"robomimic not installed. Please install robomimic. Details: {e}")

    task_reg = DATASET_REGISTRY.get(args.task, {})
    type_reg = task_reg.get(args.dataset_type, {})

    # Match requested hdf5_type or fallback gracefully
    entry = type_reg.get(args.hdf5_type)
    if not entry:
        # Check available types
        for candidate in ["low_dim_sparse", "low_dim_dense", "low_dim", "raw"]:
            if candidate in type_reg and type_reg[candidate].get("url"):
                entry = type_reg[candidate]
                args.hdf5_type = candidate
                break

    if not entry or not entry.get("url"):
        sys.exit(
            f"Could not find download URL for task='{args.task}', "
            f"dataset_type='{args.dataset_type}', hdf5_type='{args.hdf5_type}' in DATASET_REGISTRY."
        )

    url = entry["url"]
    filename = url.split("/")[-1]
    download_dir = out.parent / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded_file = download_dir / filename

    if not downloaded_file.exists():
        print(f"Downloading {args.task}/{args.dataset_type}/{args.hdf5_type} from {url} ...")
        FileUtils.download_url(url=url, download_dir=str(download_dir))
    else:
        print(f"Using cached download at {downloaded_file}")

    if not downloaded_file.exists():
        sys.exit(f"Downloaded file not found at {downloaded_file}")

    shutil.copy2(downloaded_file, out)
    print(f"Successfully staged dataset to {out}")


if __name__ == "__main__":
    main()
