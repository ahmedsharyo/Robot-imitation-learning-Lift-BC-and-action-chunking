#!/usr/bin/env python3
"""Download robomimic Lift low-dimensional dataset using the official utility."""
import argparse, subprocess, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/lift.hdf5")
    p.add_argument("--dataset", default="lift/mg")
    args = p.parse_args()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import robomimic
        from robomimic.utils.dataset_registry import get_dataset_path
    except Exception as e:
        sys.exit(f"robomimic not installed. Install with `pip install -e '.[sim]'`. Details: {e}")
    try:
        resolved = Path(get_dataset_path(args.dataset))
    except Exception as e:
        sys.exit(f"Could not resolve official registry key {args.dataset!r}: {e}")
    if not resolved.exists():
        sys.exit(f"robomimic resolved {args.dataset!r} to {resolved}, but the dataset is not present. Use the official robomimic download_dataset.py utility for the selected release, then copy the HDF5 file to {out}.")
    import shutil
    shutil.copy2(resolved, out)
    print(f"Copied official robomimic dataset {args.dataset} from {resolved} to {out}")

if __name__ == "__main__": main()
