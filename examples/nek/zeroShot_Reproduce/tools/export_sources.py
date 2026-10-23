#!/usr/bin/env python3
"""Copy only the reviewed source manifest into an empty destination."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.absolute()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"Destination must be absent or empty: {destination}")
    if destination.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("Export destination must be outside the source bundle")
    inventory = ROOT / "manifests/bundle_files.json"
    records = json.loads(inventory.read_text())["files"]
    for record in records:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Invalid source manifest path")
        source = ROOT / relative
        if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"Source changed since bundle manifest was written: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for record in records:
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / record["path"], target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
            raise RuntimeError(f"Copy verification failed: {target}")
    (destination / "manifests").mkdir(parents=True, exist_ok=True)
    shutil.copy2(inventory, destination / "manifests/bundle_files.json")
    print(f"Exported {len(records) + 1} source/manifest files to {destination}")


if __name__ == "__main__":
    main()
