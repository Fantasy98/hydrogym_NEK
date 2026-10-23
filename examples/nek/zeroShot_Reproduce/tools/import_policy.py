#!/usr/bin/env python3
"""Copy and verify the reference model/VecNormalize pair without deserializing it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import load_settings


def verify(path, record):
    if path.stat().st_size != record["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise RuntimeError(f"Policy checksum mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--install", action="store_true", help="Copy files; default verifies source only")
    args = parser.parse_args()
    records = json.loads((ROOT / "dependencies.json").read_text())["deployment_policy"]["files"]
    for record in records:
        verify(args.source / record["name"], record)
    if args.install:
        target_dir = Path(load_settings("deploy")["WING_TCF_CHECKPOINT_DIR"])
        if target_dir.resolve() == args.source.resolve():
            raise RuntimeError("Source and destination must differ")
        # Validate existing files before writing anything; never overwrite them.
        for record in records:
            target = target_dir / record["name"]
            if target.is_symlink():
                raise RuntimeError(f"Policy destination cannot be a symlink: {target}")
            if target.exists():
                verify(target, record)
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".policy-import-", dir=target_dir) as temp:
            for record in records:
                target = target_dir / record["name"]
                if target.exists():
                    continue
                staged = Path(temp) / record["name"]
                shutil.copy2(args.source / record["name"], staged)
                verify(staged, record)
                staged.rename(target)
        for record in records:
            verify(target_dir / record["name"], record)
        print(f"Installed the verified model/VecNormalize pair in {target_dir}")
    else:
        print("Verified reference pair; no files copied (use --install to copy)")


if __name__ == "__main__":
    main()
