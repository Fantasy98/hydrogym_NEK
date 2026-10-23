#!/usr/bin/env python3
"""Verify the exact original 36 channel restart files; optionally install a copy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/channel_restarts.json"


def sha256(path):
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f"File changed during verification: {path}")
    return digest.hexdigest()


def verify(directory, records, independent=False):
    directory = Path(directory)
    expected = {record["path"] for record in records}
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    if actual != expected:
        raise RuntimeError(f"Restart inventory mismatch at {directory}: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    for record in records:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid manifest path")
        path = directory / relative
        parents = [directory / Path(*relative.parts[:i]) for i in range(len(relative.parts))]
        if independent and (path.is_symlink() or any(p.is_symlink() for p in parents) or path.stat().st_nlink != 1):
            raise RuntimeError(f"Destination must contain independent regular files: {path}")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Restart checksum mismatch: {path}")


def install(source, destination, records):
    source, destination = Path(source).resolve(), Path(destination).absolute()
    verify(source, records)
    if destination.exists() or destination.is_symlink():
        verify(destination, records, independent=True)
        if destination.resolve() == source:
            raise RuntimeError("Source and destination must be different directories")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".channel-import-", dir=destination.parent) as temp:
        staged = Path(temp) / "channel_restarts"
        staged.mkdir()
        for record in records:
            target = staged / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / record["path"], target)
        verify(staged, records, independent=True)
        if destination.exists() or destination.is_symlink():
            raise RuntimeError("Destination appeared during import; refusing to replace it")
        staged.rename(destination)
    verify(destination, records, independent=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing restart root; default verifies destination")
    parser.add_argument("--destination", type=Path, default=ROOT / "data/channel_restarts")
    parser.add_argument("--install", action="store_true", help="Copy verified source into destination")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if args.install:
        if args.source is None:
            parser.error("--install requires --source")
        install(args.source, args.destination, manifest["files"])
        directory = args.destination
    else:
        directory = args.source if args.source is not None else args.destination
        verify(directory, manifest["files"], independent=args.source is None)
    print(f"Verified {len(manifest['files'])} files, {manifest['total_bytes']} bytes: {directory}")


if __name__ == "__main__":
    main()
