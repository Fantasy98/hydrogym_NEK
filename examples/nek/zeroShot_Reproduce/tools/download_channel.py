#!/usr/bin/env python3
"""Download the pinned large-channel package using only Python's standard library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.config import load_settings
from tools import channel_restarts


def checked_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Invalid HF manifest path: {relative}")
    target = root / relative
    for path in (root, *[root / Path(*relative.parts[:i]) for i in range(1, len(relative.parts) + 1)]):
        if path.is_symlink():
            raise RuntimeError(f"Refusing a symlink in the download destination: {path}")
    if target.exists() and (not target.is_file() or target.stat().st_nlink != 1):
        raise RuntimeError(f"Expected an independent regular file: {target}")
    return target


def verify_file(path, record):
    before = path.stat()
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {record['bytes']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha256.update(block)
            blob.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed during verification: {path}")
    if after.st_size != record["bytes"]:
        raise RuntimeError(f"Size mismatch: {path}")
    expected = record.get("sha256") or record["git_blob_sha1"]
    actual = sha256.hexdigest() if "sha256" in record else blob.hexdigest()
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch: {path}")
    return sha256.hexdigest()


def download_package(manifest, destination):
    destination = Path(destination).absolute()
    records = manifest["files"]
    expected = {record["path"] for record in records}
    if len(expected) != len(records):
        raise ValueError("Duplicate HF manifest paths")
    # Validate existing files before writing; never repair them by overwriting.
    existing = {}
    for record in records:
        path = checked_path(destination, record["path"])
        if path.exists():
            existing[record["path"]] = verify_file(path, record)
    if destination.exists():
        extra = {p.relative_to(destination).as_posix() for p in destination.rglob("*")
                 if p.is_file() or p.is_symlink()} - expected
        if extra:
            raise RuntimeError(f"Unexpected files in package: {sorted(extra)}")
    outcomes = []
    base_url = ("https://huggingface.co/datasets/" + manifest["repository"] + "/resolve/"
                + manifest["revision"] + "/" + manifest["environment"] + "/")
    for record in records:
        path = checked_path(destination, record["path"])
        digest = existing.get(record["path"])
        if digest is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {record['path']} ({record['bytes']} bytes)", flush=True)
            with tempfile.NamedTemporaryFile(prefix=".download-", dir=path.parent) as temporary:
                with urlopen(base_url + quote(record["path"], safe="/"), timeout=120) as response:
                    count = 0
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        count += len(block)
                        if count > record["bytes"]:
                            raise RuntimeError(f"Response exceeds expected size: {record['path']}")
                        temporary.write(block)
                temporary.flush()
                digest = verify_file(Path(temporary.name), record)
                # Atomic publication without replacing a file created meanwhile.
                # Closing the temporary file removes its name, leaving nlink=1.
                os.link(temporary.name, path)
            checked_path(destination, record["path"])
        outcomes.append({**record, "actual_sha256": digest, "resolved_path": str(path.resolve()),
                         "acquisition": "verified_existing" if record["path"] in existing else "downloaded"})
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download and install; default prints a plan only")
    args = parser.parse_args()
    env = load_settings("train")
    manifest = json.loads((ROOT / "manifests/channel_hf.json").read_text())
    for setting, key in (("ZS_HF_REPO", "repository"), ("ZS_HF_REVISION", "revision"), ("TCF_ENV_NAME", "environment")):
        if env[setting] != manifest[key]:
            raise RuntimeError(f"{setting} differs from the reviewed HF manifest")
    restart_manifest = json.loads(channel_restarts.MANIFEST.read_text())
    downloaded_restarts = {r["path"][len("restart_files/"):]: r for r in manifest["files"]
                           if r["path"].startswith("restart_files/")}
    if set(downloaded_restarts) != {r["path"] for r in restart_manifest["files"]}:
        raise RuntimeError("HF restart inventory differs from the original manifest")
    for record in restart_manifest["files"]:
        remote = downloaded_restarts[record["path"]]
        if (remote["bytes"], remote.get("sha256")) != (record["bytes"], record["sha256"]):
            raise RuntimeError(f"HF restart identity differs from the original: {record['path']}")
    destination = Path(env["ZS_DATA_ROOT"]) / "hf" / manifest["environment"]
    restart_destination = Path(env["TCF_RESTART_SOURCE"])
    plan = {"mode": "download" if args.download else "plan_only", "repository": manifest["repository"],
            "revision": manifest["revision"], "files": len(manifest["files"]), "bytes": manifest["total_bytes"],
            "package_destination": str(destination), "restart_destination": str(restart_destination)}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.download:
        return
    outcomes = download_package(manifest, destination)
    channel_restarts.install(destination / "restart_files", restart_destination, restart_manifest["files"])
    report = Path(env["ZS_ROOT"]) / "validation/channel-download.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({**plan, "verified_files": outcomes, "original_restart_manifest_matched": True}, indent=2) + "\n")
    print(f"Verified all {len(outcomes)} HF files; installed all 36 exact original restart files.")
    print(f"Evidence: {report}")


if __name__ == "__main__":
    main()
