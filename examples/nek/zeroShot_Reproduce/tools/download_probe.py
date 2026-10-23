#!/usr/bin/env python3
"""Download one pinned HF configuration to validate routing, only with --download."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    hf = json.loads((ROOT / "dependencies.json").read_text())["hf"]
    filename = hf["wing_environment"] + "/environment_config.yaml"
    directory = ROOT / "validation/download-probe"
    print(f"Source: {hf['repository']}@{hf['revision']}/{filename}")
    print(f"Destination: {directory / filename}")
    if not args.download:
        print("PLAN ONLY; no network access or directory creation")
        return
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo_id=hf["repository"], repo_type="dataset",
                               revision=hf["revision"], filename=filename,
                               local_dir=str(directory), cache_dir=str(ROOT / ".cache/huggingface/hub")))
    if not path.resolve().is_relative_to(directory.resolve()):
        raise RuntimeError("Downloaded file escaped the sandbox probe directory")
    print(json.dumps({"file": str(path), "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
