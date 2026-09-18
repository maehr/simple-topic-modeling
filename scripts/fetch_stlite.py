#!/usr/bin/env python3
"""Fetch the pinned stlite runtime into assets/stlite/build.

The stlite runtime is not committed. This script downloads it from the npm
registry and verifies it against the integrity hash in scripts/stlite.lock.json.
Both the GitHub Pages workflow and a local run use this script, so the deployed
site and a developer machine always get the same bytes.

Usage:
    python3 scripts/fetch_stlite.py                 # fetch the pinned version
    python3 scripts/fetch_stlite.py --keep-maps     # keep the source maps
    python3 scripts/fetch_stlite.py --update 1.9.2  # repin to a new version
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_FILE = REPO_ROOT / "scripts" / "stlite.lock.json"
TARGET_DIR = REPO_ROOT / "assets" / "stlite" / "build"
REGISTRY = "https://registry.npmjs.org"


def read_lock() -> dict:
    return json.loads(LOCK_FILE.read_text())


def registry_metadata(package: str, version: str) -> dict:
    url = f"{REGISTRY}/{package}/{version}"
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def integrity_of(payload: bytes) -> str:
    digest = hashlib.sha512(payload).digest()
    return "sha512-" + base64.b64encode(digest).decode()


def download(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def extract(payload: bytes, target: Path, keep_maps: bool) -> int:
    """Extract package/build/** into target. Return the number of files written."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    written = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            # npm tarballs put everything under a single "package/" directory.
            parts = Path(member.name).parts
            if len(parts) < 3 or parts[0] != "package" or parts[1] != "build":
                continue
            relative = Path(*parts[2:])
            if ".." in relative.parts or relative.is_absolute():
                raise ValueError(f"refusing unsafe path in archive: {member.name}")
            if not keep_maps and relative.suffix == ".map":
                continue

            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            destination.write_bytes(source.read())
            written += 1
    return written


def update_pin(package: str, version: str) -> None:
    metadata = registry_metadata(package, version)
    lock = read_lock()
    lock["version"] = metadata["version"]
    lock["integrity"] = metadata["dist"]["integrity"]

    payload = download(metadata["dist"]["tarball"])

    # Verify before the hash is written to the lock file. Otherwise a registry
    # response whose metadata and payload disagree would be recorded as
    # trusted, and every later build would accept it.
    declared = metadata["dist"]["integrity"]
    actual = integrity_of(payload)
    if actual != declared:
        raise SystemExit(
            "REFUSING TO REPIN: the tarball does not match the integrity hash "
            f"the registry declares.\n  declared: {declared}\n  actual:   {actual}"
        )

    wheels = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        wheels = [
            Path(name).name
            for name in archive.getnames()
            if name.startswith("package/build/wheels/") and name.endswith(".whl")
        ]
    for wheel in wheels:
        if wheel.startswith("streamlit-"):
            parts = wheel.split("-")
            lock["bundled_streamlit"] = parts[1]
            lock["bundled_python"] = parts[2].replace("cp", "").replace("3", "3.", 1)

    LOCK_FILE.write_text(json.dumps(lock, indent="\t") + "\n")
    print(f"Repinned {package} to {lock['version']}.")
    print(f"Bundled Streamlit is {lock.get('bundled_streamlit')}.")
    print("Check that pyproject.toml allows that Streamlit version.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-maps", action="store_true", help="keep .map files")
    parser.add_argument("--update", metavar="VERSION", help="repin to this version")
    args = parser.parse_args()

    if args.update:
        lock = read_lock()
        update_pin(lock["package"], args.update)
        return 0

    lock = read_lock()
    package, version, expected = lock["package"], lock["version"], lock["integrity"]
    print(f"Fetching {package}@{version} ...")

    metadata = registry_metadata(package, version)
    payload = download(metadata["dist"]["tarball"])

    actual = integrity_of(payload)
    if actual != expected:
        print("INTEGRITY CHECK FAILED", file=sys.stderr)
        print(f"  expected: {expected}", file=sys.stderr)
        print(f"  actual:   {actual}", file=sys.stderr)
        return 1
    print(f"Integrity verified: {expected[:24]}...")

    count = extract(payload, TARGET_DIR, args.keep_maps)
    entry = TARGET_DIR / "stlite.js"
    if not entry.is_file():
        print(f"ERROR: {entry} is missing after extraction.", file=sys.stderr)
        return 1

    size = sum(f.stat().st_size for f in TARGET_DIR.rglob("*") if f.is_file())
    where = TARGET_DIR.relative_to(REPO_ROOT)
    print(f"Wrote {count} files to {where} ({size // 1024 // 1024} MB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
