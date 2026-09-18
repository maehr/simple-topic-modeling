#!/usr/bin/env python3
"""Rebuild the browser-safe pyLDAvis wheel shipped in assets/dist/.

Why this script exists
-----------------------
The upstream pyLDAvis 3.4.1 wheel on PyPI cannot be installed as-is in a
Pyodide/micropip browser environment:

1. It vendors a full copy of pip and setuptools under
   ``pyLDAvis/lib/python3.11/site-packages/``. That is an upstream packaging
   bug (those tools have no business being inside an application wheel), and
   shipping them to a browser bloats the download for no reason.
2. Its metadata declares ``gensim`` and ``numexpr`` as required dependencies.
   Neither package can be installed in Pyodide, and neither is needed for the
   scikit-learn code path this app uses (``pyLDAvis.lda_model``, not
   ``pyLDAvis.gensim_models``). If those Requires-Dist lines stay, micropip
   refuses to install the wheel at all.

This script downloads the real upstream wheel, verifies it against a pinned
SHA-256 (supply-chain check), strips the vendored pip/setuptools tree and the
gensim-specific modules, drops only the two offending Requires-Dist lines
from METADATA (every other requirement, including ``pandas (>=2.0.0)``, is
left exactly as upstream wrote it), regenerates a spec-compliant RECORD file,
and writes the result to assets/dist/pyLDAvis-3.4.1-py3-none-any.whl.

The output is byte-identical across runs: every ZipInfo entry gets a fixed
timestamp and a fixed compression level, and entries are written in a stable
sorted order.

Usage
-----
    python3 scripts/build_pyldavis_wheel.py

Standard library only. No third-party dependencies.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

# --- Pinned inputs (the supply-chain check) ---------------------------------

WHEEL_URL = (
    "https://files.pythonhosted.org/packages/6b/5a/"
    "66364c6799f2362bfb9b7100bc1ce6ffcdfe7f17e8d2e85a591bfe427643/"
    "pyLDAvis-3.4.1-py3-none-any.whl"
)
EXPECTED_SOURCE_SHA256 = (
    "8a525b187d2d0fad7304ef66bee8d0f25dfccce6a595f1be675a9017fa4d3c7f"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "assets" / "dist" / "pyLDAvis-3.4.1-py3-none-any.whl"

# Fixed timestamp so the output zip is byte-identical across runs.
FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)
COMPRESSION_LEVEL = 6

# Requires-Dist lines to drop from METADATA. gensim and numexpr cannot be
# installed in Pyodide and are not needed by the scikit-learn code path
# (pyLDAvis.lda_model) that this app uses.
DROPPED_REQUIRES_DIST_NAMES = {"gensim", "numexpr"}

# Path prefixes to drop entirely: the vendored pip/setuptools tree shipped
# inside pyLDAvis/lib/, which is an upstream packaging bug and must never
# reach a browser bundle.
DROPPED_PATH_PREFIXES = ("pyLDAvis/lib/",)

# Individual files to drop: gensim-integration modules that pull in gensim,
# which cannot be installed in Pyodide and is unused by this app.
DROPPED_FILES = {"pyLDAvis/gensim.py", "pyLDAvis/gensim_models.py"}

# Junk files that must never ship, regardless of where they show up.
JUNK_BASENAMES = {".DS_Store", "Thumbs.db"}


def download_source_wheel() -> bytes:
    """Download the upstream wheel and verify it against the pinned hash."""
    print(f"Downloading {WHEEL_URL}", file=sys.stderr)
    with urllib.request.urlopen(WHEEL_URL) as response:
        data = response.read()

    actual = hashlib.sha256(data).hexdigest()
    if actual != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            "REFUSING TO PROCEED: downloaded pyLDAvis wheel does not match "
            "the pinned SHA-256.\n"
            f"  expected: {EXPECTED_SOURCE_SHA256}\n"
            f"  actual:   {actual}\n"
            "This is the supply-chain integrity check. Do not bypass it; "
            "investigate why the upstream artifact changed instead."
        )
    print(f"Verified source wheel SHA-256: {actual}", file=sys.stderr)
    return data


def should_drop(name: str) -> bool:
    if name.endswith("/"):
        # Directory entries: drop if they live under a dropped prefix.
        return any(name.startswith(p) for p in DROPPED_PATH_PREFIXES)
    if any(name.startswith(p) for p in DROPPED_PATH_PREFIXES):
        return True
    if name in DROPPED_FILES:
        return True
    if Path(name).name in JUNK_BASENAMES:
        return True
    return False


def rewrite_metadata(data: bytes) -> bytes:
    """Drop only the gensim/numexpr Requires-Dist lines. Leave everything
    else, including the pandas constraint, exactly as upstream wrote it."""
    text = data.decode("utf-8")
    lines = text.splitlines(keepends=True)

    kept_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Requires-Dist:"):
            # e.g. "Requires-Dist: gensim" or "Requires-Dist: numexpr (>=2.8.4)"
            match = re.match(r"Requires-Dist:\s*([A-Za-z0-9_.\-]+)", stripped)
            if match and match.group(1).lower() in DROPPED_REQUIRES_DIST_NAMES:
                continue
        kept_lines.append(line)

    return "".join(kept_lines).encode("utf-8")


def record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def build_wheel(source_bytes: bytes) -> bytes:
    src_zip = zipfile.ZipFile(io.BytesIO(source_bytes))

    # name -> bytes, for every entry we keep (directories excluded; wheels
    # don't need explicit directory entries and dropping them keeps things
    # simple and deterministic).
    kept_files: dict[str, bytes] = {}
    dist_info_dir = None

    for info in src_zip.infolist():
        name = info.filename
        if name.endswith("/"):
            continue  # directory entry, nothing to keep
        if should_drop(name):
            continue

        contents = src_zip.read(info)

        if name.endswith(".dist-info/METADATA"):
            contents = rewrite_metadata(contents)

        if name.endswith(".dist-info/RECORD"):
            # We regenerate RECORD from scratch below; drop the original.
            dist_info_dir = name.rsplit("/", 1)[0]
            continue

        kept_files[name] = contents

    if dist_info_dir is None:
        # Fall back to detecting the dist-info dir from any remaining entry.
        for name in kept_files:
            if ".dist-info/" in name:
                dist_info_dir = name.split(".dist-info/")[0] + ".dist-info"
                break
    if dist_info_dir is None:
        raise SystemExit("Could not locate .dist-info directory in source wheel")

    record_path = f"{dist_info_dir}/RECORD"

    # Build RECORD: one line per kept file, plus a trailing line for RECORD
    # itself with empty hash and size fields, per the wheel spec.
    record_lines = []
    for name in sorted(kept_files):
        contents = kept_files[name]
        record_lines.append(f"{name},{record_hash(contents)},{len(contents)}")
    record_lines.append(f"{record_path},,")
    record_bytes = ("\n".join(record_lines) + "\n").encode("utf-8")

    kept_files[record_path] = record_bytes

    # Write the output zip deterministically: fixed timestamps, fixed
    # compression level, stable sorted entry order.
    out_buffer = io.BytesIO()
    with zipfile.ZipFile(out_buffer, "w", zipfile.ZIP_DEFLATED) as out_zip:
        for name in sorted(kept_files):
            zi = zipfile.ZipInfo(filename=name, date_time=FIXED_DATE_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            out_zip.writestr(zi, kept_files[name], compresslevel=COMPRESSION_LEVEL)

    return out_buffer.getvalue()


def main() -> None:
    source_bytes = download_source_wheel()
    output_bytes = build_wheel(source_bytes)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(output_bytes)

    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Output SHA-256: {output_sha256}")


if __name__ == "__main__":
    main()
