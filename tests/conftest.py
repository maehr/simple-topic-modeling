"""Pytest configuration shared by all tests.

`src/` is not an installed package: it is mounted flat into a Streamlit
runtime (both the local Poetry environment and, via stlite, an in-browser
Pyodide runtime). We can't add a `pyproject.toml` package path (Phase 2/3
of this task is not allowed to touch `pyproject.toml`), so tests put
`src/` on `sys.path` themselves, the same way both runtimes see it.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
