"""Build the Rust bridge (crates/sml-py-bridge) and place it in the package.

Run from python/:  uv run python scripts/build_bridge.py

The compiled module is git-ignored; CI and local test runs build it fresh, so
it always matches the Rust code in this checkout.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "python" / "src" / "spodeian_ml"


def main() -> int:
    subprocess.run(
        ["cargo", "build", "--release", "-p", "sml-py-bridge"], cwd=ROOT, check=True
    )
    release = ROOT / "target" / "release"
    if sys.platform == "win32":
        built, dest = release / "spodeian_py_bridge.dll", PKG / "spodeian_py_bridge.pyd"
    elif sys.platform == "darwin":
        built, dest = release / "libspodeian_py_bridge.dylib", PKG / "spodeian_py_bridge.so"
    else:
        built, dest = release / "libspodeian_py_bridge.so", PKG / "spodeian_py_bridge.so"
    shutil.copy2(built, dest)
    print(f"copied {built.name} -> {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
