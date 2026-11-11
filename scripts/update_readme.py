"""Regenerate the registry table in README.md from the registry.

    python scripts/update_readme.py           # rewrite README.md
    python scripts/update_readme.py --check   # exit 1 if README.md is out of date
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saturncube._docs import splice  # noqa: E402

path = ROOT / "README.md"
old = path.read_text(encoding="utf-8")
new = splice(old)
if "--check" in sys.argv:
    if new != old:
        print("README.md is out of date; run: python scripts/update_readme.py")
        sys.exit(1)
    print("README.md is up to date")
else:
    path.write_text(new, encoding="utf-8")
    print("README.md updated" if new != old else "README.md already up to date")
